import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
from models.yolo import Detect

import numpy as np

import tensorrt as trt
from yolov3trt import common
TRT_LOGGER = trt.Logger()

output_shapes = [(1, 3, 12, 20, 34), (1, 3, 24, 40, 34), (1, 3, 48, 80, 34)]
stride = [32, 16, 8]
anchors = [
  [116,90, 156,198, 373,326],  # P5/32
  [30,61, 62,45, 59,119],  # P4/16
  [10,13, 16,30, 33,23]
]
nc = 29  # number of classes
no = nc + 5  # number of outputs per anchor
nl = len(anchors)  # number of detection layers
na = len(anchors[0]) // 2  # number of anchors
grid = [torch.zeros(1)] * nl  # init grid
a = torch.tensor(anchors).float().view(nl, -1, 2)
anchor_grid = a.clone().view(nl, 1, -1, 1, 1, 2)  # shape(nl,1,na,1,1,2)
'''
tensor([[[[[[116.,  90.]]],
          [[[156., 198.]]],
          [[[373., 326.]]]]],
        [[[[[ 30.,  61.]]],
          [[[ 62.,  45.]]],
          [[[ 59., 119.]]]]],
        [[[[[ 10.,  13.]]],
          [[[ 16.,  30.]]],
          [[[ 33.,  23.]]]]]])
'''
anchors = a / torch.tensor(stride).view(3,1,1)
'''
tensor([[[ 3.62500,  2.81250],
         [ 4.87500,  6.18750],
         [11.65625, 10.18750]],

        [[ 1.87500,  3.81250],
         [ 3.87500,  2.81250],
         [ 3.68750,  7.43750]],

        [[ 1.25000,  1.62500],
         [ 2.00000,  3.75000],
         [ 4.12500,  2.87500]]], device='cuda:0', dtype=torch.float16)
'''

def postprocessor(trt_outputs):
    trt_outputs1 = [trt_outputs[1], trt_outputs[0], trt_outputs[2]]
    trt_outputs = trt_outputs1
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    x = [torch.from_numpy(output).cuda() for output in trt_outputs]
    z = []
    for i in range(3):
        bs, _, ny, nx, _ = output_shapes[i]  # x(bs,255,20,20) to x(bs,3,20,20,85)

        grid = Detect._make_grid(nx, ny).to(x[i].device)

        y = x[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid.to(x[i].device)) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i].to(x[i].device)  # wh
        z.append(y.view(bs, -1, no))

    return torch.cat(z, 1)


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    with open("weights/yolov5.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        # img = torch.ones((1, 3, 384, 640)).cuda() * 0.2
        # pred = model(img, augment=opt.augment)[0]

        inputs[0].host = img.detach().cpu().numpy()
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        pred = postprocessor(trt_outputs)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()

        # Update all models
        # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt.weights, opt.weights)
