import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
import sys
sys.path.append("/media/zw/DL/ly/workspace/project04/yolov5-master")
# import tensorrt as trt
# from yolov3trt import common

# TRT_LOGGER = trt.Logger()

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(Conv, self).__init__()
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # padding
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    import tensorrt as trt
    from yolov3trt import common
    TRT_LOGGER = trt.Logger()

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1,3,320,320]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main(dummy_input):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    import tensorrt as trt
    from yolov3trt import common
    TRT_LOGGER = trt.Logger()

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = "upsample-1.6.0.onnx"
    engine_file_path = "upsample-1.6.0.trt"
    # Download a dog image and save it to the following file path:

    image = dummy_input.numpy()

    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        inputs[0].host = image
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    return trt_outputs

if __name__ == "__main__":

    image = cv2.imread("test.jpg")[:,:,::-1] # 1,3,320,320
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as
    image = np.array(image, dtype=np.float32, order='C')
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    torch_model = Focus(c1=3,c2=64,k=3)
    torch_model.eval()
    # # torch.save(torch_model, "origin.pt")
    # torch_model = torch.load("origin.pt")
    output = torch_model(image)

    # onnx 1.5.0 版本的转换
    # torch_out = torch.onnx._export(torch_model, image, 'upsample-1.5.0.onnx', verbose=True) # 1,3,640,640
    # onnx 1.6.0 版本的转换
    torch_out = torch.onnx._export(torch_model, image, 'upsample-1.6.0.onnx', verbose=True, opset_version=11)
    # np.testing.assert_almost_equal(output.data.cpu().numpy(), torch_out.data.cpu().numpy(), decimal=3)
    #
    # torch_out = torch_out.squeeze().data.cpu().numpy()
    # torch_out = np.transpose(torch_out, [1, 2, 0])[:,:,::-1]
    # cv2.imwrite("torch_out.jpg", torch_out)

    trt_outputs = main(image)[0].reshape(1, 64, 160, 160)
    # print(trt_outputs)
    np.testing.assert_almost_equal(output.data.cpu().numpy(), trt_outputs, decimal=3)
    #
    # trt_outputs = trt_outputs.squeeze()
    # trt_outputs = np.transpose(trt_outputs, [1, 2, 0])[:,:,::-1]
    # cv2.imwrite("trt_out.jpg", trt_outputs)




