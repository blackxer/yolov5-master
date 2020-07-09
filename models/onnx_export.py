"""Exports a pytorch *.pt model to *.onnx format

Usage:
    $ export PYTHONPATH="$PWD" && python models/onnx_export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse

import onnx

from models.common import *
from utils import google_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../weights/last.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    print(opt)

    # Parameters
    f = opt.weights.replace('.pt', '.onnx')  # onnx filename
    img = torch.zeros((opt.batch_size, 3, 384, 640))  # image size, (1, 3, 320, 192) iDetection

    # Load pytorch model
    google_utils.attempt_download(opt.weights)
    model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
    model.eval()
    model.fuse()

    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    _ = model(img)  # dry run
    pred = torch.onnx._export(model, img, f, verbose=True, opset_version=11)  # output_names=['classes', 'boxes']
    '''
    -3.70608e-01, -1.13203e-01, -5.24490e-01
    -9.06645e-01, -8.91624e-01, -2.03106e-01
    -1.27915e-01, -9.55773e-03,  1.64982e-01
    '''
    print("over")
    # Check onnx model
    # model = onnx.load(f)  # load onnx model
    # onnx.checker.check_model(model)  # check onnx model
    # print(onnx.helper.printable_graph(model.graph))  # print a human readable representation of the graph
    # print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
