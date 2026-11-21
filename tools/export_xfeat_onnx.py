"""
Export the XFeat network to ONNX for the C++ front-end (ONNX Runtime, also on Android).

The exported graph takes a grayscale image (1, 1, H, W) with H and W multiples of 32 and pixel values in [0, 255],
and returns:
  descriptors  (1, 64, H/8, W/8)  L2-normalized dense descriptors
  heatmap      (1, 1, H, W)       keypoint probability (softmax over the 8x8 cells, rearranged to full resolution)
  reliability  (1, 1, H/8, W/8)   reliability map
The rest of detectAndCompute (NMS, scores, top-k, descriptor sampling) is implemented in C++.

    pixi run python tools/export_xfeat_onnx.py      # -> models/xfeat.onnx
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'accelerated_features'))
from modules.xfeat import XFeat


class XFeatDense(nn.Module):
    def __init__(self, xfeat: XFeat):
        super().__init__()
        self.xfeat = xfeat
    
    def forward(self, image):
        descriptors, logits, reliability = self.xfeat.net(image)
        return F.normalize(descriptors, dim=1), self.xfeat.get_kpts_heatmap(logits), reliability


def unfold2d_exportable(x, ws=2):
    """XFeatModel._unfold2d for a single-channel input: identical to pixel_unshuffle, which (unlike Tensor.unfold)
    can be exported to ONNX with dynamic height and width."""
    return F.pixel_unshuffle(x, ws)


if __name__ == '__main__':
    output = os.path.join(os.path.dirname(__file__), '..', 'models', 'xfeat.onnx')
    xfeat = XFeat()
    xfeat.dev = torch.device('cpu')
    xfeat.net = xfeat.net.cpu().eval()
    
    sample = torch.rand(1, 1, 64, 96)
    assert torch.equal(xfeat.net._unfold2d(sample, ws=8), unfold2d_exportable(sample, ws=8))
    xfeat.net._unfold2d = unfold2d_exportable
    model = XFeatDense(xfeat).eval()
    
    example = torch.rand(1, 1, 480, 736) * 255
    torch.onnx.export(model, (example,), output, input_names=['image'], output_names=['descriptors', 'heatmap', 'reliability'],
                      dynamic_axes={'image': {2: 'height', 3: 'width'}, 'descriptors': {2: 'h8', 3: 'w8'},
                                    'heatmap': {2: 'height', 3: 'width'}, 'reliability': {2: 'h8', 3: 'w8'}},
                      opset_version=17, dynamo=False)
    
    # Check against PyTorch on a different resolution
    import onnxruntime
    session = onnxruntime.InferenceSession(output, providers=['CPUExecutionProvider'])
    test = torch.rand(1, 1, 480, 640) * 255
    with torch.no_grad(): reference = [t.numpy() for t in model(test)]
    exported = session.run(None, {'image': test.numpy()})
    for name, a, b in zip(['descriptors', 'heatmap', 'reliability'], reference, exported):
        print(f'{name:12s} {a.shape}  max |difference| {np.abs(a - b).max():.2e}')
    print(f'Saved {os.path.abspath(output)} ({os.path.getsize(output) / 1e6:.1f} MB)')
