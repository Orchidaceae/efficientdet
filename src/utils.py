import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
        else:
            self.mean = mean
        if std is None:
            self.std = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32)
        else:
            self.std = std
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def forward(self, boxes, deltas):
        pred_boxes = _bbox_forward(self.std, self.mean, boxes, deltas)
        return pred_boxes

@torch.jit.script
def _bbox_forward(std, mean, boxes, deltas):
    widths = boxes[:, :, 2] - boxes[:, :, 0]
    heights = boxes[:, :, 3] - boxes[:, :, 1]
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0] * std[0] + mean[0]
    dy = deltas[:, :, 1] * std[1] + mean[1]
    dw = deltas[:, :, 2] * std[2] + mean[2]
    dh = deltas[:, :, 3] * std[3] + mean[3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

    return pred_boxes

class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = torch.tensor([0.5, 1, 2])
        if scales is None:
            self.scales = torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shapes = _get_image_shape(self.pyramid_levels, image)

        all_anchors = torch.zeros((0, 4), dtype=torch.float32)
        # if isinstance(self.ratios, np.ndarray):
        #     self.ratios = torch.tensor(self.ratios)

        # if isinstance(self.scales, np.ndarray):
        #     self.scales = torch.tensor(self.scales)

        
        all_anchors = _generate_shifted_anchors(all_anchors, self.pyramid_levels, self.sizes, self.ratios, self.scales, image_shapes, self.strides)
        
        anchors = torch.unsqueeze(all_anchors, 0)

        # if torch.cuda.is_available():
        #     anchors = anchors.cuda()
        return anchors

def generate_anchors(base_size=16,
                    ratios=torch.tensor([0.5, 1, 2]), 
                    scales=torch.tensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])):
    # type: (int, Tensor, Tensor) -> Tensor

    num_anchors = ratios.shape[0] * scales.shape[0]
    anchors = torch.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * scales.repeat(2, ratios.shape[0]).t() # torch repeat does the same as numpy tile
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = torch.sqrt(areas / torch.repeat_interleave(ratios, scales.shape[0]))
    anchors[:, 3] = anchors[:, 2] * torch.repeat_interleave(ratios, scales.shape[0])
    anchors[:, 0::2] -= (anchors[:, 2] * 0.5).repeat(2, 1).t()
    anchors[:, 1::2] -= (anchors[:, 3] * 0.5).repeat(2, 1).t()

    return anchors


def compute_shape(image_shape, pyramid_levels):
    image_shape = torch.tensor(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def shift(shape, stride, anchors):
    # type: (Tensor, int, Tensor) -> Tensor
    shift_x = (torch.arange(0, shape[1]) + 0.5) * stride
    shift_y = (torch.arange(0, shape[0]) + 0.5) * stride
    # indexing needs to be xy (numpy default is cartesian a.k.a xy), while torch default is ij (matrix indexing), pytorch 1.4 only implements ij-indexing for meshgrid (later versions of the api includes indexing parameter!)
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
    # the xy-indexing is instead achieved by transposing each coordinate vector
    shift_x = shift_x.t()
    shift_y = shift_y.t()

    # Torch flatten should be equal to numpy ravel
    shifts = torch.stack((
        torch.flatten(shift_x), torch.flatten(shift_y),
        torch.flatten(shift_x), torch.flatten(shift_y)
    ), dim=0).t()

    A = anchors.size()[0]
    K = shifts.size()[0]
    # torch permute has the same function as numpy transpose for 3d tensors
    all_anchors = (torch.reshape(anchors, (1, A, 4)) + torch.reshape(shifts, (1, K, 4)).permute(1, 0, 2))
    all_anchors = torch.reshape(all_anchors, (K * A, 4))

    return all_anchors

@torch.jit.script
def _get_image_shape(pyramid_levels, image):
    # type: (List[int], Tensor) -> List[Tensor]
    image_shape = image.shape[2:]
    # import pdb; pdb.set_trace()
    image_shape: torch.Tensor = torch.tensor(image_shape)
    image_shapes: List[torch.Tensor] = []
    for i in range(len(pyramid_levels)):
        x = pyramid_levels[i]
        t = (image_shape + (2 ** x - 1))
        #import pdb; pdb.set_trace()
        t = torch.floor(torch.div(t,(2 ** x))) # floor division
        t = t.to(torch.int)
        image_shapes.append(t)
    return image_shapes

@torch.jit.script
def _generate_shifted_anchors(all_anchors,
                            pyramid_levels,
                            sizes,
                            ratios,
                            scales,
                            image_shapes,
                            strides):
    # type: (Tensor, List[int], List[int], Tensor, Tensor, List[Tensor], List[int]) -> Tensor      
    for idx in range(len(pyramid_levels)):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = torch.cat((all_anchors, shifted_anchors), dim=0)
    return all_anchors