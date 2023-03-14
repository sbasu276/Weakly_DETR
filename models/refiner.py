import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import box_ops
import numpy as np


def batch_box_area(boxes):
	return (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])

def batch_box_iou(boxes1, boxes2):
    """
    Args:
        boxes1: Tensor of shape [batch_size, num_boxes1, 4]
        boxes2: Tensor of shape [batch_size, num_boxes2, 4]

    Returns:
        iou: Tensor of shape [batch_size, num_boxes1, num_boxes2] containing pairwise IoU values
    """
    boxes1 = box_ops.box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_ops.box_cxcywh_to_xyxy(boxes2)
    area1 = batch_box_area(boxes1)  # [batch_size, num_boxes1]
    area2 = batch_box_area(boxes2)  # [batch_size, num_boxes2]
    lt = torch.max(boxes1[:, :, None, :2], boxes2[:, None, :, :2])  # [batch_size, num_boxes1, num_boxes2, 2]
    rb = torch.min(boxes1[:, :, None, 2:], boxes2[:, None, :, 2:])  # [batch_size, num_boxes1, num_boxes2, 2]
    wh = (rb - lt).clamp(min=0)  # [batch_size, num_boxes1, num_boxes2, 2]
    inter = wh[:, :, :, 0] * wh[:, :, :, 1]  # [batch_size, num_boxes1, num_boxes2]
    union = area1[:, :, None] + area2[:, None, :] - inter
    iou = inter / union
    return iou


def select_proposals(boxes, cls_prob, im_labels, device, lambda_gt=0.5, bg_sep=True):

    # if cls_prob have the background logit, separate it
    if bg_sep:
        cls_prob = cls_prob[:, :, 1:]
    
    orig_size = cls_prob.size()
    #  avoiding NaNs.
    eps = 1e-9
    cls_prob = cls_prob.clone().clamp(eps, 1 - eps)

    batch_size = im_labels.size(0)
    max_val, max_cls_probs_idx = torch.max(cls_prob[:, :, im_labels-1], dim=1)
    max_cls_probs_idx = torch.diagonal(max_cls_probs_idx, 0)
    max_boxes = torch.gather(boxes, 1, max_cls_probs_idx.view(batch_size, 1, 1).expand(batch_size, 1, 4))
    
    overlaps = batch_box_iou(max_boxes, boxes)
    overlaps = overlaps.view(batch_size, -1)
    ind = overlaps>lambda_gt
    
    labels = np.zeros((orig_size[0], orig_size[1], orig_size[2]+1))
    for i in range(batch_size):
        labels[i, ind[i].cpu().numpy(), im_labels[i].cpu().numpy()] = 1
        labels[i, ~ind[i].cpu().numpy(), 0] = 1
    
    labels = torch.tensor(labels)
    cls_weights = torch.diagonal(max_val, 0).unsqueeze(-1).expand(orig_size[:2])

    return {'labels': labels.to(device), 'cls_weights': cls_weights}


class RefinementAgents(nn.Module):
    def __init__(self, hidden_dim, num_classes, refiner_depth=3):
        super().__init__()
        self.refine_score = []
        self.refiner_depth = refiner_depth
        for i in range(self.refiner_depth):
            self.refine_score.append(nn.Linear(hidden_dim, num_classes))
        self.refine_score = nn.ModuleList(self.refine_score)
        self._init_weights()
        #self.r0 = nn.Linear(hidden_dim, num_classes)
        #self.r1 = nn.Linear(hidden_dim, num_classes)
        #self.r2 = nn.Linear(hidden_dim, num_classes)

    def _init_weights(self):
        for i_refine in range(self.refiner_depth):
            nn.init.normal_(self.refine_score[i_refine].weight, std=0.01)
            nn.init.constant_(self.refine_score[i_refine].bias, 0)

    def forward(self, x):
        refine_score = [refine(x) for refine in self.refine_score]
        #s0 = self.r0(x)
        #s1 = self.r1(x)#F.softmax(t0, dim=1)
        #s2 = self.r2(x)#, dim=1)
        #for i in range(self.refiner_depth):
        #    refine_score.append(F.softmax(self.refine_score[i](x), dim=1))
        return refine_score
    
if __name__ == "__main__":
    boxes = torch.sort(torch.rand((2,10,4))).values
    cls_prob = torch.rand((2,10,6)) #5 cls, 1 bg
    im_labels = torch.tensor([1,3])
    res = select_proposals(boxes,cls_prob,im_labels)
    print(res)


