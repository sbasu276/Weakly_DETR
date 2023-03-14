# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .refiner import RefinementAgents, select_proposals


class DETRMiL(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, device, lambda_gt,
                 refiner_depth=3, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        #self.class_embed = nn.Linear(hidden_dim, num_classes) 
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.refiner = Refiner(hidden_dim, num_classes, refiner_depth, device, lambda_gt)

    def forward(self, samples: NestedTensor, labels):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        
        hs = hs.squeeze(0)
        #outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        #outputs_props = self.prop_embed(hs)
        refiner_result = self.refiner(hs, outputs_coord, labels)

        out = { 'pred_logits': refiner_result['final_score'], \
                'pred_boxes': outputs_coord,\
                'cls_logits': refiner_result['cls_logits'],\
                'refiner_out': refiner_result }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class Refiner(nn.Module):
    def __init__(self, hidden_dim, num_classes, refiner_depth, device, lambda_gt):
        super().__init__()
        self.refiner_depth = refiner_depth
        #self.box_features = RoiPoolLayer(self.backbone.hidden_dim, self.backbone.spatial_scale)
        self.mil = MIL(hidden_dim, num_classes)
        self.mil_loss = nn.CrossEntropyLoss()
        self.refinement_agents = RefinementAgents(hidden_dim, num_classes+1, refiner_depth)
        self.refine_losses = [RefineLoss() for i in range(refiner_depth)]
        self.device = device
        self.lambda_gt = lambda_gt
        
    def forward(self, x, boxes, labels):
        with torch.set_grad_enabled(self.training):
            #box_feat = self.box_features(x, rois)
            mil_score = self.mil(x)
            refine_score = self.refinement_agents(x)
            #refine_score = [F.softmax(output, dim=1) for output in outputs]
            im_cls_score = mil_score.sum(dim=1)#, keepdim=True)

            return_dict = {}
            
            if self.training:

                return_dict['losses'] = {}
                # image classification loss
                loss_mil = self.mil_loss(im_cls_score, labels-1)
                return_dict['losses']['loss_mil'] = loss_mil
                loss = 0#loss_mil
                # refinement loss
                return_dict['delta'] = self.lambda_gt

                for i, prop_score in enumerate(refine_score):
                    if i == 0:
                        refinement_output = select_proposals(boxes, mil_score, 
                                                             labels, self.device, self.lambda_gt, bg_sep=False)
                    else:
                        refinement_output = select_proposals(boxes, refine_score[i-1], 
                                                             labels, self.device, self.lambda_gt)
                    refine_loss = self.refine_losses[i](prop_score,
                                                        refinement_output['labels'],
                                                        refinement_output['cls_weights'])
                    
                    return_dict['losses']['refine_loss%d' % i] = refine_loss.clone()
                    loss = loss + refine_loss.sum()
                # pytorch0.4 bug on gathering scalar(0-dim) tensors
                #for k, v in return_dict['losses'].items():
                #    return_dict['losses'][k] = v.unsqueeze(0)
                return_dict['loss'] = loss_mil + 0.3*loss
            
            final_scores = refine_score[0]
            for i in range(1, self.refiner_depth):
                final_scores += refine_score[i]
            final_scores /= self.refiner_depth
            return_dict['final_score'] = final_scores[:, :, 1:]
            return_dict['cls_logits'] = im_cls_score

            return return_dict


class MIL(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.prop = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, 0)
        nn.init.normal_(self.prop.weight, std=0.01)
        nn.init.constant_(self.prop.bias, 0)

    def forward(self, x):
        """ x: tensor (BxNxD) 
        B: batch size, N: # proposals/ queries, D: dimension
        mil_score: BxNxC (C: # classes) 
        """
        C = self.cls(x)
        D = self.prop(x)
        mil_score = F.softmax(C, dim=2) * F.softmax(D, dim=1)
        return mil_score
    

class RefineLoss(nn.Module):
    def forward(self, prop_score, label_score, cls_weights):
        """
        prop_score: BxNx(C+1)
        label_score: BxNx(C+1)
        cls_weights: BxN
        """
        eps = 1e-6
        prop_score = F.log_softmax(prop_score, dim=1)#.log()  # avoid nan   
        num_props = prop_score.size(1)
        #print(cls_weights.size())
        cls_weights = cls_weights.unsqueeze(2).repeat((1, 1, prop_score.size(2))) # broadcast
        #print(cls_weights.size(), label_score.size(), prop_score.size())
        
        loss = cls_weights * prop_score * label_score
        loss = -loss.view(loss.size(0),-1).sum(dim=1)/num_props
        return loss

def nms(boxes, scores, iou_thresh):
    
    batch_size = boxes.shape[0]
    num_boxes = boxes.shape[1]
    # Flatten the boxes and scores tensors    
    boxes = boxes.view(-1, 4)
    scores = scores.view(-1)
    # Sort the scores in descending order    
    sorted_idx = torch.argsort(scores, descending=True)
    keep = [] # List to hold the indices of the boxes that survived NMS    
    while len(sorted_idx) > 0:
        # Select the box with highest score and add it to the keep list        
        idx = sorted_idx[0]
        keep.append(idx)
        # Compute IoU of the selected box with the remaining boxes        
        iou, _ = box_ops.box_iou(boxes[idx].unsqueeze(0), boxes[sorted_idx[1:]])
        # Remove the boxes with high IoU from the sorted_idx list        
        idx_to_remove = (iou > iou_thresh).squeeze()
        sorted_idx = sorted_idx[1:][~idx_to_remove]
    keep = torch.tensor(keep)
    ret_boxes = boxes[keep]
    return keep, ret_boxes


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, device, nms=False):
        super().__init__()
        self.device = device
        self.nms = nms

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox =  outputs['pred_logits'], outputs['pred_boxes']
        cls_preds = torch.argmax(outputs['cls_logits'], dim=1)

        assert len(out_logits) == len(target_sizes)
        #assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.clamp(boxes, min=0, max=1)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(self.device)
        boxes = boxes * scale_fct[:, None, :]
        
        if self.nms:
            keep, _ = nms(boxes[0], scores, 0.5)
            boxes = boxes[0][keep]
            scores = scores[keep]
            labels = labels[keep]

        results = [{'scores': s, 'labels': l, 'boxes': b, 'cls': c} for s, l, b, c in zip(scores, labels, boxes, cls_preds)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.dataset_file == "gbc":
        num_classes = 2
    elif args.dataset_file == "polyp":
        num_classes = 2
    elif args.dataset_file == "gist":
        num_classes = 2
    else:
        num_classes = 20
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    device = torch.device(args.device) 

    model = DETRMiL(
        backbone,
        transformer,
        num_classes=num_classes,
        device=device,
        lambda_gt=args.lambda_gt,
        num_queries=args.num_queries,
        refiner_depth=args.refiner_depth,
        aux_loss=False#args.aux_loss,
    )
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    postprocessors = {'bbox': PostProcess(device)}
    
    return model, criterion, postprocessors
