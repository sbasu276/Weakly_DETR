# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from sklearn.metrics import confusion_matrix
import numpy as np
from util import box_ops
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def train_one_mil_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args, max_norm: float = 0):
    model.train()
    #torch.autograd.set_detect_anomaly(True)
    #metric_logger = utils.MetricLogger(delimiter="  ")
    #metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    #header = 'Epoch: [{}]'.format(epoch)
    #print_freq = 10
    for samples, targets, classes in data_loader: #metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #target_classes = torch.cat([t["labels"] for t in targets])
        #print(targets, target_classes)
        classes = torch.cat(classes)
        
        target_classes = classes.to(device)
        outputs = model(samples, target_classes)
        
        loss = outputs['refiner_out']['loss']
        #for _, loss_ in outputs['refiner_out']['losses'].items():
        #    loss += loss_#.mean(dim=0, keepdim=True)
    
        optimizer.zero_grad()
        loss.backward()#retain_graph=True)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    #print("Epoch: %s, Loss: %.4f"%(epoch, loss.item()))


def get_pred_labels(pred_tensor):
    _, pred = torch.max(pred_tensor, dim=1)
    return pred.tolist()

@torch.no_grad()
def generate_scores(res, thresh):
    tgt, out = [], []
    for k, v in res.items():
        t_ = dict(boxes=v['gt'], labels=torch.tensor([1]).cuda())
        v_ = dict(boxes=v['boxes'], scores=v['scores'], labels=v['labels'])
        tgt.append(t_)
        out.append(v_)
    metric = MeanAveragePrecision(iou_thresholds=[0.1])
    metric.update(out, tgt)
    m2 = MeanAveragePrecision(iou_thresholds=[thresh])
    m2.update(out, tgt)
    m5 = MeanAveragePrecision(iou_thresholds=[0.5])
    m5.update(out, tgt)
    m25 = MeanAveragePrecision(iou_thresholds=[0.25])
    m25.update(out, tgt)
    m = MeanAveragePrecision(iou_thresholds=[0.1, 0.25, 0.5])
    m.update(out, tgt)
    return metric.compute()['map'], m2.compute()['map'].item(), \
            m5.compute()['map'].item(), m25.compute()['map'].item(), m.compute()['map'].item()
    #print("mAP: %.6f, %.6f"%(metric.compute()['map'], m2.compute()['map']))



@torch.no_grad()
def evaluate_mil(model, criterion, postprocessors, data_loader, base_ds, device, thresh, output_dir):
    model.eval()
    criterion.eval()

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    #coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    y_true, y_pred, y_pred_m = [], [], []

    miou, miou_gtk = [], 0
    frac, frac2, frac_gtk = 0, 0, 0
    thresh2 = 0.1
    res = {}
    res_dump, res_eval = {}, {}
    for samples, targets, classes in data_loader: #metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #target_classes = torch.cat([t["labels"] for t in targets])
        classes = torch.cat(classes)
        target_classes = classes.to(device)

        outputs = model(samples, target_classes)

        y_true += target_classes.tolist()
        
        #orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        orig_target_sizes = torch.stack([torch.tensor([112,112]) for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        #print("RESULT: ", results)
        #print("ORIG: ", targets)
        #res = {target['image_id'].item(): output['boxes'][torch.max(output['scores'])[-1]] \
        i = 0
        for target, output in zip(targets, results):
            ind = torch.argmax(output['scores'])
            y_pred.append(output['labels'][ind].item()) #+1
            #print(classes[i], y_pred[-1])
            if classes[i].item() != 0:
                gt = target['boxes']
                #res_dump[target['image_id'].item()] = {'gt': gt.tolist(),\
                #                                  'boxes': output['boxes'].tolist(),\
                #                                  'scores': output['scores'].tolist()}
                res_eval[target['image_id'].item()] = {'gt': gt,\
                                                  'boxes': output['boxes'][ind].unsqueeze(0),\
                                                  'scores': output['scores'][ind].unsqueeze(0),\
                                                  'labels': output['labels'][ind].unsqueeze(0)}

                pred = output['boxes'][ind]
                ious, _ = box_ops.box_iou(gt, pred.unsqueeze(0))
                miou.append(ious)
                #print(ious, gt, pred)
                if ious >= thresh:
                    frac += 1
                if ious >= thresh2:
                    frac2 += 1
                gtk_ious, _ = box_ops.box_iou(gt, output['boxes'])
                ind_gtk = torch.argmax(gtk_ious)
                miou_gtk += gtk_ious.squeeze(0)[ind_gtk].item()
                if gtk_ious.squeeze(0)[ind_gtk] >= thresh:
                    frac_gtk += 1
                if gt.size(0) != 0:
                    res[target['image_id'].item()] = {'label': y_pred[-1], \
                                                      'iou': ious.item(), \
                                                      'pred': pred.tolist(), \
                                                      'gt': gt.tolist(),\
                                                      'pred_gtk': output['boxes'][ind_gtk].tolist(),\
                                                      'score': output['scores'][ind].item(),\
                                                      'score_gtk': output['scores'][ind_gtk].item()}#, 'iou': ious[ind]}
                    r = res[target['image_id'].item()]
            i += 1
        """
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        """
    map_ = generate_scores(res_eval, thresh)
    loc_acc = frac/ len(miou)
    loc_acc2 = frac2/ len(miou)
    print("%.6f %.6f %.6f %.6f %.4f %.4f %4f"%(loc_acc2, map_[0], loc_acc, map_[1], map_[2], map_[3], map_[4]))

    #print("mIoU: %.4f, LocAcc: %.4f GTK: %.4f mIoU_GTK: %.4f"%
    #        (sum(miou)/len(miou), frac/len(miou), frac_gtk/len(miou), miou_gtk/len(miou)))
    #print([m.item() for m in miou])
    #print(r)
    #print("CFM_cls: \n", confusion_matrix(np.array(y_true), np.array(y_pred)))
    #print("CFM_mil: \n", confusion_matrix(np.array(y_true), np.array(y_pred_m)))
    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    #print("Averaged stats:", metric_logger)
    """
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = None#{k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
    """
    return {'map': map_, 'loc_acc': loc_acc, 'miou': miou, 'res': res, 'res_dump': res_dump}, coco_evaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('miou', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        #res = {target['image_id'].item(): output 
        res = {}
        mious = 0
        for target, output in zip(targets, results):
            res[target['image_id'].item()] = output
            ind = torch.argmax(output['scores'])
            gt = target['boxes']
            pred = output['boxes'][ind]
            ious, _ = box_ops.box_iou(gt, pred.unsqueeze(0))
            mious += ious
        mious = mious/len(targets)
        metric_logger.update(miou=mious)

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
