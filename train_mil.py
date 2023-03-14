# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, evaluate_mil, train_one_epoch, train_one_mil_epoch
from models import build_mil_model, build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--cos_lr', action="store_true")

    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--lambda_gt', default=0.5, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--refiner_depth', default=3, type=int,
                        help="Number of refinement stages")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Optimizer
    parser.add_argument('--optim', default='sgd', type=str,
                        help="Optimizer type, default SGD")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    #parser.add_argument('--set_cost_class', default=1, type=float,
    #                    help="Class coefficient in the matching cost")
    #parser.add_argument('--set_cost_bbox', default=5, type=float,
    #                    help="L1 box coefficient in the matching cost")
    #parser.add_argument('--set_cost_giou', default=2, type=float,
    #                    help="giou box coefficient in the matching cost")
    # * Loss coefficients
    #parser.add_argument('--mask_loss_coef', default=1, type=float)
    #parser.add_argument('--dice_loss_coef', default=1, type=float)
    #parser.add_argument('--bbox_loss_coef', default=5, type=float)
    #parser.add_argument('--giou_loss_coef', default=2, type=float)
    #parser.add_argument('--eos_coef', default=0.1, type=float,
    #                    help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='gbc')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--val_path', type=str)
    parser.add_argument('--gbc_transform', action='store_true')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--img_train_path', type=str, default="train")
    parser.add_argument('--img_val_path', type=str, default="val")

    parser.add_argument('--thresh', default=0.2, type=int)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ckpt', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    #torch.autograd.set_detect_anomaly(True)
    #print("git:\n  {}\n".format(utils.get_sha()))
    
    #args.distributed = False

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # change to build_mil_model for gist
    model, criterion, postprocessors = build_model(args)#build_mil_model(args)
    model.to(device)
    #print("model created")
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.cos_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
    #                               collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
    #                             drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if args.ckpt:
                ckpt = {}
                for k, v in checkpoint['model'].items():
                    if 'bbox_embed' in k:
                        ckpt[k] = v
                model_without_ddp.load_state_dict(ckpt, strict=False)
            #print(checkpoint['model']['bbox_embed'].keys())
            else:
                del checkpoint['model']['class_embed.weight']
                del checkpoint['model']['class_embed.bias']
                model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                #ckpt = {}
                #for k, v in checkpoint['model'].items():
                #    if 'bbox_embed' in k:
                #        ckpt[k] = v
                #print(checkpoint['model']['bbox_embed'].keys())
                #if args.num_queries != 100:
                #    del checkpoint['model']['query_embed.weight']
        #model_without_ddp.load_state_dict(ckpt, strict=False)
        #if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #    optimizer.load_state_dict(checkpoint['optimizer'])
        #    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #    args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate_mil(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.thresh, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    #print("Start training")
    start_time = time.time()
    #print(args)
    best_miou = 0
    for epoch in range(args.epochs):#args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        #train_stats = 
        train_one_mil_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args, args.clip_max_norm)
        
        lr_scheduler.step()
        
        test_stats, coco_evaluator = evaluate_mil(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.thresh, args.output_dir
        )
        
        miou = test_stats['miou']
        res = test_stats['res']

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_paths = [output_dir / Path('epoch_%s_checkpoint.pth'%(epoch))]
            res_path = '%s/epoch_%s_res.json'%(args.output_dir,epoch)
            # extra checkpoint before LR drop and every 100 epochs
            #if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            #    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            
            with open(res_path, "w") as f:
                json.dump(res, f, indent=2)

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
