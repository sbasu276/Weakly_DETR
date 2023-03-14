#/!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train_mil.py \
                --optim="sgd" \
                --lr=1e-3 \
                --weight_decay=1e-6 \
                --lr_backbone=5e-4 \
                --alpha=$3 \
                --gbc_transform \
                --dilation \
                --coco_path="gbc_coco" \
                --val_path="data/malg_gb_val.json" \
                --train_path="data/malg_gb_train.json" \
                --img_train_path="train" \
                --output_dir="outs/$2" \
                --epochs=100 \
                --hidden_dim=256 \
                --seed=42 \
                --num_workers=0 \
                --batch_size=16 \
                --num_queries=100  \
                --refiner_depth=$4 \
                --cos_lr \
                --lambda_gt=$5 \
                --resume="weights/detr.pth" \
                --ckpt

