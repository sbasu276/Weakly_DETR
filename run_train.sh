#/!/bin/bash

for i in {0..4}
do
	echo "SPLIT $i "
	CUDA_VISIBLE_DEVICES=$1 python train_mil.py \
		--optim="sgd" \
		--lr=1e-3 \
		--weight_decay=1e-6 \
		--lr_backbone=5e-4 \
		--alpha=1.0 \
		--gbc_transform \
		--dilation \
		--coco_path="5-fold-new/split_$i" \
		--val_path="5-fold-new/split_$i/val_coco.json" \
		--train_path="5-fold-new/split_$i/train_coco.json" \
		--img_train_path="train" \
		--output_dir="outs/$2/$i" \
		--epochs=100 \
		--hidden_dim=256 \
		--seed=42 \
		--num_workers=0 \
		--batch_size=16 \
		--num_queries=100  \
		--refiner_depth=4 \
		--cos_lr \
		--lambda_gt=0.6 \
		--resume="weights/detr.pth" \
		--ckpt
done
