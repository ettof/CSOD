CUDA_VISIBLE_DEVICES=4,5  python -m torch.distributed.launch  --nproc_per_node=2 --master-port=9739 train.py \
    --batch_size 4 \
    --num_workers 2 \
    --lr_rate 0.001 \
    --seed 42 \
    --model_size large \
    --data_path data/CSOD10K/train \
    --sam_ckpt checkpoints/sam2.1_hiera_large.pt \
    --save_dir output/CSSAM\
    --epochs 200 \
    --img_size 512



