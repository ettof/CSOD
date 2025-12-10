CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch  --nproc_per_node=4 --master-port=7739 train.py \
    --batch_size 4 \
    --num_workers 4 \
    --lr_rate 0.001 \
    --seed 42 \
    --model_size tiny \
    --data_path data/CSOD10K \
    --sam_ckpt checkpoints/sam2.1_hiera_tiny.pt \
    --save_dir output/Tiny\
    --epochs 300 \
    --img_size 512

