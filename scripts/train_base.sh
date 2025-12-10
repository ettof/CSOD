CUDA_VISIBLE_DEVICES=2,3,4,5  python -m torch.distributed.launch  --nproc_per_node=4 --master-port=7739 train.py \
    --batch_size 4 \
    --num_workers 4 \
    --lr_rate 0.001 \
    --seed 42 \
    --model_size base \
    --data_path data/CSOD10K \
    --sam_ckpt checkpoints/sam2.1_hiera_base_plus.pt \
    --save_dir output/base\
    --epochs 200 \
    --img_size 512

