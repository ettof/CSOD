export CUDA_VISIBLE_DEVICES=3
python evaluation.py \
    --data_path data\
    --img_size 512 \
    --checkpoint output/CSSAM_L.pth \
    --gpu_id 0 \
    --model_size "large" \
    --result_path result/CSSAM

