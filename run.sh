CUDA_VISIBLE_DEVICES=0 python -m run.qlora_train \
    --model_id hyeogi/Yi-6b-dpo-v0.2 \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 20 \
    --lr 2e-5 \
    --warmup_steps 20 \
    --save_dir ./resource/results/Yi-6b-dpo-v0.2/ \