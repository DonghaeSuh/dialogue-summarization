CUDA_VISIBLE_DEVICES=0 python -m run.train \
    --model_id {model id} \
    --batch_size {batch size} \
    --gradient_accumulation_steps {data batch size} \
    --epoch {epoch} \
    --lr 2e-5 \
    --warmup_steps 20