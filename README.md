# 한국어 일상대화 요약

## 실행 방법
### 학습 (Train)
```bash
python -m run.train_qlora \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 10 \
    --lr 2e-5 \
    --warmup_steps 20 \
    --run_name preprocess \
    --save_dir ./model
```
### 추론 (Inference)
```bash
python -m run.test \
    --output result_preprocess.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --adapter_checkpoint_path /content/drive/MyDrive/국립국어원_일상대화요약/korean_dialog/korean_dialog/run/model/MLP-KTLim/llama-3-Korean-Bllossom-8B_batch_1_preprocess7_time_2024-07-21_22:21 \
    --device cuda:0
```