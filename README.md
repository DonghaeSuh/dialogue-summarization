# 일상 대화 요약 Baseline
본 리포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '일상 대화 요약'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.  

학습, 추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.   


## 실행 방법 (How to Run)
### 학습 (Train)
```
CUDA_VISIBLE_DEVICES=1,3 python -m run.train \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --epoch 5 \
    --lr 2e-5 \
    --warmup_steps 20
```

### 추론 (Inference)
```
python -m run.test \
    --output result.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
    --adapter_checkpoint_path 'adapter 경로'
```

### Dev 추론
```
python -m run.test \
    --output dev_result.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0 \
    --adapter_checkpoint_path 'adapter 경로' \
    --is_test no
```

## Reference

huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
