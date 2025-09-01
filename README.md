# 아주소중한딥러닝챌린지 

## Environment

- Python 3.11.13
- pip : requirements.txt
- conda : environment.yml
- GPU : A5000 24GB
- CUDA Version: 12.6

### 폴더 구조 

<img width="362" height="728" alt="image" src="https://github.com/user-attachments/assets/94e68673-6807-4000-9400-912ac56e5086" />
- 소스코드 대부분 절대경로로 작업하여 경로 변환 필요

model_download.py : Qwen2.5VL 다운로드
download_image.py : 이미지 캐시 저장 
preprocess_train.py : 훈련 데이터셋 전처리
preprocess_inference.py : 추론 데이터셋 전처리 

train_model.py : finetuning 모델 훈련
inference.py : 모델 추론 진행 

# 훈련 옵션 

## preprocess_train --mode [multimodal, text]

- text only 모드만 사용합니다 
- 현재 훈련 및 추론이 text-only 로 진행하였으므로 multimodal 사용 X

## train_model.py



## inference.py 

--base 
--adapter 어댑터 경로를 설정합니다. finetuned/checkpoint-338 등 
--sample 샘플 데이터셋으로 추론을 진행합니다. 

## evaluation.py

-- Ground Truth 역할을 하는 GT.csv 를 생성 후 샘플 셋에 대한 추론 결과 검증이 가능합니다. 


