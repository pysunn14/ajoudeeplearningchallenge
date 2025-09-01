# 아주소중한딥러닝챌린지 

## Environment

- Python 3.11.13
- pip : requirements.txt
- conda : environment.yml
- GPU : A5000 24GB
- CUDA Version: 12.6

### 폴더 구조 

<img width="366" height="691" alt="image" src="https://github.com/user-attachments/assets/2377d1ee-1a5a-4fec-a7fb-1b2b6ef7a462" />


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
# Ajou Multimodal Deep Learning Challenge

> 아주소중한딥러닝챌린지 경진대회 - 멀티모달 AI 모델 개발 
> Qwen2.5-VL 7B 모델 기반 multitask 학습 시스템

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-green.svg)](https://developer.nvidia.com/cuda-downloads)

## 프로젝트 구조

```
multimodal/
├── src/                          # 소스 코드
│   ├── model_download.py         # Qwen2.5-VL 모델 다운로드
│   ├── download_images.py        # 이미지 캐시 저장 시스템
│   ├── preprocess_train.py       # 훈련 데이터 전처리
│   ├── preprocess_inference.py   # 추론 데이터 전처리  
│   ├── train_model.py           # 모델 파인튜닝 훈련
│   ├── inference.py             # 모델 추론 실행
│   ├── finetuned/              # 훈련된 모델 체크포인트 저장 
│   └── submissions/            # 제출 파일 저장소
├── dataset/deeplearningchallenge # 데이터셋 
├── image_cache/                # 이미지 캐시 
├── download_report/            # 다운로드 리포트 
├── environment.yml             # Conda 환경 설정
├── requirements.txt            # pip 패키지 목록
└── README.md                   # 프로젝트 문서
```

![프로젝트 구조](image.png)

## 🛠️ 환경 설정

### 시스템 요구사항
- **Python**: 3.11.13
- **GPU**: NVIDIA A5000 24GB (권장)
- **CUDA**: 12.6
- **RAM**: 24GB+ (권장)

### 설치 방법

- Dataset은 사전에 다운하여 저장해야합니다. 

#### Option 1: Conda 환경 (권장)
```bash
# 환경 생성 및 활성화
conda env create -f environment.yml
conda activate mlenv
```

#### Option 2: pip 설치
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 패키지 설치
pip install -r requirements.txt
```

## 사용법

### 1. 모델 다운로드
```bash
python src/model_download.py
```

### 2. 이미지 캐시 구축
```bash
python src/download_images.py
```

### 3. 데이터 전처리

#### 훈련 데이터 전처리
```bash
# 텍스트 전용 모드 (필수)
python src/preprocess_train.py --mode text

# 멀티모달 모드 (훈련 및 추론 프로세스가 text-only로 진행했으므로, 사용 X)
python src/preprocess_train.py --mode multimodal
```

#### 추론 데이터 전처리
```bash
# 3-shot 프롬프트 (기본값)
python src/preprocess_inference.py --mode three

# 1-shot 또는 2-shot 프롬프트
python src/preprocess_inference.py --mode one
python src/preprocess_inference.py --mode two

# 샘플 데이터로 테스트
python src/preprocess_inference.py --sample --mode three
```

- one shot 성능이 가장 높게 나옵니다 

### 4. 모델 훈련
```bash
# 기본 훈련 (2 에폭)
python src/train_model.py

# 사용자 정의 설정
python src/train_model.py --epochs 3 
```

#### 훈련 설정
- **에폭**: 2 (기본값)
- **배치 크기**: 4 (per device)
- **학습률**: 2e-4
- **최대 시퀀스 길이**: 1536
- **LoRA**: r=16, alpha=32
- **양자화**: 4-bit QLoRA

### 5. 모델 추론
```bash
# 기본 추론
python src/inference.py --adapter src/finetuned/checkpoint-694

# 샘플 데이터 테스트
python src/inference.py --adapter src/finetuned/checkpoint-694 --sample

# 베이스 모델 사용
python src/inference.py --base
```

### 6. 모델 샘플 평가 

- **src/submission/evaluation.py script**로 BLEU스코어가 아니라 5개의 다른 스코어 점수를 확인 가능합니다. 
- 먼저 Ground Truth 역할을 하는 GT.csv 파일을 sample dataset으로부터 수동으로 제작해야합니다.

```bash
python src/inference.py --adapter src/finetuned/checkpoint-694

pythonn src/submission/evaluation.py --submission src/submission/finetuned_sample_submission.csv --gt src/submission/GT.csv
```

## Task

1. **summarization**
2. **math_reasoning**
3. **captioning** 
4. **vqa** 
5. **text_qa**

## Architecture

- **베이스 모델**: Qwen2.5-VL-7B-Instruct
- **파인튜닝**: QLoRA (4-bit 양자화)
- **어댑터**: LoRA (Low-Rank Adaptation)
- **훈련 방식**: Supervised Fine-Tuning (SFT)

- 훈련 결과는 `training_metrics.csv`와 `training_metrics.png`로 저장됩니다.

## 주의사항

1. **현재 버전은 텍스트 학습 전용 모드**로 개발되어 있고 multimodal 모드는 사용하지 않습니다 
2. **GPU 메모리**: 24GB 이상 GPU 권장 

## License

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.



