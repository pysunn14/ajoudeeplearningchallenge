#!/usr/bin/env python3

"""
evaluation.py
종합 멀티모달 모델 평가 

- Public Score: 전체 BLEU 평균
- Private Score: 태스크별 메트릭 평균 

  * Math Reasoning: Exact Match
  * Captioning: CLIP Score  
  * VQA: Accuracy
  * Text QA: Accuracy
  * Summarization: BERT Score
  
python evaluation.py --submission submission.csv --gt GT.csv

"""
import os
import sys

os.environ['HOME'] = '/hdd1/minseok'
os.environ['HF_HOME'] = '/hdd1/minseok/dev/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/hdd1/minseok/dev/hf_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/hdd1/minseok/dev/hf_cache/datasets'
os.environ['TORCH_HOME'] = '/hdd1/minseok/dev/hf_cache/torch'

os.environ['CLIP_CACHE_DIR'] = '/hdd1/minseok/dev/hf_cache/models--openai--clip-vit-base-patch32'
os.environ['XDG_CACHE_HOME'] = '/hdd1/minseok/dev/hf_cache'

os.makedirs('/hdd1/minseok/dev/hf_cache/transformers', exist_ok=True)
os.makedirs('/hdd1/minseok/dev/hf_cache/datasets', exist_ok=True)
os.makedirs('/hdd1/minseok/dev/hf_cache/torch', exist_ok=True)
os.makedirs('/hdd1/minseok/dev/hf_cache/clip', exist_ok=True)

import pandas as pd
import argparse
from collections import Counter
import math
import re
import string
from typing import List, Tuple, Dict, Optional
import json
import warnings
import ast
from PIL import Image
import requests
from io import BytesIO
import base64
import numpy as np
import hashlib
import traceback

# === 이미지 캐시 관련 설정 (inference.py에서 가져옴) ===
FILE_PATH = '/hdd1/minseok/dev/contest/multimodal/'
IMAGE_CACHE_DIR = os.path.join(FILE_PATH, "image_cache")
DOWNLOAD_REPORT_PATH = os.path.join(FILE_PATH, "download_report")

def _url_to_cache_path(url: str) -> str:
    parsed = url.split("?")[0]
    ext = os.path.splitext(parsed)[1] if "." in parsed else ".jpg"
    name = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(IMAGE_CACHE_DIR, name + ext)

def load_successful_urls():
    successful_urls_path = os.path.join(DOWNLOAD_REPORT_PATH, "successful_urls.txt")
    
    if not os.path.exists(successful_urls_path):
        return set()
    
    successful_urls = set()
    with open(successful_urls_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url:
                successful_urls.add(url)
    
    print(f" 다운로드 성공한 URL: {len(successful_urls)}개") 
    return successful_urls

def download_image_from_url(url: str, timeout: int = 10):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        
        # 이미지 데이터를 메모리에서 PIL로 열기
        image_data = response.content
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        return image
    except Exception as e:
        print(f"HTTP 다운로드 실패 {url[:50]}...: {e}")
        return None

def load_image_from_cache_or_download(input_data, successful_urls=None, enable_download=True):
    try:
        if isinstance(input_data, str) and input_data.startswith("http"):
            # 캐시 확인
            if successful_urls and input_data in successful_urls:
                cache_path = _url_to_cache_path(input_data)
                if os.path.exists(cache_path):
                    return Image.open(cache_path).convert("RGB"), "cache_hit"

            # 캐시에 없으면 HTTP 다운로드 시도
            if enable_download:
                image = download_image_from_url(input_data)
                if image is not None:
                    return image, "http_download"
            
            # 모든 방법 실패
            return None, "cache_miss"
                
        elif isinstance(input_data, bytes):
            return Image.open(BytesIO(input_data)).convert("RGB"), "base64_bytes"
            
        elif isinstance(input_data, str):
            try:
                image_bytes = base64.b64decode(input_data)
                return Image.open(BytesIO(image_bytes)).convert("RGB"), "base64_string"
            except Exception:
                return None, "base64_error"
        else:
            return None, "unknown_format"
            
    except Exception as e:
        return None, f"error_{str(e)[:20]}"

os.environ['HF_HOME'] = '/hdd1/minseok/dev/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/hdd1/minseok/dev/hf_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/hdd1/minseok/dev/hf_cache/datasets'
os.makedirs('/hdd1/minseok/dev/hf_cache/transformers', exist_ok=True)
os.makedirs('/hdd1/minseok/dev/hf_cache/datasets', exist_ok=True)

# 평가 메트릭을 위한 imports
try:
    import open_clip
    import torch
    import torch.nn.functional as F
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    warnings.warn("open_clip 라이브러리가 없어 CLIP Score를 계산할 수 없습니다.")

# BERT Score 라이브러리 시도 
try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    warnings.warn("bert-score 라이브러리가 없어 요약문 평가가 제한됩니다.")

# Transformers 라이브러리 시도
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers 라이브러리가 없습니다.")

def calculate_bert_score(candidate: str, reference: str) -> Dict[str, float]:
    
    if not candidate or not reference:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if BERT_SCORE_AVAILABLE:
        try:
            # 로컬 BERT 모델 디렉터리 설정
            local_bert_dir = "/hdd1/minseok/dev/contest/multimodal/model/bert"
            bert_model_path = os.path.join(local_bert_dir, "models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
            
            print(f"🔍 BERT 모델 디렉터리 확인: {bert_model_path}")
            print(f"🔍 경로 존재 여부: {os.path.exists(bert_model_path)}")
            
            # 환경 변수 백업
            original_transformers_cache = os.environ.get('TRANSFORMERS_CACHE', None)
            original_hf_home = os.environ.get('HF_HOME', None)
            
            # local_cache 
            os.environ['TRANSFORMERS_CACHE'] = local_bert_dir
            os.environ['HF_HOME'] = local_bert_dir
            
            try:
                if os.path.exists(bert_model_path):
                    print(f"✅ 로컬 BERT 모델 발견")
                    
                    P, R, F1 = bert_score_func(
                        [candidate], [reference], 
                        lang="en", 
                        model_type="bert-base-uncased",  # 표준 모델명 사용
                        verbose=False,
                        device="cpu",
                        idf=False,  # IDF 가중치 비활성화로 속도 향상
                        batch_size=1
                    )

                else:
                    print(f" 로컬 BERT 모델 없음, 다운로드") 
                    P, R, F1 = bert_score_func(
                        [candidate], [reference], 
                        lang="en", 
                        model_type="bert-base-uncased",
                        verbose=False,
                        device="cpu",
                        idf=False,
                        batch_size=1
                    )
                    print(f" BERT 모델 다운로드 ") 
                
                print(f" BERT Score 계산 완료: P={float(P[0]):.4f}, R={float(R[0]):.4f}, F1={float(F1[0]):.4f}")
                
                return {
                    'precision': float(P[0]),
                    'recall': float(R[0]),
                    'f1': float(F1[0])
                }
                
            finally:
                # 환경 변수 복원
                if original_transformers_cache is not None:
                    os.environ['TRANSFORMERS_CACHE'] = original_transformers_cache
                else:
                    os.environ.pop('TRANSFORMERS_CACHE', None)
                    
                if original_hf_home is not None:
                    os.environ['HF_HOME'] = original_hf_home
                else:
                    os.environ.pop('HF_HOME', None)
                    
        except Exception as e:
            warnings.warn(f"BERT Score 계산 실패했습니다: {e}")
            print(f" 상세 에러: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    else:
        print('BERT-score 사용이 불가능합니다.')
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

def calculate_clip_score(image_path_or_url: str, text: str) -> float:
    if not text or not image_path_or_url:
        return 0.0
    
    try:
        import open_clip
        import torch
        from PIL import Image
        import requests
        from io import BytesIO
        import base64
        
        class SimpleOpenCLIPScore:
            def __init__(self):
                # 로컬 모델 경로
                local_model_path = "/hdd1/minseok/dev/contest/multimodal/model/clip/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/"
                
                # 🔍 디버깅: 경로 존재 여부 확인
                print(f"🔍 디버깅 - 모델 경로 확인: {local_model_path}")
                print(f"🔍 디렉터리 존재 여부: {os.path.exists(local_model_path)}")
                print(f"🔍 디렉터리인지 확인: {os.path.isdir(local_model_path)}")
                
                # 상위 디렉터리도 확인
                parent_dir = os.path.dirname(local_model_path)
                print(f"🔍 상위 디렉터리: {parent_dir}")
                print(f"🔍 상위 디렉터리 존재: {os.path.exists(parent_dir)}")
                
                # 디렉터리 내 파일 목록 확인
                if os.path.exists(local_model_path): 

                    try:
                        files = os.listdir(local_model_path)
                        print(f"🔍 디렉터리 내 파일들: {files}")
                        # .bin 파일 찾기
                        bin_files = [f for f in files if f.endswith('.bin')]
                        safetensors_files = [f for f in files if f.endswith('.safetensors')]
                        print(f" .bin 파일들: {bin_files}")
                        print(f" .safetensors 파일들: {safetensors_files}")

                    except Exception as e:
                        print(f" 디렉터리 읽기 실패: {e}")
                        
                # 실제 모델 파일 경로 찾기
                possible_files = [
                    os.path.join(local_model_path, "open_clip_pytorch_model.bin"),
                    os.path.join(local_model_path, "pytorch_model.bin"),
                    os.path.join(local_model_path, "model.safetensors"),
                    os.path.join(local_model_path, "open_clip_pytorch_model.safetensors")
                ]
                
                actual_model_file = None
                for file_path in possible_files:
                    print(f" 파일 확인: {file_path} -> 존재: {os.path.exists(file_path)}")
                    if os.path.exists(file_path):
                        actual_model_file = file_path
                        break
                
                if not actual_model_file:
                    raise FileNotFoundError(f" 모델 파일을 찾을 수 없습니다. 확인된 경로: {local_model_path}")
                
                print(f"✅ 실제 사용할 모델 파일: {actual_model_file}")

                # 간단하게 torch.load로 직접 로드
                self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32')
                
                # 로컬 가중치 직접 로드
                state_dict = torch.load(actual_model_file, map_location='cpu')
                self.model.load_state_dict(state_dict)
                self.model.eval()
                
                print(f" OpenCLIP 로컬 모델 로딩 완료: {actual_model_file}") 
                
            def score(self, image_path_or_url, text):
                try:
                    # 🔍 이미지 입력 디버깅
                    print(f" 이미지 입력 타입: {type(image_path_or_url)}")
                    print(f" 이미지 입력 길이: {len(str(image_path_or_url))}")
                    print(f" 이미지 입력 시작 부분: {str(image_path_or_url)[:100]}...")
                    
                    if not hasattr(self, 'successful_urls'):
                        self.successful_urls = load_successful_urls()
                    
                    # 캐시 우선 로딩 시도
                    image, load_status = load_image_from_cache_or_download(
                        image_path_or_url, self.successful_urls, enable_download=True
                    )
                    
                    print(f" 이미지 로딩 상태: {load_status}")
                    
                    if image is None:
                        print(f" 이미지 로딩 실패: {load_status}")
                        return 0.0
                    
                    print(f" 이미지 로딩 성공: {image.size}, 방법: {load_status}")
                    
                    # 전처리
                    image_input = self.preprocess(image).unsqueeze(0)
                    text_input = open_clip.tokenize([text])
                    
                    with torch.no_grad():
                        image_features = self.model.encode_image(image_input)
                        text_features = self.model.encode_text(text_input)
                        
                        # 정규화
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        
                        # 유사도 계산
                        similarity = (image_features @ text_features.T).item()
                    
                    # 0~1 범위로 정규화
                    score = max(0.0, min(1.0, (similarity + 1) / 2))
                    print(f" CLIP Score 계산 완료 : {score}")
                    return score
                    
                except Exception as e:
                    print(f" 에러 : {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    warnings.warn(f"이미지 처리 실패: {e}")
                    return 0.0
                    
        # 모델 객체 생성 (캐싱)
        if not hasattr(calculate_clip_score, "openclip_model"):
            calculate_clip_score.openclip_model = SimpleOpenCLIPScore()
        
        return calculate_clip_score.openclip_model.score(image_path_or_url, text)
        
    except Exception as e:
        warnings.warn(f"OpenCLIP Score 계산 실패: {e}")
        return 0.0

def calculate_text_qa_word_accuracy(predicted_list: List[str], ground_truth_list: List[str]) -> float:
    if not predicted_list or not ground_truth_list:
        return 0.0
    
    total_words = len(ground_truth_list)
    if total_words == 0:
        return 0.0
    
    correct_words = 0
    
    # 길이가 다른 경우 짧은 쪽에 맞춤
    min_length = min(len(predicted_list), len(ground_truth_list))
    
    for i in range(min_length):
        pred_word = str(predicted_list[i]).strip().lower()
        gt_word = str(ground_truth_list[i]).strip().lower()
        
        if pred_word == gt_word:
            correct_words += 1
    
    return correct_words / total_words

def calculate_summarization_bert_score(candidate: str, reference: str) -> float:

    bert_scores = calculate_bert_score(candidate, reference)
    return bert_scores['f1']

def parse_text_qa_ground_truth(ground_truth: str) -> List[str]:
    if not ground_truth:
        return []
    
    try:
        # JSON 파싱 시도
        if ground_truth.strip().startswith('{'):
            try:
                data = json.loads(ground_truth)
            except json.JSONDecodeError:
                # Python 딕셔너리 문자열 
                data = ast.literal_eval(ground_truth)
            
            if isinstance(data, dict) and 'input_text' in data:
                input_text = data['input_text']
                if isinstance(input_text, list):
                    return [str(item) for item in input_text]
                else:
                    return [str(input_text)]
        
        # 리스트 형태로 직접 파싱 시도
        if ground_truth.strip().startswith('['):
            try:
                data = json.loads(ground_truth)
                if isinstance(data, list):
                    return [str(item) for item in data]
            except json.JSONDecodeError:
                data = ast.literal_eval(ground_truth)
                if isinstance(data, list):
                    return [str(item) for item in data]
        
        # 단일 문자열인 경우
        return [str(ground_truth)]
        
    except Exception as e:
        warnings.warn(f"Ground truth 파싱 실패: {e}")
        return [str(ground_truth)]

def parse_text_qa_prediction(prediction: str) -> List[str]:
    if not prediction:
        return []
    
    # 콤마로 구분된 raw text를 파싱
    answers = [answer.strip() for answer in prediction.split(',')]
    return [answer for answer in answers if answer]  # 빈 문자열 제거

def calculate_vqa_accuracy(candidate: str, reference: str) -> float:
    cand_normalized = normalize_text(candidate)
    ref_normalized = normalize_text(reference)
    
    return 1.0 if cand_normalized == ref_normalized else 0.0

def normalize_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    # 구두점 제거
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 여러 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []

def extract_math_answer(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    # #### 패턴 찾기 (해시 4개 뒤의 숫자/문자)
    pattern = r'####\s*([^\s\n]+)'
    matches = re.findall(pattern, text)
    
    if matches:
        # 마지막 #### 뒤의 답안 사용
        answer = matches[-1].strip()
        # 숫자만 추출 (소수점, 음수 포함)
        number_pattern = r'^-?\d+\.?\d*$'
        if re.match(number_pattern, answer):
            return answer
        else:
            # 숫자가 아닌 경우 원본 반환
            return answer
    
    return ""

def calculate_exact_match(prediction: str, reference: str) -> bool:
    pred_answer = extract_math_answer(prediction)
    ref_answer = extract_math_answer(reference)
    
    if not pred_answer or not ref_answer:
        return False
    
    # 숫자인 경우 수치 비교
    try:
        pred_num = float(pred_answer)
        ref_num = float(ref_answer)
        # 부동소수점 비교 (작은 오차 허용)
        return abs(pred_num - ref_num) < 1e-6
    except ValueError:
        # 숫자가 아닌 경우 문자열 비교
        return pred_answer.lower() == ref_answer.lower()

def calculate_ngram_precision(candidate_tokens: List[str], reference_tokens: List[str], n: int) -> float:

    if len(candidate_tokens) < n:
        return 0.0
    
    # candidate에서 n-gram 생성
    candidate_ngrams = []
    for i in range(len(candidate_tokens) - n + 1):
        ngram = tuple(candidate_tokens[i:i+n])
        candidate_ngrams.append(ngram)
    
    # reference에서 n-gram 생성
    reference_ngrams = []
    for i in range(len(reference_tokens) - n + 1):
        ngram = tuple(reference_tokens[i:i+n])
        reference_ngrams.append(ngram)
    
    if not candidate_ngrams:
        return 0.0
    
    # n-gram 카운트
    candidate_counts = Counter(candidate_ngrams)
    reference_counts = Counter(reference_ngrams)
    
    # 클리핑된 카운트 (reference에서 나타나는 최대 횟수로 제한)
    clipped_counts = 0
    total_counts = len(candidate_ngrams)
    
    for ngram, count in candidate_counts.items():
        clipped_counts += min(count, reference_counts.get(ngram, 0))
    
    return clipped_counts / total_counts if total_counts > 0 else 0.0

def calculate_brevity_penalty(candidate_length: int, reference_length: int) -> float:
    
    if candidate_length > reference_length:
        return 1.0
    elif candidate_length == 0:
        return 0.0
    else:
        return math.exp(1 - reference_length / candidate_length)

def calculate_bleu_score(candidate: str, reference: str, max_n: int = 4) -> Tuple[float, dict]:

    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    
    if not candidate_tokens or not reference_tokens:
        return 0.0, {"1-gram": 0.0, "2-gram": 0.0, "3-gram": 0.0, "4-gram": 0.0, "BP": 0.0}
    
    # n-gram precision 계산
    precisions = []
    precision_details = {}
    
    for n in range(1, max_n + 1):
        precision = calculate_ngram_precision(candidate_tokens, reference_tokens, n)
        precisions.append(precision)
        precision_details[f"{n}-gram"] = precision
    
    # Brevity Penalty 계산
    bp = calculate_brevity_penalty(len(candidate_tokens), len(reference_tokens))
    precision_details["BP"] = bp
    
    # BLEU 스코어 계산 (기하평균)
    if any(p == 0 for p in precisions):
        bleu_score = 0.0
    else:
        log_precisions = [math.log(p) for p in precisions]
        bleu_score = bp * math.exp(sum(log_precisions) / len(log_precisions))
    
    return bleu_score, precision_details

def load_and_validate_data(submission_path: str, gt_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    print(f" Submission: {submission_path}")
    print(f" Ground Truth: {gt_path}")
    
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission 파일을 찾을 수 없습니다: {submission_path}")
    
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground Truth 파일을 찾을 수 없습니다: {gt_path}")
    
    submission_df = pd.read_csv(submission_path)
    gt_df = pd.read_csv(gt_path)
    
    # GT 데이터에 id 컬럼 추가 (0부터 시작)
    gt_df = gt_df.reset_index(drop=True)
    gt_df['id'] = gt_df.index
    
    print(f" [ 데이터 로딩 완료 ] ")
    
    return submission_df, gt_df

def calculate_metrics(submission_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    
    merged_df = pd.merge(submission_df, gt_df, on='id', suffixes=('_pred', '_gt'))
    
    if len(merged_df) == 0:
        raise ValueError("매칭되는 샘플 없음. ")
        
    print(f"매칭 샘플: {len(merged_df)}개")
    
    individual_results = []
    task_scores = {}
    all_bleu_scores = [] 
    
    evaluation_summary = {
        'captioning': 'CLIP Score ',
        'vqa': 'Accuracy ',
        'text_qa': 'Word-level Accuracy ',
        'math_reasoning': 'Exact Match ',
        'summarization': 'BERT Score F1 '
    }
    
    for idx, row in merged_df.iterrows():
        candidate = row['output_pred']
        reference = row['output_gt']
        task = row['task']
        input_data = row.get('input', '')
        
        # 1. Public Score용 BLEU 계산 (모든 태스크)
        bleu_score, _ = calculate_bleu_score(candidate, reference)
        all_bleu_scores.append(bleu_score)
        
        # 2. Private Score용 태스크별 전문 메트릭
        primary_score = 0.0
        metric_type = 'unknown'
        additional_info = {}
        
        if task == 'captioning':
            if input_data:  # 이미지 URL/경로가 있는 경우
                primary_score = calculate_clip_score(input_data, candidate)
                metric_type = 'clip_score'
            else:
                primary_score = bleu_score
                metric_type = 'bleu_fallback'
                
        elif task == 'vqa':
            primary_score = calculate_vqa_accuracy(candidate, reference)
            metric_type = 'vqa_accuracy'
            
        elif task == 'text_qa':
            pred_words = parse_text_qa_prediction(candidate)  # 콤마 
            gt_words = parse_text_qa_ground_truth(reference)  # JSON 딕셔너리
            primary_score = calculate_text_qa_word_accuracy(pred_words, gt_words)
            metric_type = 'word_accuracy'
            additional_info = {
                'pred_word_count': len(pred_words),
                'gt_word_count': len(gt_words)
            }
            
        elif task == 'math_reasoning':
            # Exact Match
            primary_score = calculate_exact_match(candidate, reference)
            metric_type = 'exact_match'
            additional_info = {
                'pred_answer': extract_math_answer(candidate),
                'gt_answer': extract_math_answer(reference)
            }
            
        elif task == 'summarization':
            # BERT Score - 원본 요약문 input과 모델 출력 비교
            original_input = row.get('input', '')  # 원본 요약문
            if original_input:
                primary_score = calculate_summarization_bert_score(candidate, original_input)
                comparison_target = 'original_input'
            else:
                # input이 없는 경우 기존 방식 사용 (하위 호환성)
                primary_score = calculate_summarization_bert_score(candidate, reference)
                comparison_target = 'ground_truth'
            
            metric_type = 'bert_score_f1'
            additional_info = {'comparison_target': comparison_target}
            
        else:
            # 알 수 없는 태스크는 BLEU 사용
            primary_score = bleu_score
            metric_type = 'bleu_fallback'
        
        # 개별 결과 저장
        individual_results.append({
            'id': row['id'],
            'task': task,
            'primary_score': primary_score,
            'bleu_score': bleu_score,
            'metric_type': metric_type,
            'additional_info': additional_info
        })
        
        # 태스크별 점수 누적
        if task not in task_scores:
            task_scores[task] = []
        task_scores[task].append(primary_score)
    
    # Public Score
    public_score = sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0.0
    
    # Private Score
    task_averages = {}
    task_means = []
    
    for task, scores in task_scores.items():
        if scores:
            mean_score = sum(scores) / len(scores)
            task_averages[task] = {
                'count': len(scores),
                'mean_score': mean_score,
                'min_score': min(scores),
                'max_score': max(scores),
                'metric_type': evaluation_summary.get(task, 'unknown')
            }
            task_means.append(mean_score)
    
    private_score = sum(task_means) / len(task_means) if task_means else 0.0
    
    return {
        'public_score': public_score,
        'private_score': private_score,
        'overall_bleu_reference': public_score,  # 호환성
        'task_averages': task_averages,
        'individual_results': individual_results,
        'total_samples': len(merged_df),
        'evaluation_summary': evaluation_summary,
        'task_distribution': {task: len(scores) for task, scores in task_scores.items()}
    }

def print_results(results: dict):
    print(f"\n" + "="*80)
    print(f"[ 멀티모달 모델 평가 결과 ]")
    print(f"="*80)
    
    print(f"[ 전체 성능 요약 ]") 
    print(f"  - 전체 평가 샘플: {results['total_samples']:,}개")
    print(f"  - 참고용 전체 BLEU: {results['overall_bleu_reference']:.4f}")
    
    print(f"평가 체계:") 
    
    for task, description in results['evaluation_summary'].items():
        if task in results['task_averages']:
            print(f"  - {task}: {description}")
            
    print(f" 태스크별 상세 성능: ")
    for task, stats in results['task_averages'].items():
        print(f"\n  🔹 {task}:")
        print(f"    ├ 샘플 수: {stats['count']}개")
        print(f"    ├ 평가 메트릭: {stats['metric_type']}")
        print(f"    ├ 평균 점수: {stats['mean_score']:.4f}")
        print(f"    ├ 최고 점수: {stats['max_score']:.4f}")
        print(f"    └ 최저 점수: {stats['min_score']:.4f}")
        
        mean_score = stats['mean_score']
        if mean_score >= 0.8:
            grade = "🟢"
        elif mean_score >= 0.6:
            grade = "🟡"
        elif mean_score >= 0.4:
            grade = "🟠"
        else:
            grade = "🔴"
        
        print(f" 성능 등급: {grade} ({mean_score:.3f})")
    
    # 전체 성능 종합 평가
    overall_scores = [stats['mean_score'] for stats in results['task_averages'].values()]
    if overall_scores:
        overall_mean = sum(overall_scores) / len(overall_scores)
        if overall_mean >= 0.7:
            overall_grade = "🟢 "
        elif overall_mean >= 0.5:
            overall_grade = "🟡 "
        elif overall_mean >= 0.3:
            overall_grade = "🟠 "
        else:
            overall_grade = "🔴 "
        
        print(f" [ 종합 평가 ] {overall_grade}")
        print(f" 평균 성능 점수: {overall_mean:.4f}")
        print(f" BLEU: {results['overall_bleu_reference']:.4f}")
    
    # 최종 점수 강조 출력
    print(f"\n" + "="*80)
    print(f"[ 최종 평가 점수 ]")
    print(f"="*80)
    print(f"Public Score (BLEU): {results.get('public_score', 0.0):.4f}")
    print(f"Private Score (Task Average): {results.get('private_score', 0.0):.4f}")
    print(f"="*80)

def save_detailed_results(results: dict, output_path: str = "evaluation_results.json"):
    
    # JSON 
    serializable_results = {
        'evaluation_summary': results['evaluation_summary'],
        'task_averages': results['task_averages'],
        'overall_bleu_reference': results['overall_bleu_reference'],
        'total_samples': results['total_samples'],
        'individual_results': results['individual_results'][:100],  # 처음 100개만 저장
        'metadata': {
            'evaluation_type': 'multimodal',
            'metrics_used': {
                'summarization': 'BERT Score (F1)',
                'math_reasoning': 'Exact Match',
                'captioning': 'CLIP Score',
                'vqa': 'Accuracy',
                'text_qa': 'Accuracy',
                'reference_bleu': 'BLEU (모든 태스크)'
            },
            'libraries_available': {
                'transformers': TRANSFORMERS_AVAILABLE,
                'bert_score': BERT_SCORE_AVAILABLE
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
    print(f" [ 저장 완료 ]")

def main():
    parser = argparse.ArgumentParser(description='멀티모달 모델 평가 (BERT Score, CLIP Score, Exact Match, Accuracy)')

    parser.add_argument('--submission', type=str, default='finetuned_submission_sample.csv')
    parser.add_argument('--gt', type=str, default='GT.csv')
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    
    args = parser.parse_args()
    
    try:
        # 데이터 로드 및 검증
        submission_df, gt_df = load_and_validate_data(args.submission, args.gt)
        
        # 평가 수행
        results = calculate_metrics(submission_df, gt_df)
        
        # 결과 출력
        print_results(results)
        
        # 결과 저장
        if not args.no_save:
            save_detailed_results(results, args.output)

        print(f"총 {results['total_samples']} sample, {len(results['task_averages'])} task metric eval")

    except Exception as e:
        print(f"오류 : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        print(f"오류 : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()