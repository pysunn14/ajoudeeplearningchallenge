import torch
from tqdm import tqdm
import base64
import transformers
import pandas as pd
from datasets import load_dataset
import io
from PIL import Image
import requests
from peft import PeftModel
import os
import hashlib
import argparse
import warnings

# Qwen
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# PIL 이미지 크기 제한 
PIL_MAX_PIXELS = 89478485  
Image.MAX_IMAGE_PIXELS = PIL_MAX_PIXELS

FILE_PATH = '/hdd1/minseok/dev/contest/multimodal/'
IMAGE_CACHE_DIR = os.path.join(FILE_PATH, "image_cache")
DOWNLOAD_REPORT_PATH = os.path.join(FILE_PATH, "download_report")
MAX_NEW_TOKENS = 512

def _url_to_cache_path(url: str) -> str:
    parsed = url.split("?")[0]
    ext = os.path.splitext(parsed)[1] if "." in parsed else ".jpg"
    name = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(IMAGE_CACHE_DIR, name + ext) 

def load_successful_urls():
    successful_urls_path = os.path.join(DOWNLOAD_REPORT_PATH, "successful_urls.txt")
    
    if not os.path.exists(successful_urls_path):
        print("다운로드 성공 URL 목록이 없음")
        return set()
    
    successful_urls = set()
    with open(successful_urls_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url:
                successful_urls.add(url)
    
    print(f"다운로드 성공 URL: {len(successful_urls)}개")
    return successful_urls

def is_base64_image(image_data):
    if not isinstance(image_data, str):
        return False
    
    # data:image/... 형태
    if image_data.startswith('data:image'):
        return True
    
    # Base64 Check 
    if len(image_data) > 100:  # 최소 길이 체크
        try:
            # Base64 디코딩 시도
            decoded = base64.b64decode(image_data, validate=True)
            # 이미지 헤더 확인 (JPEG, PNG, GIF 등)
            if decoded.startswith(b'\xff\xd8\xff') or \
               decoded.startswith(b'\x89PNG') or \
               decoded.startswith(b'GIF8'):
                return True
        except Exception:
            pass
    
    return False

def load_image_from_base64(base64_str):
    try:
        # Base64 디코딩
        if base64_str.startswith('data:image'):
            # data:image/jpeg;base64, 형태인 경우 헤더 제거
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image

    except Exception as e:
        print(f" Base64 이미지 로드 실패")
        return None

def has_images_in_message(messages):
    for msg in messages:
        for content in msg.get('content', []):
            if content.get('type') == 'image' and content.get('image') is not None:
                return True
    return False

#  메시지에서 None 값인 image/text 속성을 제거하고 Base64 이미지를 PIL Image로 변환
def clean_message_content(messages):
    cleaned_messages = []

    for msg in messages:
        cleaned_content = []
        for content_item in msg['content']:
            cleaned_item = {'type': content_item['type']}
            
            if content_item['type'] == 'text' and content_item.get('text') is not None:
                cleaned_item['text'] = content_item['text']
            elif content_item['type'] == 'image' and content_item.get('image') is not None:
                image_data = content_item['image']
                
                # 문자열인 경우 Base64인지 파일 경로인지 확인
                if isinstance(image_data, str):
                    if is_base64_image(image_data):
                        # Base64 이미지인 경우 PIL Image로 변환
                        pil_image = load_image_from_base64(image_data)
                        if pil_image is not None:
                            cleaned_item['image'] = pil_image
                        else:
                            print(f"Base64 이미지 변환 실패")
                            # Base64 변환 실패 시 원본 문자열 유지 
                            cleaned_item['image'] = image_data
                    else:
                        # 파일 경로인 경우 그대로 
                        cleaned_item['image'] = image_data
                else:
                    # 이미 PIL Image인 경우 그대로 사용
                    cleaned_item['image'] = image_data
            
            if ('text' in cleaned_item and cleaned_item['text'] is not None) or \
               ('image' in cleaned_item and cleaned_item['image'] is not None):
                cleaned_content.append(cleaned_item)
        
        if cleaned_content:
            cleaned_messages.append({
                'role': msg['role'],
                'content': cleaned_content
            })    
    return cleaned_messages
                
def inference_process(dataset, processor, model, successful_urls,args):

    results = []
    id = 0
    
    # 통계 
    stats = {
        'cache_hit': 0,
        'http_download': 0,
        'cache_miss': 0,
        'base64_string': 0,
        'base64_bytes': 0,
        'base64_error': 0,
        'unknown_format': 0,
        'image_load_errors': 0,
        'processed_samples': 0
    }

    #  모든 메시지에 대해서 Clean화 적용 
    cleaned_dataset = []
    for sample in tqdm(dataset, desc="메시지 클린화"):
        cleaned_sample = sample.copy()    
        cleaned_sample['messages'] = clean_message_content(sample['messages'])
        cleaned_dataset.append(cleaned_sample)
    
    for i, sample in enumerate(tqdm(cleaned_dataset, desc="Inference")):
        try:
            cleaned_messages = sample['messages']

            text = processor.apply_chat_template(
                cleaned_messages, tokenize=False, add_generation_prompt=True 
            )

            # Image : 비전 정보 추출 
            has_images = has_images_in_message(cleaned_messages)
            
            if has_images:
                image_inputs, _ = process_vision_info(cleaned_messages) 
                
                # 오류 체크 
                if image_inputs is None:
                    stats['image_load_errors'] += 1
                    print(f" Sample {id}: process_vision_info returned None")
                    image_inputs = []
            else:
                image_inputs = []

            # Image            
            if len(image_inputs) > 0:
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

            # Text-Only
            else:
                inputs = processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
            
            # Inference Generation
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Output Decoding
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_texts = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            final_output = output_texts[0] if output_texts and output_texts[0] is not None else "default"
            results.append({"id": id, "output": final_output})
            stats['processed_samples'] += 1
            
            print(f'\nSample {id} - Output: {output_texts[0][:100]}...\n')
            id += 1
            
        except Exception as e:
            print(f"Sample {id} 처리 오류 - {e}")
            results.append({"id": id, "output": "UNIFIED_INFERENCE_ERROR"})
            stats['image_load_errors'] += 1
            id += 1
            continue
        
    # 통계 출력
    print(f"[ 추론 통계 ]")
    print(f"성공 : {stats['processed_samples']}개")
    print(f"오류 : {stats['image_load_errors']}개")
    print(f"결과: {len(results)}개")
    
    return results

def main():

    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--adapter', type=str, default="finetuned/checkpoint-694")
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--preprocessed-data', type=str, default="preprocessed_inference/")
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    
    # Model Type 
    is_base_model = args.base

    if is_base_model:
        print("Base Model : 파인튜닝 없이 원본 Qwen2.5-VL 사용")
        model_type = "base"
    else:
        print("Finetuned Model: 어댑터 가중치 적용")
        model_type = "finetuned"

    # data path        
    if is_base_model:
        if args.sample:
            base_name = "base_sample_submission"
        else:
            base_name = "base_submission"
    else:
        if args.sample:
            base_name = "finetuned_sample_submission"
        else:
            base_name = "finetuned_submission"
        
    output_filename = f"{base_name}.csv"

    if args.sample:
        print("Sample Mode : 샘플 테스트 데이터로 추론 (빠른 테스트)")
    else:
        print("Default Mode : 모든 테스트 샘플로 추론")
    use_gt_csv = False
    
    if not is_base_model:
        print(f"어댑터 경로: {args.adapter}")
    
    # Load the model and processor
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="../model/qwen/Qwen2.5-VL-7B-Instruct"
    )       

    # 모델 선택 
    if is_base_model:
        print(" 베이스 모델 사용 (파인튜닝 적용 안함)")
        model = base_model
    else:
        # Load the Adapter
        print(f"Finetuned Adapter: {args.adapter}")
        model = PeftModel.from_pretrained(
            base_model, 
            args.adapter,
            torch_dtype=torch.bfloat16
        )

        print("Merging adapter weights...")
        model = model.merge_and_unload()  # 성능 최적화

    processor = AutoProcessor.from_pretrained(model_name, padding_side='left')
    transformers.logging.set_verbosity_error()

    print("이미지 캐시 초기화 중...")
    successful_urls = load_successful_urls()

    results = []
    id = 0
    model.eval()

    # Dataset 로드
    if model is not None:
        # 전처리된 추론 데이터 사용 (기본 모드)
        print("🔄 전처리된 추론 데이터 로드 중...")
        
        # 샘플 모드에 따른 데이터셋 선택
        if args.sample:
            dataset_name = "sample_dataset"
        else:
            dataset_name = "inference_dataset"
        
        preprocessed_path = FILE_PATH + args.preprocessed_data + dataset_name
        
        if not os.path.exists(preprocessed_path):
            print(f"전처리된 데이터가 없음: {preprocessed_path}")
            return
        
        from datasets import load_from_disk
        sample_dataset = load_from_disk(preprocessed_path)
        print(f" 데이터 로드 완료: {len(sample_dataset)}개 sample ")
        
        if args.sample:
            metadata_filename = "sample_metadata.json"
        else:
            metadata_filename = "inference_metadata.json"
            
        metadata_path = FILE_PATH + args.preprocessed_data + metadata_filename

        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            print(f"[ 전처리 통계 ]")
            print(f"총 샘플: {metadata['total_samples']}")
            print(f"이미지 성공: {metadata['image_success']}")
            print(f"이미지 실패: {metadata['image_failed']}")
            
        use_unified_process = True  

    results = inference_process(
        sample_dataset, 
        processor, 
        model, 
        successful_urls, 
        args
    )

    result_df = pd.DataFrame(results)

    # submissions  
    submissions_dir = os.path.join(FILE_PATH, "src/submissions") 
    os.makedirs(submissions_dir, exist_ok=True)
    output_path = os.path.join(submissions_dir, output_filename)
    result_df.to_csv(output_path, index=False)

    print(result_df.head())

    # 최종 결과 요약
    print(f" [ 최종 결과 ] ") 
    print(f" Model Type : {'base model' if is_base_model else 'finetuned model'}")
    print(f"예측 결과 : {len(result_df)}개")
    
if __name__ == "__main__":
    main()

