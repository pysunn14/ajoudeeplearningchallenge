import os
import pandas as pd
import torch
from datasets import load_from_disk
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from torch.utils.data import DataLoader
from transformers import set_seed
import json
from PIL import Image
import base64
import io

from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import argparse

# Qwen VL utils for processing vision info
from qwen_vl_utils import process_vision_info

# TF32 활성화 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 재현성 설정
def setup_reproducibility(seed=42):

    print(f"재현성 설정: seed={seed}")
    
    # Python 
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Transformers 
    set_seed(seed)
    
    torch.backends.cudnn.deterministic = False  # 성능 우선
    torch.backends.cudnn.benchmark = True       # 성능 최적화 활성화

    print("CUDA 성능 최적화 활성화 (benchmark=True)")

# 설정 
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

FILE_PATH = '/hdd1/minseok/dev/contest/multimodal/'
PREPROCESSED_TRAINING_PATH = FILE_PATH + 'preprocessed_training/training_dataset'
TEST_SIZE = 0.1

MAX_SEQUENCE_LENGTH = 1024  

def load_preprocessed_training_data():
    
    if not os.path.exists(PREPROCESSED_TRAINING_PATH):
        print(f"전처리된 훈련 데이터가 없음: {PREPROCESSED_TRAINING_PATH}")
        return None
    
    try:
        dataset = load_from_disk(PREPROCESSED_TRAINING_PATH)
        print(f"전처리된 데이터 로드 : {len(dataset)}")
        return dataset
        
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None

def is_base64_image(image_data):
    if not isinstance(image_data, str):
        return False
    
    # data:image/... format
    if image_data.startswith('data:image'):
        return True
    
    # Base64 Check
    if len(image_data) > 100:
        try:
            decoded = base64.b64decode(image_data, validate=True)
            if decoded.startswith(b'\xff\xd8\xff') or \
               decoded.startswith(b'\x89PNG') or \
               decoded.startswith(b'GIF8'):
                return True
        except Exception:
            pass
    
    return False

# Base64 문자열을 PIL Image로 변환
def load_image_from_base64(base64_str):
    try:
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Base64 이미지 로드 실패: {e}")
        return None

def has_images_in_message(messages):
    for msg in messages:
        for content in msg.get('content', []):
            if content.get('type') == 'image' and content.get('image') is not None:
                image_data = content.get('image')
                if isinstance(image_data, str) and image_data.strip() == "":
                    continue
                return True
    return False

def clean_and_prepare_message(messages):
    cleaned_messages = []

    for msg in messages:
        cleaned_content = []
        for content_item in msg['content']:
            cleaned_item = {'type': content_item['type']}
            
            if content_item['type'] == 'text' and content_item.get('text') is not None:
                cleaned_item['text'] = content_item['text']
            elif content_item['type'] == 'image' and content_item.get('image') is not None:
                image_data = content_item['image']
                
                # 빈 문자열 Check
                if isinstance(image_data, str) and image_data.strip() == "":
                    continue  
                
                if isinstance(image_data, str):
                    if is_base64_image(image_data):
                        # Base64 이미지인 경우 PIL Image로 변환
                        pil_image = load_image_from_base64(image_data)
                        if pil_image is not None:
                            cleaned_item['image'] = pil_image
                        else:
                            print(f"Base64 이미지 변환 실패")
                            continue
                    else:
                        # 파일 경로인 경우 - 존재하는지 확인하고 로드
                        if os.path.exists(image_data):
                            try:
                                pil_image = Image.open(image_data).convert('RGB')
                                cleaned_item['image'] = pil_image
                            except Exception as e:
                                print(f"이미지 파일 로드 실패 ({image_data}): {e}")
                                continue
                        else:
                            print(f"이미지 파일이 존재하지 않음: {image_data}")
                            continue
                else:
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

def create_training_collator(processor):
    
    def collate_fn(batch):

        texts = []
        images_list = []
        messages_batch = []
        
        for item in batch:
            message = item['messages']
            messages_batch.append(message)
            
            cleaned_message = clean_and_prepare_message(message)
            
            text = processor.apply_chat_template(
                cleaned_message,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            
            sample_images = []
            if has_images_in_message(cleaned_message):
                try:
                    image_inputs, _ = process_vision_info(cleaned_message)
                    if image_inputs is not None:
                        sample_images = image_inputs  
                    else:
                        print(f"process_vision_info에서 None 반환")
                        sample_images = []

                except Exception as e:
                    print(f"이미지 처리 실패: {e}")
                    sample_images = []
            else:
                sample_images = []
            
            images_list.append(sample_images)
        
        # 이미지가 있는 샘플과 없는 샘플을 분리
        has_any_images = any(len(img_list) > 0 for img_list in images_list)
        
        try:
            inputs = processor.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                max_length=1536
            )
            
            inputs = apply_prompt_masking(inputs, processor, messages_batch)
            return inputs
            
        except Exception as e:
            print(f"배치 처리 실패: {e}")
            raise e  
    
    return collate_fn

def apply_prompt_masking(inputs, processor, messages_batch):

    labels = inputs["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    labels[labels == pad_id] = -100

    for i in range(labels.size(0)):
        try:
            messages = messages_batch[i]
            cleaned_message = clean_and_prepare_message(messages)
            input_messages = [msg for msg in cleaned_message if msg['role'] != 'assistant']

            input_text = processor.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            input_tokens = processor.tokenizer(
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            )
            input_length = input_tokens['input_ids'].size(1)
            
            mask_end_pos = min(input_length, inputs['input_ids'][i].size(0))
            
            if mask_end_pos >= inputs['input_ids'][i].size(0) - 1:
                seq_len = (inputs['input_ids'][i] != pad_id).sum().item()
                mask_end_pos = int(seq_len * 0.80)
                mask_end_pos = max(1, min(mask_end_pos, inputs['input_ids'][i].size(0) - 2))
            
            labels[i, :mask_end_pos] = -100
            
        except Exception as e:
            print(f"마스킹 실패 (샘플 {i}): {e}")
            # 폴백 
            seq_len = (inputs['input_ids'][i] != pad_id).sum().item()
            fallback_pos = int(seq_len * 0.60) 
            fallback_pos = max(1, min(fallback_pos, inputs['input_ids'][i].size(0) - 2))
            labels[i, :fallback_pos] = -100
    
    inputs["labels"] = labels
    return inputs

def setup_model_and_processor():
    
    # Quantization 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 모델 
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2", 
        cache_dir="../model/qwen/Qwen2.5-VL-7B-Instruct"
    )
    
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(
        model_name, 
        padding_side='right',        # 훈련 시 Right Padding
        min_pixels=256*28*28,        
        max_pixels=1280*28*28        
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"  - pad_token: {processor.tokenizer.pad_token}")
    print(f"  - eos_token: {processor.tokenizer.eos_token}")

    return model, processor 

def setup_lora_config():

    target_modules = [ 
        # Attention
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
    ]

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
def setup_training_args():

    config = SFTConfig(
        output_dir="finetuned",
        num_train_epochs=1,

        # Optimization
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1, 
        
        # 메모리 최적화
        gradient_checkpointing=True,  # 메모리 절약을 위한 gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False}, # 호환성/속도 확보
        gradient_accumulation_steps=8,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,

        optim="paged_adamw_8bit",
        learning_rate=2e-4, 
        lr_scheduler_type="linear", 
        logging_steps=100, 
        eval_strategy="epoch",  
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        push_to_hub=False,
        packing=False,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to=[],
                
        save_safetensors=True,              
    )
    
    return config

def visualize_training_metrics(trainer, file_path):
    def _aggregate_epoch_metrics(log_history):
        train_losses = {}
        eval_losses = {}

        for entry in log_history:
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            epoch_idx = int(math.floor(epoch)) if epoch >= 1 else 0

            if "loss" in entry:
                train_losses.setdefault(epoch_idx, []).append(float(entry["loss"]))
            if "eval_loss" in entry:
                eval_losses[epoch_idx] = float(entry["eval_loss"])

        epochs = sorted(set(list(train_losses.keys()) + list(eval_losses.keys())))
        rows = []
        for e in epochs:
            tr_loss = float(np.mean(train_losses[e])) if e in train_losses else None
            ev_loss = eval_losses.get(e, None)
            rows.append({"epoch": e, "train_loss": tr_loss, "eval_loss": ev_loss})
        return rows

    log_history = trainer.state.log_history if hasattr(trainer.state, "log_history") else []
    metrics_rows = _aggregate_epoch_metrics(log_history)

    if metrics_rows:
        try:
            df_metrics = pd.DataFrame(metrics_rows)
            csv_path = os.path.join(file_path, "training_metrics.csv")
            df_metrics.to_csv(csv_path, index=False)
            print(f"메트릭 CSV 저장: {csv_path}")
        except Exception as e:
            print(f"CSV 저장 실패: {e}")

        try:
            epochs = [r["epoch"] for r in metrics_rows]
            train_loss = [r["train_loss"] for r in metrics_rows]
            eval_loss = [r["eval_loss"] for r in metrics_rows]

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(epochs, train_loss, marker="o", label="train_loss")
            ax.plot(epochs, eval_loss, marker="o", label="eval_loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title("Training Loss")
            ax.grid(True)
            ax.legend()

            plt.tight_layout()
            img_path = os.path.join(file_path, "training_metrics.png")
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            print(f"시각화 저장: {img_path}")
            plt.close(fig)
        except Exception as e:
            print(f"시각화 생성 실패: {e}")
            
def main():
    # 재현성 
    setup_reproducibility(seed=42)
    
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--mode', choices=['default', 'sampling'], default='default')
    args = parser.parse_args()
    
    print("Training Start")
    dataset = load_preprocessed_training_data() 
    if dataset is None:
        print('훈련 데이터 로드 불가 ')
        return
    
    # Dataset Shuffling
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=42)
        print("데이터셋 셔플링 Done")
    else:
        print("데이터셋 셔플링 Skip")

    # Split    
    split_data = dataset.train_test_split(test_size=TEST_SIZE, seed=42)
    train_dataset = split_data['train']
    eval_dataset = split_data['test']
    
    print(f"[ 데이터 분할]") 
    print(f" train : {len(train_dataset)}")
    print(f" eval : {len(eval_dataset)}")
    
    model, processor = setup_model_and_processor()
    lora_config = setup_lora_config()
    
    training_args = setup_training_args()
    training_args.num_train_epochs = args.epochs
    
    print(f"[ 훈련 설정 ]")
    print(f" Epochs: {training_args.num_train_epochs}")
    print(f" Batch Size: {training_args.per_device_train_batch_size}")
    print(f" Learning Rate: {training_args.learning_rate}")
    print(f" Max Sequence Length: {MAX_SEQUENCE_LENGTH}")
    
    # 전처리 데이터 구조 분석
    sample = train_dataset[0]
    message = sample['messages']
    has_image = has_images_in_message(message)
    
    print(f"[데이터 구조 분석]")
    print(f"샘플 구조: {list(sample.keys())}")
    print(f"메시지 길이: {len(message)}")
    print(f"첫 번째 샘플 이미지 여부: {has_image}")
    # 이미지 샘플 개수
    image_count = sum(1 for item in train_dataset if has_images_in_message(item['messages']))
    print(f"이미지 포함 샘플: {image_count}개 ({image_count/len(train_dataset)*100:.1f}%)")
    
    # --mode sampling: 각 task별로 25% 샘플링
    if args.mode == 'sampling':
        print("mode=sampling: 각 task별 25% 샘플링")
        # group indices by task
        from collections import defaultdict
        task_to_indices = defaultdict(list)

        for idx, item in enumerate(train_dataset):
            task = None

            if 'task' in item:
                task = item['task']

            else:
                try:
                    messages = item.get('messages', [])
                    if messages and isinstance(messages, list):
                        for m in messages:
                            if m.get('role') == 'system':
                                txt = ''
                                for c in m.get('content', []):
                                    if c.get('type') == 'text':
                                        txt = c.get('text', '')
                                        break

                                # 간단한 키워드로 태스크 추정
                                if 'text-qa' in txt or 'text_qa' in txt or 'text-qa' in txt.lower():
                                    task = 'text_qa'
                                    break
                                if 'math_reasoning' in txt or 'math' in txt.lower():
                                    task = 'math_reasoning'
                                    break
                                if 'summarization' in txt or 'summary' in txt.lower():
                                    task = 'summarization'
                                    break
                except Exception:
                    task = 'unknown'

            if task is None:
                task = 'unknown'
            task_to_indices[task].append(idx)

        # sample 25% indices per task
        sampled_indices = []
        for task, indices in task_to_indices.items():
            if not indices:
                continue
            k = max(1, int(len(indices) * 0.25))
            # deterministic sampling with seed
            random.Random(42).shuffle(indices)
            sampled = indices[:k]
            sampled_indices.extend(sampled)

        sampled_indices = sorted(sampled_indices)
        print(f"원본 train size: {len(train_dataset)} -> After Sampling : {len(sampled_indices)}")
        train_dataset = train_dataset.select(sampled_indices)
        
    collator = create_training_collator(processor)
        
    # SFTTrainer 
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=collator,
        processing_class=processor.tokenizer,  # 텍스트 전용: tokenizer 
    )

    # 훈련 시작
    print("Start Training")
    print("="*50)
    try:
        trainer.train()
        print("훈련 완료")
                
        # 모델 저장
        trainer.save_model()
        print(f"모델 저장 완료")
        
    except Exception as e:
        print(f"훈련 오류: {e}")
        import traceback
        traceback.print_exc()
        return
            
    # 시각화
    visualize_training_metrics(trainer, FILE_PATH)
    
    # 결과 요약
    print(f" [ 훈련 결과 ] ") 
    print(f"  train samples : {len(train_dataset):,}")
    print(f"  eval samples : {len(eval_dataset):,}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Path : {training_args.output_dir}")
    
if __name__ == "__main__":
    main()

