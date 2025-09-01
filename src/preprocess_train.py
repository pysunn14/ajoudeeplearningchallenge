import os
import json
import pickle
import hashlib
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image
import io
import base64
from tqdm import tqdm
import argparse

# 설정 
FILE_PATH = '/hdd1/minseok/dev/contest/multimodal/'
TRAIN_PATH = 'dataset/deeplearningchallenge/deep_chal_multitask_dataset.parquet'

PREPROCESSED_DATA_PATH = FILE_PATH + 'preprocessed_inference' 
IMAGE_CACHE_DIR = os.path.join(FILE_PATH, "image_cache")
DOWNLOAD_REPORT_PATH = os.path.join(FILE_PATH, "download_report")

# 훈련용 TASK별 프롬프트 
def prompt_no_shot(task):
    
    base_system = '''<SYSTEM> You are a helpful multimodal assistant. Analyze the user's query to identify the specific <USER TASK> and generate a response that strictly follows the format shown in the corresponding example below.
<OUTPUT_RULE> IMPORTANT: You MUST NOT include the <Response> tag.
'''
    task_specific_prompts = {
        'summarization': '''<TASK> summarization
<Instruction> Provide a high-level summary of the document's primary function and purpose. Focus on what the document establishes, requires, and authorizes, except procedural details.
</SYSTEM>''',

        'math_reasoning': '''<TASK> math_reasoning
<Instruction> Solve the math problem by your reasoning step-by-step. You must show all reasoning steps. 
<Format> Provide your all reasoning steps and final answer with "####" (4 hash) followed by the numerical result.
</SYSTEM>''',

        'captioning': '''<TASK> captioning
<Instruction> Describe the provided image in rich detail. Focus on the main subjects, their attributes (like color and clothing), their actions or interactions, the background setting, and the overall composition of the scene.
<Example> 
    [Image: A picture of a golden retriever playing in a park]
    <Response> The image is the cover of a book titled "Devil's Highway" by Hank Janson. The cover features a man in a blue suit and a woman in a pink dress. The man is holding a gun and the woman is looking up at him with a surprised expression on her face. The title of the book is written in red and yellow letters at the top of the cover, with the subtitle "Nine Million Sale" written in smaller letters below. The number "26" is written on the bottom right corner. The background is a dark green color.
</SYSTEM>''',

        'vqa': '''<TASK> vqa
<Instruction> Answer the question about the image in a single word or a very short phrase.
<Format> single word / very short phrase, not long sentence. Don't attach 'Response' keyword.
</SYSTEM>''',

        'text_qa': '''<TASK> text-qa  
<Instruction> Based on the provided input, answer the raw text, separate them with a comma (,)
each answer should be single word / very short phrase from the input
<Important> DO NOT repeat the questions in your output!! 
<Format> answer1, answer2, answer3 ...
</SYSTEM>'''
    }
    
    task_prompt = task_specific_prompts.get(task)
    return base_system + task_prompt

def prompt_one_shot(task):
    base_system = '''<SYSTEM> You are a helpful multimodal assistant. Analyze the user's query to identify the specific <USER TASK> and generate a response that strictly follows the format shown in the corresponding example below.
<OUTPUT_RULE> IMPORTANT: You MUST NOT include the <Response> tag.
'''
    
    task_specific_prompts = {
        'summarization': '''<TASK> summarization
<Instruction> Provide a high-level summary of the document's primary function and purpose. Focus on what the document establishes, requires, and authorizes, except procedural details.
<Example>
    <Input> [A legal text about restructuring the federal budget.]
    <Response> Federal Budget Structure Act of 1993 - Amends Federal law to require that the budget the President submits to the Congress be a unified budget comprising an operating budget and a capital budget, each presented separately for total funds, Federal funds, and trust funds.  Restricts the capital budget to the major activities, projects, and programs supporting the acquisition, construction, alteration, and rehabilitation of capital assets.  Includes all other items in the operating budget. 
Requires the following reports to specified congressional committees on capital activities and operating activities associated with:  (1) roadways and bridges, airports and airway facilities, and mass transportation systems; (2) waste water treatment and related facilities; (3) water resource projects; and (4) public buildings.
</SYSTEM>''',

        'math_reasoning': '''<TASK> math_reasoning
<Instruction> Solve the math problem by your reasoning step-by-step. You must show all reasoning steps. 
<Format> Provide your all reasoning steps and final answer with "####" (4 hash) followed by the numerical result.
<Example> 
    <Input> A grocery store sells apples for $2 each and bananas for $1 each. If I buy 3 apples and 5 bananas, how much do I spend in total?
    <Response> 
    Reasoning:
    I need to calculate the total cost of the apples and the bananas. 
    Cost of apples: 3 * 2 = <<3*2=6>> 
    Cost of bananas: 5 * 1 = <<5*1=5>>
    Total cost: 6 + 5 = <<6+5=11>>
    #### 11  
</SYSTEM>''',

        'captioning': '''<TASK> captioning
<Instruction> Describe the provided image in rich detail. Focus on the main subjects, their attributes (like color and clothing), their actions or interactions, the background setting, and the overall composition of the scene.
<Example> 
    [Image: A picture of a golden retriever playing in a park]
    <Response> The image is the cover of a book titled "Devil's Highway" by Hank Janson. The cover features a man in a blue suit and a woman in a pink dress. The man is holding a gun and the woman is looking up at him with a surprised expression on her face. The title of the book is written in red and yellow letters at the top of the cover, with the subtitle "Nine Million Sale" written in smaller letters below. The number "26" is written on the bottom right corner. The background is a dark green color.
</SYSTEM>''',

        'vqa': '''<TASK> vqa
<Instruction> Answer the question about the image in a single word or a very short phrase.
<Format> single word / very short phrase, not long sentence. Don't attach 'Response' keyword.
<Example>
    [Image: A picture of three red apples on a table]
    <Question> How many apples are there?
    <Response> Three
</SYSTEM>''',

        'text_qa': '''<TASK> text-qa  
<Instruction> Based on the provided input, answer the raw text, separate them with a comma (,)
each answer should be single word / very short phrase from the input
<Important> DO NOT repeat the questions in your output!! 
<Format> answer1, answer2, answer3 ...
<Example>   
    <Input> The patient showed a high fever of 39 degrees Celsius. The doctor prescribed medicine. After taking the medicine, his condition improved.
    <Question> ["What was the patient's temperature?", "Did the medicine work?"]
    <Response> 39 degrees Celsius, yes
</SYSTEM>'''
    }
    
    task_prompt = task_specific_prompts.get(task)
    return base_system + task_prompt

def get_user_query(task, input_text="", question_text=""):
    user_query = f'''<USER_QUERY>
  <USER_TASK>{task}</USER_TASK>
  <USER_INPUT>'''
    
    if task in ['summarization', 'math_reasoning', 'text_qa']:
        # 텍스트 task
        user_query += f'''
    <Input>{input_text}</Input>'''
        
        if task == 'text_qa' and question_text:
            user_query += f'''
    <Question>{question_text}</Question>'''

    elif task == 'vqa':
        user_query += f'''
    <Question>{question_text}</Question>'''
    
    user_query += '''
  </USER_INPUT>
</USER_QUERY>
'''    
    return user_query

def load_successful_urls():
    successful_urls_path = os.path.join(DOWNLOAD_REPORT_PATH, "successful_urls.txt")
    
    if not os.path.exists(successful_urls_path):
        print('다운로드 이미지 먼저 실행하시오')
        return set()
    
    successful_urls = set()
    with open(successful_urls_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url:
                successful_urls.add(url)
    
    print(f"다운로드 성공한 URL: {len(successful_urls)}개")
    return successful_urls

# messages 포맷으로 반환 
def format_training_sample(sample):
    task = sample['task']
    input_text = str(sample.get('input', '')) if sample.get('input') is not None else ''
    question_text = str(sample.get('question', '')) if sample.get('question') is not None else ''
    output_text = str(sample.get('output', '')) if sample.get('output') is not None else ''
    
    # [ text-qa : 콤마 구분 문자열로 파싱 로직 ]
    if task == 'text_qa' and output_text:
        try:
            # Is dict type ?
            output_data = sample.get('output', '')
            if isinstance(output_data, dict):

                if 'input_text' in output_data and isinstance(output_data['input_text'], list):
                    output_text = ', '.join(output_data['input_text'])
                    print(f"text_qa 변환 (dict): {len(output_data['input_text'])}개")

            else:
                # Is Python Literal type?
                import ast
                output_dict = ast.literal_eval(output_text)
                if 'input_text' in output_dict and isinstance(output_dict['input_text'], list):
                    output_text = ', '.join(output_dict['input_text'])
                    print(f"text_qa 변환 (literal_eval): {len(output_dict['input_text'])}개")

        except (ValueError, SyntaxError, KeyError, TypeError) as e:
            print(f"파싱 실패 하여 원본 사용")
                
    # System Prompt
    system_content = prompt_one_shot(task)
    
    # Message
    message = [
        {"role": "system", "content": [{"type": "text", "text": system_content}]}
    ]
    
    # User Query
    user_content = [{'type':'text', 'text': get_user_query(task, input_text, question_text)}]

    # Image Case
    has_image = (sample.get('image') is not None and str(sample['image']).strip() != "")

    if has_image:
        user_content.append({"type": "image", "image": sample['image']}) 

    else:
        # 이미지 없어도 None image 필드 추가하여 스키마 통일
        user_content.append({"type": "image", "image": ""}) 
    
    message.append({"role": "user", "content": user_content})

    # Training Target    
    message.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})

    return message

# URL to Cache
def _url_to_cache_path(url: str) -> str:
    parsed = url.split("?")[0]
    ext = os.path.splitext(parsed)[1] if "." in parsed else ".jpg"
    name = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(IMAGE_CACHE_DIR, name + ext)

def preprocess_generator(dataset, successful_urls, text_only_mode=False):

    failed_count = 0
    image_success = 0
    image_failed = 0
    no_downloaded = 0
    task_filtered = 0
    
    # Text-Only Mode
    text_only_tasks = {'text_qa', 'math_reasoning', 'summarization'}

    for i, sample in enumerate(tqdm(dataset, desc="Preprocessing - Training")):
        try:
            # text-only mode
            task = sample.get('task', '')
            if text_only_mode and task not in text_only_tasks:
                task_filtered += 1
                continue  # captioning, vqa 태스크는 X
            
            # Information
            training_sample = {
                'id': i,
                'task': task,
                'input_type': sample['input_type'],
                'input': sample['input'],
                'question': sample.get('question', ''),
                'output': sample.get('output', ''),
                'image': None 
            }
            
            if not text_only_mode and sample['input_type'] == 'image':
                input_data = sample['input']

                if input_data is None or input_data == "":
                    image_failed += 1
                elif isinstance(input_data, str) and input_data.startswith("http"):
                    if input_data in successful_urls:
                        cache_path = _url_to_cache_path(input_data)
                        if os.path.exists(cache_path):
                            training_sample['image'] = cache_path 
                            image_success += 1
                        else:
                            image_failed += 1
                    else:
                        image_failed += 1
                        no_downloaded += 1
                        continue  
                else:
                    training_sample['image'] = input_data
                    image_success += 1
            # Text Only Mode 
            elif text_only_mode:
                training_sample['image'] = None

            # Message Generator
            try:
                messages = format_training_sample(training_sample)
                yield {'id': i, 'messages': messages}
                
            except Exception as format_error:
                print(f"샘플 {i} message 생성 오류 : {format_error}")
                print(f"샘플 task: {training_sample.get('task', 'unknown')}")
                print(f"샘플 input type: {training_sample.get('input_type', 'unknown')}")
                failed_count += 1
                continue

        except Exception as e:
            print(f"샘플 {i} 처리 오류 : {e}")
            print(f"샘플 Info: {sample.keys() if hasattr(sample, 'keys') else 'unknown'}")
            failed_count += 1
            continue
    
    # 제너레이터가 끝나면 최종 리포트 출력
    mode_text = "TEXT-ONLY MODE" if text_only_mode else "MULTIMODAL MODE"
    print('[ 훈련 전처리 완료 ]', mode_text)    

    if text_only_mode:
        print(f"태스크 필터링: {task_filtered}개 skip (captioning, vqa)")
    else:
        print(f"이미지 성공: {image_success}, 실패: {image_failed}, 미다운로드: {no_downloaded}")
    print(f"전체 실패: {failed_count}")

def create_training_dataset(successful_urls, output_path=FILE_PATH + 'preprocessed_training', text_only_mode=False):

    mode_text = "TEXT-ONLY MODE" if text_only_mode else "MULTIMODAL MODE"
    print('[ 훈련 전처리 시작 ]', mode_text)  

    if text_only_mode:
        print("학습 TASK : text_qa, math_reasoning, summarization")
        print("제외 TASK : captioning, vqa")
    
    # data load
    original_dataset = load_dataset('parquet', data_files=FILE_PATH + TRAIN_PATH, split='train')
    print(f"원본 데이터 로드: {len(original_dataset)}")
    
    os.makedirs(output_path, exist_ok=True)
    dataset_save_path = os.path.join(output_path, "training_dataset")
    
    batch_size = 500 
    total_samples = 0
    saved_datasets = []
    
    for batch_num, start_idx in enumerate(range(0, len(original_dataset), batch_size)):
        end_idx = min(start_idx + batch_size, len(original_dataset))
        batch_dataset = original_dataset.select(range(start_idx, end_idx))
        
        print(f"BATCH {batch_num+1} processing: {start_idx}-{end_idx}")
        
        # 배치 처리 - text_only_mode 전달
        batch_samples = []
        generator = preprocess_generator(batch_dataset, successful_urls, text_only_mode=text_only_mode)
        
        for sample in generator:
            batch_samples.append(sample)
        
        if batch_samples:  
            # 배치를 Dataset으로 변환하고 임시 저장
            batch_dataset_obj = Dataset.from_list(batch_samples)
            temp_path = os.path.join(output_path, f"temp_batch_{batch_num}")
            batch_dataset_obj.save_to_disk(temp_path)
            saved_datasets.append(temp_path)
            
            total_samples += len(batch_samples)
            print(f"BATCH {batch_num+1} completed : {len(batch_samples)}")
        
        # 메모리 정리
        del batch_samples
        del batch_dataset
        if 'batch_dataset_obj' in locals():
            del batch_dataset_obj
    
    print(f"[ 배치 처리 완료 ] {total_samples}개 ")
    
    # 모든 배치 데이터셋을 하나로 합치기
    print("Batch Data Merging")
    if saved_datasets:
        datasets_to_merge = []
        for temp_path in saved_datasets:
            temp_dataset = Dataset.load_from_disk(temp_path)
            datasets_to_merge.append(temp_dataset)
        
        # 데이터셋 합치기
        final_dataset = concatenate_datasets(datasets_to_merge)
        
        # 최종 저장
        final_dataset.save_to_disk(dataset_save_path)
        
        # 임시 파일들 정리
        print("임시 파일 정리")
        for temp_path in saved_datasets:
            import shutil
            shutil.rmtree(temp_path)
    else:
        print("저장할 샘플 없음")
        return None
    
    # 메타데이터 저장
    metadata = {
        'total_samples': total_samples,
        'preprocessing_type': 'training_optimized_batch',
        'text_only_mode': text_only_mode,
        'allowed_tasks': ['text_qa', 'math_reasoning', 'summarization'] if text_only_mode else 'all',
        'excluded_tasks': ['captioning', 'vqa'] if text_only_mode else 'none'
    }
    
    with open(os.path.join(output_path, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f" 저장 완료: {output_path}")
    return final_dataset

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Preprocessing - Training') 
    parser.add_argument('--mode', choices=['multimodal', 'text'], default='multimodal')
    
    args = parser.parse_args()
    text_only_mode = (args.mode == 'text') 
    
    # 성공 URL 로드
    successful_urls = load_successful_urls()
    if not successful_urls and not text_only_mode:
        print("다운로드된 이미지가 없음. 먼저 download_images.py를 실행")
        print("또는 --mode text로 텍스트 전용 모드를 사용")
        exit(0)
        
    if text_only_mode:
        print("TEXT ONLY MODE : text-qa, summarization, math_reasoning")
        successful_urls = set()  
        
    else:
        print(" MULTIMODAL : ALL TASK ")

    create_training_dataset(successful_urls=successful_urls, text_only_mode=text_only_mode)

if __name__ == "__main__":
    main()

    