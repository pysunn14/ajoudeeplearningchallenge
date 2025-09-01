import os
import hashlib
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
from PIL import Image
import io
from tqdm import tqdm
from datasets import load_dataset
import json
from collections import defaultdict

FILE_PATH = '/hdd1/minseok/dev/contest/multimodal/'
TRAIN_SAMPLE_PATH = 'dataset/deeplearningchallenge/deep_chal_multitask_dataset.parquet'
TEST_SAMPLE_PATH = 'dataset/deeplearningchallenge/deep_chal_multitask_dataset_test.parquet'
IMAGE_CACHE_DIR = os.path.join(FILE_PATH, "image_cache")
DOWNLOAD_REPORT_PATH = os.path.join(FILE_PATH, "download_report")

os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_REPORT_PATH, exist_ok=True)

download_stats = {
    'total_attempts': 0,
    'successful': 0,
    'failed': 0,
    'already_cached': 0,
    'failure_reasons': defaultdict(int),
    'failed_urls': [],
    'successful_urls': []
}

def _create_session(retries=3, backoff_factor=1.0):
    s = requests.Session()
    retry = Retry(
        total=retries,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "qwen-finetune/1.0"})
    return s

def _url_to_cache_path(url: str) -> str:
    parsed = url.split("?")[0]
    ext = os.path.splitext(parsed)[1] if "." in parsed else ".jpg"
    name = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(IMAGE_CACHE_DIR, name + ext)

def _download_single_image(session, url: str, timeout: int = 15):
    global download_stats
    download_stats['total_attempts'] += 1
    
    cache_path = _url_to_cache_path(url)
    
    # 이미 캐시에 있으면 skip
    if os.path.exists(cache_path):
        download_stats['already_cached'] += 1
        download_stats['successful'] += 1
        download_stats['successful_urls'].append(url)
        return cache_path
    
    try:
        resp = session.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        
        # 이미지 형식 검증 (PIL)
        content = b''
        for chunk in resp.iter_content(1024 * 32):
            if chunk:
                content += chunk
        
        # 유효성 검사
        try:
            img = Image.open(io.BytesIO(content))
            img.verify()  
        except Exception:
            raise ValueError("Invalid image format")
        
        with open(cache_path + ".tmp", "wb") as f:
            f.write(content)
        os.replace(cache_path + ".tmp", cache_path)
        
        download_stats['successful'] += 1
        download_stats['successful_urls'].append(url)
        return cache_path
        
    except Exception as e:
        download_stats['failed'] += 1
        
        # 실패 원인 분류
        error_str = str(e)
        if "404" in error_str:
            download_stats['failure_reasons']['404_not_found'] += 1
        elif "403" in error_str:
            download_stats['failure_reasons']['403_forbidden'] += 1
        elif "429" in error_str:
            download_stats['failure_reasons']['429_rate_limit'] += 1
        elif "timeout" in error_str.lower():
            download_stats['failure_reasons']['timeout'] += 1
        elif "Invalid image" in error_str:
            download_stats['failure_reasons']['invalid_image'] += 1
        else:
            download_stats['failure_reasons']['other'] += 1
        
        download_stats['failed_urls'].append({
            'url': url,
            'error': error_str,
            'domain': url.split('/')[2] if url.startswith('http') else 'unknown'
        })
        
        # 임시 파일 정리
        if os.path.exists(cache_path + ".tmp"):
            try:
                os.remove(cache_path + ".tmp")
            except:
                pass
        return None

def collect_image_urls_from_dataset(dataset, dataset_name):
    
    urls = set()
    url_to_samples = defaultdict(list)  # URL별 샘플 인덱스 매핑
    
    for idx, item in enumerate(dataset):
        if item.get("input_type") == "image":
            inp = item.get("input")
            if isinstance(inp, str) and inp.startswith("http"):
                urls.add(inp)
                url_to_samples[inp].append(idx)
    
    print(f"{dataset_name}: {len(urls)}개")
    print(f"{dataset_name}: image task sample : {sum(len(samples) for samples in url_to_samples.values())}")
    
    return list(urls), url_to_samples

def collect_all_image_urls(include_train=True, include_test=True):
    all_urls = set()
    all_url_mappings = {}
    
    if include_train:
        print("TRAIN PROCESSING")
        try:
            train_dataset = load_dataset('parquet', data_files=FILE_PATH + TRAIN_SAMPLE_PATH, split='train')
            print(f"훈련 데이터셋 크기: {len(train_dataset)}")
            
            train_urls, train_url_to_samples = collect_image_urls_from_dataset(train_dataset, "TRAIN_SET")
            all_urls.update(train_urls)
            all_url_mappings['train'] = {
                'url_to_samples': dict(train_url_to_samples),
                'total_unique_urls': len(train_urls),
                'total_image_samples': sum(len(samples) for samples in train_url_to_samples.values())
            }
        except Exception as e:
            print(f"훈련 데이터셋 로드 실패: {e}")
            all_url_mappings['train'] = None
    
    if include_test:
        print("TEST PROCESSING")
        try:
            test_dataset = load_dataset('parquet', data_files=FILE_PATH + TEST_SAMPLE_PATH, split='train')
            print(f"테스트 데이터셋 크기: {len(test_dataset)}")
            
            test_urls, test_url_to_samples = collect_image_urls_from_dataset(test_dataset, "TEST_SET")
            all_urls.update(test_urls)
            all_url_mappings['test'] = {
                'url_to_samples': dict(test_url_to_samples),
                'total_unique_urls': len(test_urls),
                'total_image_samples': sum(len(samples) for samples in test_url_to_samples.values())
            }
        except Exception as e:
            print(f"테스트 데이터셋 로드 실패: {e}")
            all_url_mappings['test'] = None
    
    # 중복 제거된 전체 통계
    print(f"[ 전체 통계 ]")
    print(f"URLs: {len(all_urls)}")
    
    if all_url_mappings.get('train'):
        print(f"  - 훈련셋 고유 URL: {all_url_mappings['train']['total_unique_urls']}")
        print(f"  - 훈련셋 이미지 샘플: {all_url_mappings['train']['total_image_samples']}")
    
    if all_url_mappings.get('test'):
        print(f"  - 테스트셋 고유 URL: {all_url_mappings['test']['total_unique_urls']}")
        print(f"  - 테스트셋 이미지 샘플: {all_url_mappings['test']['total_image_samples']}")
    
    # URL-샘플 매핑 
    all_url_mappings['combined'] = {
        'total_unique_urls': len(all_urls),
        'datasets_processed': []
    }
    
    if include_train:
        all_url_mappings['combined']['datasets_processed'].append('train')
    if include_test:
        all_url_mappings['combined']['datasets_processed'].append('test')
    
    with open(os.path.join(DOWNLOAD_REPORT_PATH, "url_mapping_combined.json"), "w", encoding="utf-8") as f:
        json.dump(all_url_mappings, f, indent=2, ensure_ascii=False)
    
    return list(all_urls), all_url_mappings

def download_images_parallel(urls, max_workers=16):
    
    session = _create_session(retries=3, backoff_factor=1.0)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single_image, session, url): url for url in urls}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            url = futures[future]
            try:
                result = future.result()
                if result is None:
                    time.sleep(0.02)
            except Exception as e:
                print(f"예외 발생 {url}: {e}")
                time.sleep(0.02)

def generate_download_report(dataset_info):
    print("다운로드 결과")
    
    total = download_stats['total_attempts']
    success = download_stats['successful']
    failed = download_stats['failed']
    cached = download_stats['already_cached']
    
    if total > 0:
        success_rate = (success / total) * 100
        failure_rate = (failed / total) * 100
        cache_rate = (cached / total) * 100
        
        print(f"[ 다운로드 통계 ]")
        print(f"총 시도: {total:,}개")
        print(f"성공: {success:,}개 ({success_rate:.1f}%)")
        print(f"실패: {failed:,}개 ({failure_rate:.1f}%)")
        print(f"캐시 활용: {cached:,}개 ({cache_rate:.1f}%)")
        
        if download_stats['failure_reasons']:
            print(f"실패 원인:")
            for reason, count in download_stats['failure_reasons'].items():
                percentage = (count / failed) * 100 if failed > 0 else 0
                print(f"  - {reason}: {count}개 ({percentage:.1f}%)")
    
    # 상세 리포트 저장
    report = {
        'download_statistics': download_stats,
        'dataset_info': dataset_info,
        'summary': {
            'total_attempts': total,
            'successful': success,
            'failed': failed, 
            'success_rate': success_rate if total > 0 else 0,
            'failure_rate': failure_rate if total > 0 else 0
        }
    }
    
    with open(os.path.join(DOWNLOAD_REPORT_PATH, "download_report_combined.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(DOWNLOAD_REPORT_PATH, "successful_urls.txt"), "w", encoding="utf-8") as f:
        for url in download_stats['successful_urls']:
            f.write(url + "\n")
    
    with open(os.path.join(DOWNLOAD_REPORT_PATH, "failed_urls_combined.txt"), "w", encoding="utf-8") as f:
        for failed_item in download_stats['failed_urls']:
            f.write(f"{failed_item['url']}\t{failed_item['error']}\n")
    
    return success_rate

def main():
    parser = argparse.ArgumentParser(description='Image Downloading (훈련셋 + 테스트셋)')
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    # 처리할 데이터셋 결정
    include_train = not args.test_only
    include_test = not args.train_only
    
    if args.train_only:
        print("TRAIN_ONLY")
    elif args.test_only:
        print("TEST_ONLY")
    else:
        print("TRAIN_AND_TEST")

    urls, dataset_info = collect_all_image_urls(include_train=include_train, include_test=include_test)
    
    if not urls:
        print("이미지 URL이 없음")
        return
    
    download_images_parallel(urls, max_workers=args.workers)
    
    success_rate = generate_download_report(dataset_info)
    
    print("[ 다운로드 완료 ]")
    print(f"성공률: {success_rate:.1f}%")
    print(f"cache: {IMAGE_CACHE_DIR}")
    print(f"report: {DOWNLOAD_REPORT_PATH}")

if __name__ == "__main__":
    main()