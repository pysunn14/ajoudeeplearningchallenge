"""스크립트 실행 위치와 무관한 프로젝트 경로 및 평가용 캐시 설정."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def configure_evaluation_cache():
    # 모델 라이브러리를 불러오기 전에 적용하되 사용자가 지정한 캐시는 보존한다.
    hf_home = os.environ.setdefault('HF_HOME', project_path('hf_cache'))
    defaults = {
        'TRANSFORMERS_CACHE': os.path.join(hf_home, 'transformers'),
        'HF_DATASETS_CACHE': os.path.join(hf_home, 'datasets'),
        'TORCH_HOME': os.path.join(hf_home, 'torch'),
        'CLIP_CACHE_DIR': os.path.join(hf_home, 'clip'),
    }
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    for name, default in defaults.items():
        Path(os.environ.setdefault(name, default)).mkdir(parents=True, exist_ok=True)
