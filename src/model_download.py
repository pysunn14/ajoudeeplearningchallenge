# "Qwen/Qwen2.5-VL-7B-Instruct" model download script
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

# Load the model and processor
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir="../model/qwen/Qwen2.5-VL-7B-Instruct"
)       

processor = AutoProcessor.from_pretrained(model_name, padding_side='left')

