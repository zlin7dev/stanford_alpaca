import transformers

MODEL_PATH = '/home/zhen7/cache/huggingface/hub/decapoda-research/llama-7b-hf/'

import os

if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)