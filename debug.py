import os
from importlib import reload

import transformers

import _settings

reload(_settings)


from huggingface_hub import snapshot_download
from transformers.trainer_pt_utils import get_module_class_from_name

json_file = os.path.join(_settings.MODEL_PATH, 'tokenizer_config.json')

if not os.path.isfile(json_file):
        snapshot_download(repo_id='decapoda-research/llama-7b-hf', local_dir=_settings.MODEL_PATH)

import json

with open(json_file) as fin:
        dd = json.load(fin)
dd.update({"tokenizer_class": "LlamaTokenizer"})
with open(json_file, 'w') as fout:
        json.dump(dd, fout)


if True:
        model = transformers.AutoModelForCausalLM.from_pretrained(
                #model_args.model_name_or_path,
                _settings.MODEL_PATH,
                cache_dir=None,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
                #model_args.model_name_or_path,
                _settings.MODEL_PATH,
                cache_dir=None,
                model_max_length=512,
                padding_side="right",
                use_fast=False,
        )


        transformer_cls = get_module_class_from_name(model, 'LlamaDecoderLayer')

"""
full_completion = model_current.generate(
    inputs=input_ids,
    attention_mask=attention_mask,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    max_new_tokens=600,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
"""