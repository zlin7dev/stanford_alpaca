from importlib import reload

import transformers

import _settings

reload(_settings)


from huggingface_hub import snapshot_download
from transformers.trainer_pt_utils import get_module_class_from_name

snapshot_download(repo_id='decapoda-research/llama-7b-hf', local_dir=_settings.MODEL_PATH)


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