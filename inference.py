from transformers import (AutoModelForCausalLM, AutoModelWithLMHead,
                          AutoTokenizer, pipeline)

from _settings import FINETUNED_PATH
from train import PROMPT_DICT, preprocess

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH,
                                          model_max_length=512,
                                          padding_side="right",
                                          use_fast=False,
                                          )
model = AutoModelForCausalLM.from_pretrained(FINETUNED_PATH)
device = 'cuda:7'
model.to(device)

prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
"""
# Define your prompt
MY_PROMPT = "Give three tips for staying healthy."

list_data_dict = [{
    'instruction': MY_PROMPT,
    'input': "",
    'output': '',
    #"output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    }]

sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]

targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
data_dict = preprocess(sources, targets, tokenizer)
input_ids = data_dict['input_ids']
"""

MY_PROMPT = "Who invented neural networks?"
text = prompt_no_input.format_map({ 'instruction': MY_PROMPT, 'input': "", 'output': ''})
example = tokenizer(text, return_tensors="pt", padding="longest", max_length=512, truncation=True)
example = {_[0]: _[1].to(device) for _ in example.items()}
out = model.generate(input_ids=example['input_ids'], attention_mask=example['attention_mask'], temperature=0.7,top_p=0.9,do_sample=True,num_beams=1,max_new_tokens=600,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,)
print(f"Question: {MY_PROMPT}")
print(tokenizer.decode(out[0][len(example['input_ids'][0]):]))




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