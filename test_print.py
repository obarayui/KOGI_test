import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

import time
import sys
import random
import IPython
from google.colab import output

class T5ModelWrapper:
    def __init__(self, model_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self, text):
        inputs = self.tokenizer.batch_encode_plus(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs.input_ids.to(torch.long)
        attention_mask = inputs.attention_mask.to(torch.long)
        
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return generated_text[0]

# モデルのパスを指定してラッパーオブジェクトを作成
model_path = "parasora/KOGI"
t5_wrapper = T5ModelWrapper(model_path)

# 文字列からモデルの出力を得る
input_text = "あいうえお"
output_text = t5_wrapper.generate(input_text)
