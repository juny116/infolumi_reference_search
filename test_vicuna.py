from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from utils import chat_history_to_prompt
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM, 
    T5ForConditionalGeneration, 
    T5Config,
    LlamaForCausalLM, 
    LlamaTokenizer
)
import torch
import copy

path = "/data/transformers/vicuna-13b"

class Vicuna():
    def __init__(self, config):
        self.config = config
        self.tokenizer = LlamaTokenizer.from_pretrained(path)
        print("Loading model...")
        self.model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        print("Model loaded")
        self.device = torch.device(f'cuda:{self.config["device"]}')
        self.model.to(self.device)

    def generate(self, prompt):
        print(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=self.config['max_tokens'], truncation=True).input_ids.to(self.device)
        input_len = input_ids.shape[1]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            num_return_sequences=1,
            remove_invalid_values=True,
            return_dict_in_generate=True,
        )
        response = self.tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()

        print("Vicuna: "+response)
        return response
    
if __name__ == "__main__":
    config = {
        "model_path": path,
        "device": 0,
        "max_tokens": 2048,
    }
    vicuna = Vicuna(config)

    while True:
        prompt = input("Prompt: ")
        vicuna.generate(prompt)
