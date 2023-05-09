from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from utils import messages_to_prompt
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


class ChatGPT():
    def __init__(self, config):
        self.config = config
        self.model = ChatOpenAI(model_name=self.config['model_name'],
                                temperature=self.config['temperature'], 
                                max_tokens=self.config['max_tokens'])
        print(self.model.json)

    def generate(self, history, name):
        messages = history.get_messages()
        # For debug purposes
        prompt = messages_to_prompt(messages)
        print(prompt)
        messages.append(AIMessage(content=f"{name}:"))
        response = self.model(messages=messages).content.strip()
        response = f"{name}: " + response
        token_length = self.model.get_num_tokens(response)

        print(response)
        return response, token_length

    def get_num_tokens(self, text):
        return self.model.get_num_tokens(text)

class OpenAIGPT():
    def __init__(self, config):
        self.config = config
        self.model = OpenAI(model_name=self.config['model_name'],
                            temperature=self.config['temperature'], 
                                max_tokens=self.config['max_tokens'])
        print(self.model.json)

    def generate(self, history, name):
        prompt = history.get_prompt() + f"{name}:"
        print(prompt)
        response = self.model(prompt).strip()
        response = f"{name}: " + response
        token_length = self.model.get_num_tokens(response)
        print(response)
        return response, token_length

    def get_num_tokens(self, text):
        return self.model.get_num_tokens(text)
    

#TODO: update to work properly
class FlanUL2():
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.model = self.init_model()


    def init_model(self):
        model_config = T5Config.from_pretrained(self.config['model_name'])
        model_config.max_length = self.config['max_tokens']

        max_memory={i: self.config['vram_per_device'] for i in self.config['devices']}
        max_memory[self.config["devices"][0]] = self.config['first_device_limit']  # to fit lm_head to the same device as the inputs

        self.input_device = torch.device(f'cuda:{self.config["devices"][0]}')

        with init_empty_weights():
            model = T5ForConditionalGeneration(model_config)
            device_map = infer_auto_device_map(model, no_split_module_classes=["T5Block"], dtype=torch.float16, max_memory=max_memory)
        device_map['lm_head'] = device_map["decoder.embed_tokens"]

        return T5ForConditionalGeneration.from_pretrained(self.config['model_name'], device_map=device_map, torch_dtype=torch.float16)

    def generate(self, history, name):
        prompt = history.get_prompt() + f"{name}:"
        print(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=self.config['max_tokens'], truncation=True).input_ids.to(self.input_device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            num_return_sequences=1,
            remove_invalid_values=True,
            return_dict_in_generate=True,
        )
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
        response = f"{name}: " + response
        print(response)
        return response

class FlanT5():
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        print("Loading model...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config['model_name'], torch_dtype=torch.float16)
        print("Model loaded")
        self.device = torch.device(f'cuda:{self.config["device"]}')
        self.model.to(self.device)

    def generate(self, history, name):
        prompt = history.get_prompt() + f"{name}:"
        print(prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=self.config['max_tokens'], truncation=True).input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            num_return_sequences=1,
            remove_invalid_values=True,
            return_dict_in_generate=True,
        )
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
        response = f"{name}: " + response
        print(response)
        return response


class Vicuna():
    def __init__(self, config):
        self.config = config
        self.tokenizer = LlamaTokenizer.from_pretrained(self.config['model_path'])
        print("Loading model...")
        self.model = LlamaForCausalLM.from_pretrained(self.config['model_path'], torch_dtype=torch.float16)
        print("Model loaded")
        self.device = torch.device(f'cuda:{self.config["device"]}')
        self.model.to(self.device)

    def generate(self, history, name):
        prompt = history.get_prompt() + f"{name}:"
        print('***********************************')
        print(prompt, end=' ')
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=self.config['max_tokens'], truncation=True).input_ids.to(self.device)
        input_len = input_ids.shape[1]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=self.config['temperature'],
            do_sample=True,
            num_return_sequences=1,
            remove_invalid_values=True,
            return_dict_in_generate=True,
        )
        response = self.tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True).strip()
        print(response)
        response = f"{name}: " + response

        return response


type_to_class = {
    'chatgpt': ChatGPT,
    'openai': OpenAIGPT,
    'flan-ul2': FlanUL2,
    'flan-t5': FlanT5,
    'vicuna': Vicuna,
}

def load_model(config):
    model_type = config['type']
    return type_to_class[model_type](config)
