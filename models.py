from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from utils import chat_history_to_prompt


class ChatGPT():
    def __init__(self, config):
        self.config = config
        self.model = ChatOpenAI(model_name=self.config['model_name'],
                                temperature=self.config['temperature'], 
                                max_tokens=self.config['max_tokens'])
        print(self.model.json)

    def generate(self, history):
        return self.model(messages=history.messages).content


class OpenAIGPT():
    def __init__(self, config):
        self.config = config
        self.model = OpenAI(model_name=self.config['model_name'],
                            temperature=self.config['temperature'], 
                                max_tokens=self.config['max_tokens'])
        print(self.model.json)

    def generate(self, history):
        prompt = chat_history_to_prompt(history)
        return self.model(prompt)



type_to_class = {
    'chatgpt': ChatGPT,
    'openai': OpenAIGPT
}

def load_model(config):
    model_type = config['type']
    return type_to_class[model_type](config)
