import re
from typing import List

from pydantic import BaseModel

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


class ChatMessageHistoryWithTokenLength(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = []
    max_length: int = 1024

    def __init__(self, system_message: SystemMessage = None) -> None:
        super().__init__()
        self.set_system_message(system_message)

    def add_user_message(self, message: str, token_length: int) -> None:
        self.messages.append(HumanMessage(content=message, additional_kwargs={'token_length': token_length}))

    def add_ai_message(self, message: str, token_length: int) -> None:
        self.messages.append(AIMessage(content=message, additional_kwargs={'token_length': token_length}))

    def clear(self) -> None:
        self.messages = []
    
    def get_system_message(self) -> SystemMessage:
        if len(self.messages) == 0:
            return None
        else:
            return self.messages[0]
    
    def set_system_message(self, message: SystemMessage) -> None:
        if message is None:
            return
        if not message.additional_kwargs.get('token_length'):
            raise ValueError('SystemMessage must have token_length')
        if len(self.messages) == 0:
            self.messages.append(message)
        else:
            self.messages[0] = message

    def get_messages_within_max_length(self) -> List[BaseMessage]:
        # Always add system message
        messages = [self.messages[0]]
        current_length = self.messages[0].additional_kwargs['token_length']
        # Add messages until max length is reached, starting from the recent message
        reversed_order = []
        for message in reversed(self.messages[1:]):
            if current_length + message.additional_kwargs['token_length'] > self.max_length:
                print(f'Current length: {current_length}, max length: {self.max_length}, token length: {message.additional_kwargs["token_length"]}')
                break
            reversed_order.append(message)
            current_length += message.additional_kwargs['token_length']
        reversed_order.reverse()
        messages.extend(reversed_order)

        return messages

    def get_prompt_within_max_length(self) -> str:
        messages = self.get_messages_within_max_length()
        prompt = ''
        for message in messages:
            prompt += message.content+"\n"

        return prompt


def remove_mention(text):
    text = re.sub(r'<@(everyone|here|[!&]?[0-9]{17,20})>', '', text).strip()
    text = re.sub(r'@(everyone|here)', '', text).strip()
    return text

   
def convert_name_to_id(name_to_id, text):
    def repl(mention):
        name = mention.group(1)
        if name == 'everyone' or name == 'here':
            return f'@{name}'
        else:
            return f'<@{name_to_id[name]}>'

    text = re.sub(r'<@([a-zA-Z0-9\uac00-\ud7a3]*)>', repl, text, re.MULTILINE)

    return text


def convert_id_to_name(id_to_name, text):
    def repl(mention):
        id = mention.group(1)
        if id == 'everyone' or id == 'here':
            return f'@{id}'
        else:
            return f'<@{id_to_name[id]}>'

    text = re.sub(r'<@([!&]?[0-9]{17,20})>', repl, text, re.MULTILINE)

    return text

def add_author(name, text):
    if not text.startswith(name):
        return f"{name}: {text}"
    else:
        return text

def remove_author(name, text):
    text = text.strip()
    if text.startswith(name):
        return text[len(name) + 2:].strip()
    else:
        return text
    
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += message.content+"\n"

    return prompt

if __name__ == "__main__":
    test = "Of course, <@김연아>. ZeldaLink, <@everyone> to defeat Ganon, you must first understand the power of the Force within you. Trust in your training and let the Force guide you. Remember, fear is the path to the dark side. Face your fears and overcome them with the power of the Force. May the Force be with you, <@ZeldaLink>."
    test2 = "<@1097712066250416249>hello why not<@eveyone>"

    print(convert_name_to_id({"김연아": "123123", "ZeldaLink": "1097712066250416249"}, test))
    print(convert_id_to_name({"123123": "juny116", "1097712066250416249": "ZeldaLink"}, test2))

    temp = SystemMessage(content="test", additional_kwargs={'token_length':1})
    history = ChatMessageHistoryWithTokenLength(temp)
    print(history.messages)
    print(history.get_system_message())
    temp2 = SystemMessage(content="test2", additional_kwargs={'token_length':2})
    history.set_system_message(temp2)
    print(history.get_system_message())