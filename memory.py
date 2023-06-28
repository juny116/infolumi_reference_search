from elasticsearch import Elasticsearch
from typing import List

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


class ChatMessageHistoryWithContextWindow(BaseChatMessageHistory):
    """
    Chat message history with token length
    Keeps track of the token length of each message
    If the token length exceeds the max length, the messages within the max length will be returned
    """

    messages: List[BaseMessage] = []
    window_size: int = 1024

    def add_user_message(self, message: str, token_length: int) -> None:
        self.messages.append(
            HumanMessage(
                content=message, additional_kwargs={"token_length": token_length}
            )
        )

    def add_ai_message(self, message: str, token_length: int) -> None:
        self.messages.append(
            AIMessage(content=message, additional_kwargs={"token_length": token_length})
        )

    def clear(self) -> None:
        self.messages = []

    def set_system_message(self, message: SystemMessage) -> None:
        if message is None:
            return
        if not message.additional_kwargs.get("token_length"):
            raise ValueError("SystemMessage must have token_length")
        if len(self.messages) == 0:
            self.messages.append(message)
        else:
            self.messages[0] = message

    def get_system_message(self) -> SystemMessage:
        if len(self.messages) == 0:
            return None
        else:
            return self.messages[0]

    def get_messages_within_window_size(self) -> List[BaseMessage]:
        # Always add system message
        messages = [self.messages[0]]
        current_length = self.messages[0].additional_kwargs["token_length"]
        # Add messages until max length is reached, starting from the recent message
        reversed_order = []
        for message in reversed(self.messages[1:]):
            if (
                current_length + message.additional_kwargs["token_length"]
                > self.window_size
            ):
                # print(f'Current length: {current_length}, max length: {self.window_size}, token length: {message.additional_kwargs["token_length"]}')
                break
            reversed_order.append(message)
            current_length += message.additional_kwargs["token_length"]
        reversed_order.reverse()
        messages.extend(reversed_order)

        return messages

    def _messages_to_prompt(self, messages: BaseMessage) -> str:
        prompt = ""
        for message in messages:
            prompt += message.content + "\n"
        return prompt

    def get_prompt_within_window_size(self) -> str:
        messages = self.get_messages_within_window_size()
        return self._messages_to_prompt(messages)


class ChatMessageHistoryWithLongTerm(ChatMessageHistoryWithContextWindow):
    """
    Chat message history with long term storage
    Keep long term storage of the messages with Elasticsearch
    Always search for relevant messages first, if exists, append to the messages
    """

    messages: List[BaseMessage] = []
    host: str = "localhost:9200"
    window_size: int = 1024
    memory_size: int = 512
    index: str = None
    key: int = 0
    current_length: int = 0

    def __init__(self, host, index) -> None:
        super().__init__()
        self.host = host
        self.index = index.lower()
        self.es = Elasticsearch(self.host)
        self.create_index()

    def create_index(self) -> None:
        if self.es.indices.exists(index=self.index):
            pass
        else:
            self.es.indices.create(index=self.index)

    def delete_index(self) -> None:
        if self.es.indices.exists(index=self.index):
            self.es.indices.delete(index=self.index)
        else:
            pass

    def add_user_message(self, message: str, token_length: int) -> None:
        if self.current_length + token_length > self.memory_size:
            self._save_long_term()
        self.messages.append(
            HumanMessage(
                content=message,
                additional_kwargs={"token_length": token_length, "saved": None},
            )
        )
        self.current_length += token_length

    def add_ai_message(self, message: str, token_length: int) -> None:
        if self.current_length + token_length > self.memory_size:
            self._save_long_term()
        self.messages.append(
            AIMessage(
                content=message,
                additional_kwargs={
                    "token_length": token_length,
                    "saved": None,
                },
            )
        )
        self.current_length += token_length

    def get_messages_with_long_term(self) -> List[BaseMessage]:
        # Always add system message
        messages_within_window_size = self.get_messages_within_window_size()
        messages = [messages_within_window_size[0]]
        long_term = self._get_long_term(self.messages[-1].content)
        if long_term:
            messages.extend(long_term)

        messages.extend(messages_within_window_size[1:])

        return messages

    def get_prompt_with_long_term(self) -> str:
        messages = self.get_messages_with_long_term()
        return self._messages_to_prompt(messages)

    def _save_long_term(self) -> None:
        print("Saving long term")
        reversed_order = []
        for message in reversed(self.messages[1:]):
            if message.additional_kwargs["saved"] is not None:
                break
            message.additional_kwargs["saved"] = {"index": self.index, "key": self.key}
            reversed_order.append(message)
        reversed_order.reverse()

        prompt = ""
        for message in reversed_order:
            if type(message) == HumanMessage:
                prompt += f"<USER>|" + message.content + "<EOS>"
            elif type(message) == AIMessage:
                prompt += f"<AI>|" + message.content + "<EOS>"

        self._insertData(self.index, self.key, prompt)
        self.current_length = 0
        self.key += 1

    def _get_long_term(self, query) -> str:
        prompt = self._getData(self.index, query)
        # prompt to message
        messages = []
        if prompt is not None:
            for line in prompt.split("<EOS>")[:-1]:
                print(line)
                message_type, content = line.split("|", 2)
                if message_type == "<USER>":
                    messages.append(HumanMessage(content=content))
                elif message_type == "<AI>":
                    messages.append(AIMessage(content=content))

        return messages

    def _insertData(self, index, key, log):
        doc = {"log": log}
        self.es.index(index=index, doc_type="_doc", body=doc, id=key)

    def _getData(self, index, query):
        req_body = {
            "query": {
                "multi_match": {
                    "query": query,
                }
            }
        }
        res = self.es.search(
            search_type="dfs_query_then_fetch", index=index, body=req_body, size=5
        )
        if len(res["hits"]["hits"]) == 0:
            return None
        else:
            return res["hits"]["hits"][0]["_source"]["log"]


if __name__ == "__main__":
    from time import sleep

    es = Elasticsearch("localhost:9200")
    index = "masteryoda"
    es.indices.delete(index=index, ignore=[400, 404])
    if es.indices.exists(index=index):
        pass
    else:
        es.indices.create(index=index)
    query = "dinner menus"
    # req_body = {"query":{"match_all":{}}}
    req_body = {
        "query": {
            "multi_match": {
                "query": query,
            }
        }
    }
    doc = {"log": "I will have hamberger for dinner."}

    temp = es.index(index=index, doc_type="_doc", body=doc, id=0)
    sleep(1)

    res = es.search(
        search_type="dfs_query_then_fetch", index=index, body=req_body, size=5
    )
    print(res)
    print(res["hits"]["hits"][0])
    es.indices.delete(index=index, ignore=[400, 404])

    # temp = SystemMessage(content="test", additional_kwargs={'token_length':1})
    # history = ChatMessageHistoryWithContextWindow(temp)
    # print(history.messages)
    # print(history.get_system_message())
    # temp2 = SystemMessage(content="test2", additional_kwargs={'token_length':2})
    # history.set_system_message(temp2)
    # print(history.get_system_message())
