from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    messages_from_dict,
    messages_to_dict,
)
from langchain.memory import ChatMessageHistory

import discord
import requests
from discord.utils import escape_mentions, remove_markdown
import json
from models import load_model
from utils import remove_mention, convert_id_to_name, convert_name_to_id, add_author, remove_author
import hydra
from omegaconf import DictConfig


class LLMBot(discord.Client):
    def __init__(self, config, intents):
        super().__init__(intents=intents)
        self.config = config
        self.history = ChatMessageHistory()
        self.model = load_model(self.config['model'])

    def create_user_dict(self):
        self.name_to_id = {str(member.name): str(member.id) for member in self.get_all_members() if member.name != 'UtilBot'}
        self.id_to_name = {str(member.id): str(member.name) for member in self.get_all_members() if member.name != 'UtilBot'}
        self.member_list = [member.name for member in self.get_all_members() if member.name != 'UtilBot']

    def clear_history(self):
        self.current_turn = 0
        self.history.clear()

    def renew_system(self, add_prompt=''):
        system_message_prompt = SystemMessagePromptTemplate.from_template(self.config['template']['system'])
        system_message = system_message_prompt.format(member_list=', '.join(self.member_list), current_member=self.user.name, additional_prompt=add_prompt)
        self.clear_history()
        self.history.messages.append(system_message)

    def save_history(self, content):
        save_path = self.config['save_path']
        try:
            fname = content.split(' ')[1]
            fname = f'{save_path}/{self.user.name}{fname}.json'
        except:
            fname = f'{save_path}/{self.user.name}.json'
        with open(fname, 'w') as f:
            json.dump(messages_to_dict(self.history.messages), f)
    
        return fname

    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        self.max_turns = 30
        self.create_user_dict()
        self.renew_system()
        await self.change_presence(status=discord.Status.online, activity=discord.Game("대기중"))
 
    async def on_message(self, message):
        # Do not respond to ourselves
        if message.author == self.user:
            return
        # Do not respond to other system messages
        if message.content.startswith('***'):
            return
        # In case where command is sent
        elif message.content.startswith('!'):
            # Only respond to the command if it is sent to the bot by mentioning
            if self.user.mentioned_in(message):
                # remove mention from the message to get the command
                content = remove_mention(message.content).strip()
 
                if content.startswith('!clear'):
                    self.clear_history()
                    await message.channel.send('*** History Cleared ***')

                if content.startswith('!prompt'):
                    add_prompt = ' '.join(content.split(' ')[1:])
                    self.renew_system(add_prompt=add_prompt)
                    await message.channel.send(f'*** History Cleared And Prompt `{add_prompt}` Added ***')

                elif content.startswith('!save'):
                    fname = self.save_history(content)
                    await message.channel.send(f"*** Saved to {fname} ***")

                elif content.startswith('!max'):
                    max_turns = content.split(' ')[-1]
                    self.max_turns = int(max_turns)
                    await message.channel.send(f"*** Max turn set to {self.max_turns} ***")
                
                elif content.startswith('!ping'):
                    await message.channel.send('*** pong {0.author.mention} ***'.format(message))
    
        # In case where message is sent
        else:
            # Include mention in the message
            content = convert_id_to_name(self.id_to_name, message.content)
            # If author is not specified, add author to the message
            content = add_author(message.author.name, content)
            # Only respond to the message if it is sent to the bot by mentioning
            if self.user.mentioned_in(message):
                # Check if the maximum number of turns has been reached
                if self.current_turn >= self.max_turns:
                    await message.channel.send("*** You have reached the maximum number of turns. Please start a new conversation. ***")
                else:
                    self.current_turn += 1
                    self.history.add_user_message(content)
                    response = self.model.generate(self.history, self.user.name)
                    # remove author from the message to send chat without author
                    response = remove_author(self.user.name, response)
                    # save the response to the history with author
                    self.history.add_ai_message(add_author(self.user.name, response))
                    response = convert_name_to_id(self.name_to_id, response)
                    await message.channel.send(response)
            else:
                self.history.add_user_message(content)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    intents = discord.Intents.all()
    client = LLMBot(config=config, intents=intents)
    client.run(config['discord']['token'])

if __name__ == '__main__':
    main()