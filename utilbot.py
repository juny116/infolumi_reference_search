import discord
import requests
from discord.utils import escape_mentions, remove_markdown
import json
from utils import remove_mention, convert_id_to_name, convert_name_to_id
import hydra
from omegaconf import DictConfig


class UtilBot(discord.Client):
    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
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
                    try:
                        limit = int(content.split(' ')[1])
                    except:
                        limit = 500
                    await message.channel.send(f'*** Clearing messages... Limit {limit} ***')
                    deleted = await message.channel.purge(limit=limit)
                    await message.channel.send(f'*** Deleted {len(deleted)} message(s) ***')

                elif content.startswith('!ping'):
                    await message.channel.send('*** Pong {0.author.mention} ***'.format(message))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    intents = discord.Intents.all()
    client = UtilBot(intents=intents)
    client.run(config['discord']['token'])

if __name__ == '__main__':
    main()