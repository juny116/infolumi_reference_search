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

import discord
import requests
from discord.utils import escape_mentions, remove_markdown
import json
from models import load_model
from utils import (
    remove_mention,
    convert_id_to_name,
    convert_name_to_id,
    add_author,
    remove_author,
)
import hydra
from omegaconf import DictConfig
from time import sleep, time
import asyncio
import docx
import json
import requests
import xmltodict
import tempfile


def GetMedlinePage(start, end):
    if start == end:
        return start

    start_with_zero = (len(end) - len(start)) * "0" + start
    same_cnt = 0
    for e, s in zip(end, start_with_zero):
        if e == s:
            same_cnt += 1
        else:
            break
    medline_page = f":{start}-{end[same_cnt:]}"
    return medline_page


class LLMBot(discord.Client):
    def __init__(self, config, intents):
        super().__init__(heartbeat_timeout=60, intents=intents)
        self.config = config
        self.model = load_model(self.config["model"])

    def create_user_dict(self):
        if self.config["discord"].get("channel_id"):
            members = list(
                self.get_channel(self.config["discord"]["channel_id"]).members
            )
        else:
            members = list(self.get_all_members())
        self.name_to_id = {
            str(member.name): str(member.id)
            for member in members
            if member.name != "UtilBot"
        }
        self.id_to_name = {
            str(member.id): str(member.name)
            for member in members
            if member.name != "UtilBot"
        }
        self.member_list = [
            member.name for member in members if member.name != "UtilBot"
        ]

    def clear_history(self):
        self.current_turn = 0
        self.history.clear()
        self.history.delete_index()
        # self.history.create_index()

    def renew_system(self, add_prompt=""):
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.config["template"]["system"]
        )
        system_message = system_message_prompt.format(
            member_list=", ".join(self.member_list),
            current_member=self.user.name,
            additional_prompt=add_prompt,
        )
        num_tokens = self.model.get_num_tokens(system_message.content)
        system_message.additional_kwargs["token_length"] = num_tokens
        self.history.set_system_message(system_message)

    def save_history(self, content):
        save_path = self.config["save_path"]
        try:
            fname = content.split(" ")[1]
            fname = f"{save_path}/{self.user.name}{fname}.json"
        except:
            fname = f"{save_path}/{self.user.name}.json"
        with open(fname, "w") as f:
            json.dump(messages_to_dict(self.history.messages), f)

        return fname

    async def on_ready(self):
        print("Logged on as {0}!".format(self.user))
        await self.change_presence(
            status=discord.Status.online, activity=discord.Game("대기중")
        )

    async def on_message(self, message):
        # Do not respond to ourselves
        if message.author == self.user:
            return
        # Do not respond to other system messages
        if message.content.startswith("***"):
            return
        # Only respond to the message if it is sent to the bot by mentioning
        if self.user.mentioned_in(message):
            # Check if the maximum number of turns has been reached
            await message.channel.send("waiting for file to download...")
            for x in message.attachments:
                print("attachment-->", x.url)
                headers = {
                    "User-Agent": "DiscordBot (https://github.com/Rapptz/discord.py 0.2) Python/3.9 aiohttp/2.3"
                }
                d_url = requests.get(x.url, headers=headers)
                file_name = x.url.split("/")[-1]
                # response = wget.download(x.url, file_name)

                # file_name = "KRCP-21-043_ref.docx"
                # print(file_name)
                with open(file_name, "wb") as f:
                    f.write(d_url.content)
                self.loop.create_task(self.test(file_name))
                break
            await message.channel.send("STARTED")

    async def test(self, fname):
        # check the total time
        start_time = time()
        channel = self.get_channel(self.config["discord"]["channel_id"])
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.config["template"]["system"]
        )
        parse_template = self.config["template"]["parse"]
        human_message_prompt = HumanMessagePromptTemplate.from_template(parse_template)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        ref_list = []
        ref_size = 5
        templ = docx.Document(fname)
        for x, paragraph in enumerate(templ.paragraphs):
            ref_list.append(paragraph.text)
        tasks = []
        # TODO: limit the max task per single run
        max_turn = int(len(ref_list) / ref_size) + 1
        for i in range(max_turn):
            sub_list = ref_list[i * ref_size : (i + 1) * ref_size]
            temp = chat_prompt.format_prompt(
                references="\n".join(sub_list)
            ).to_messages()
            tasks.append(self.parse_reference(temp, channel, i, max_turn))
            break
        results_list = await asyncio.gather(*tasks)
        dummy = {
            "authors": ["dummy"],
            "title": "dummy",
            "journal": "dummy",
            "year": 2023,
            "month": None,
            "day": None,
            "volume": None,
            "issue": None,
            "start_page": None,
            "end_page": None,
        }
        # try to json load the results if fail then add dummy
        for i, results in enumerate(results_list):
            try:
                results = json.loads(results)
            except:
                results = [dummy for i in range(ref_size)]

        results_list = [json.loads(results) for results in results_list]
        results_list = [item for sublist in results_list for item in sublist]
        uid_list = []
        for i, reference in enumerate(results_list):
            if reference == dummy:
                uid_list.append(None)
            else:
                uid_list.append(self.search_pubmed(reference))
                await asyncio.sleep(0.3)
        print("uid_list", uid_list)
        results = self.fetch_pubmed(uid_list)
        fp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        for r in results:
            print(r, file=fp)
        fname = fp.name
        fp.close()

        await channel.send(
            f"Done in {(time()-start_time):.2} seconds", file=discord.File(fname)
        )

    async def parse_reference(self, messages, channel, index, max_turn):
        results = await self.model.agenerate(messages)
        await channel.send(f"parsing references {index+1}/{max_turn} done")
        return results

    def search_pubmed(self, reference):
        params = {
            "method": "auto",
            "authors": reference["authors"][0],
            "title": reference["title"],
            "pdat": str(reference["year"]),
            "volume": reference["volume"],
            "journal": reference["journal"],
        }
        try:
            res = requests.get(
                "https://pubmed.ncbi.nlm.nih.gov/api/citmatch/?", params=params
            )
            result = res.json().get("result")
            uids = result.get("uids")
            if not uids:
                return None
            # return the first uid
            for uid in uids:
                for k, v in uid.items():
                    if k == "pubmed":
                        return v
        except:
            return None

        return None

    def fetch_pubmed(self, uid_list):
        # uids without None
        uid_string = ",".join([uid for uid in uid_list if uid is not None])
        params = {"db": "pubmed", "id": uid_string, "retmode": "xml"}
        res = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?", params=params
        )
        results = xmltodict.parse(res.text)

        uid_index = 0
        revised_refs = []
        for k, uid in enumerate(uid_list):
            max_authors = False
            revised = f"{k+1}. "
            if uid is None:
                revised_refs.append(revised + "Not Found")
                continue

            authors = results["PubmedArticleSet"]["PubmedArticle"][uid_index][
                "MedlineCitation"
            ]["Article"]["AuthorList"]["Author"]
            title = results["PubmedArticleSet"]["PubmedArticle"][uid_index][
                "MedlineCitation"
            ]["Article"]["ArticleTitle"]
            journal = results["PubmedArticleSet"]["PubmedArticle"][uid_index][
                "MedlineCitation"
            ]["Article"]["Journal"]
            page = results["PubmedArticleSet"]["PubmedArticle"][uid_index][
                "MedlineCitation"
            ]["Article"]["Pagination"]
            if len(authors) > 6:
                authors = authors[:3]
                max_authors = True

            if type(authors) == list:
                authors = [
                    f"{author['LastName']} {author['Initials']}" for author in authors
                ]
            elif type(authors) == dict:
                authors = [f"{authors['LastName']} {authors['Initials']}"]
            revised += ", ".join(authors)
            if max_authors:
                revised += ", et al"

            revised += f". {title} "
            revised += f"{journal['ISOAbbreviation']}. "
            revised += f"{journal['JournalIssue']['PubDate']['Year']}"
            if journal["JournalIssue"].get("Volume"):
                revised += f";{journal['JournalIssue']['Volume']}"
            if journal["JournalIssue"].get("Issue"):
                if journal["JournalIssue"].get("Volume") is None:
                    revised += f";({journal['JournalIssue']['Issue']})"
                else:
                    revised += f"({journal['JournalIssue']['Issue']})"

            if page.get("EndPage"):
                revised += GetMedlinePage(page.get("StartPage"), page.get("EndPage"))
            else:
                revised += f":{page.get('StartPage')}"
            revised += "."
            uid_index += 1
            revised_refs.append(revised)

        return revised_refs


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    intents = discord.Intents.all()
    client = LLMBot(config=config, intents=intents)
    client.run(config["discord"]["token"])


if __name__ == "__main__":
    main()
