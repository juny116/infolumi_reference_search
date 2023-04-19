import re


def remove_mention(text):
    text = re.sub(r'<@(everyone|here|[!&]?[0-9]{17,20})>', '', text).strip()
    text = re.sub(r'@(everyone|here)', '', text).strip()
    return text

def convert_name_to_id(name_to_id, text):
    mentions = re.findall(r'<@[a-zA-Z0-9]+>', text, re.MULTILINE)
    for mention in mentions:
        text = text.replace(mention, f"<@{name_to_id[mention.replace('<@', '').replace('>', '').replace('&','')]}>")

    return text

def convert_id_to_name(id_to_name, text):
    mentions = re.findall(r'<@([!&]?[0-9]{17,20})>', text, re.MULTILINE)
    for mention in mentions:
        text = text.replace(mention, f"<@{id_to_name[mention.replace('<@', '').replace('>', '').replace('&','')]}>")
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
    
def chat_history_to_prompt(histroy):
    prompt = ""
    for message in histroy.messages:
        prompt += message.content + "\n"

    return prompt
