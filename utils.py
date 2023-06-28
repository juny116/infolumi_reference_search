import re


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
    test = "Of course, <@test1>. ZeldaLink, <@everyone> to defeat Ganon, you must first understand the power of the Force within you. Trust in your training and let the Force guide you. Remember, fear is the path to the dark side. Face your fears and overcome them with the power of the Force. May the Force be with you, <@ZeldaLink>."
    test2 = "<@1097712066250416249>hello why not<@eveyone>"

    print(convert_name_to_id({"test1": "123123", "ZeldaLink": "1097712066250416249"}, test))
    print(convert_id_to_name({"123123": "test1", "1097712066250416249": "ZeldaLink"}, test2))
