# llm-discord-bot

## How to run
1. Setup OpenAI API [API Reference](https://platform.openai.com/docs/api-reference)
2. Create Discord bot [Discords Bot Documentation](https://discord.com/developers/docs/intro)
    * Add your discord bot token to config/discord yaml (Check sample.yaml)
3. (If using longterm memory) Setup elasticsearch. It is easy to setup via [elasticsearch docker](https://hub.docker.com/_/elasticsearch)
4. Run ```pip install -r requirements.txt```
5. Run ```python llmbot.py```

## About config
* 