from config.settings import settings

print("API key:   ", settings.openai_api_key[:15])
print("Model:     ", settings.openai_model)
print("Base URL:  ", settings.openai_base_url or "(OpenAI default)")
print("Is local:  ", bool(settings.openai_base_url))