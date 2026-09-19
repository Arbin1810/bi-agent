import json
from typing import Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
from app.utils.logger import logger

class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"LLM call: {len(prompt)} chars")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def complete_json(self, prompt: str, system: str = "", temperature: float = 0.1) -> dict:
        raw = self.complete(prompt, system, temperature)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Strip markdown fences if present
            cleaned = raw.strip().strip("```json").strip("```").strip()
            return json.loads(cleaned)

llm = LLMClient()