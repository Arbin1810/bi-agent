import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
from app.utils.logger import logger


class LLMClient:
    def __init__(self):
        kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**kwargs)
        self.model = settings.openai_model
        self.is_local = bool(settings.openai_base_url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def complete(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"LLM call: {len(prompt)} chars (local={self.is_local})")

        # Local models (Ollama) sometimes ignore response_format, so fall back
        if self.is_local:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        else:
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
            cleaned = raw.strip()
            # Strip markdown fences
            if "```" in cleaned:
                parts = cleaned.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        cleaned = part
                        break
            # Extract first {...} block
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end + 1]
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {raw[:500]}")
                raise e


llm = LLMClient()