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
    def complete(self, prompt: str, system: str = "", temperature: float = 0.1,
                 max_tokens: int = 4096) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"LLM call: {len(prompt)} chars (local={self.is_local})")

        if self.is_local:
            # Ollama: bigger context + output. Do NOT pass response_format —
            # some Ollama builds reject it, and its token accounting differs
            # enough to truncate multi-object JSON. JSON strictness is
            # enforced by the prompt instead.
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "options": {
                        "num_ctx": 8192,
                        "num_predict": 4096,
                    }
                },
            )
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        return resp.choices[0].message.content

    def complete_json(self, prompt: str, system: str = "", temperature: float = 0.1) -> dict:
        raw = self.complete(prompt, system, temperature)
        parsed = self._try_parse(raw)
        if parsed is not None:
            return parsed

        logger.warning("First JSON parse failed, asking model to repair")
        repair_prompt = (
            "The following text was supposed to be valid JSON but is broken "
            "or truncated. Return ONLY a valid, complete JSON object. "
            "Do not include markdown fences or commentary.\n\n"
            f"BROKEN:\n{raw[:3000]}"
        )
        raw2 = self.complete(repair_prompt, temperature=0.0, max_tokens=4096)
        parsed2 = self._try_parse(raw2)
        if parsed2 is not None:
            return parsed2

        logger.error(f"JSON repair failed. Original: {raw[:500]}")
        return {"error": "unparseable_json", "raw_excerpt": raw[:500]}

    @staticmethod
    def _try_parse(raw: str):
        if not raw:
            return None
        s = raw.strip()

        if s.startswith("```"):
            s = s.split("```", 2)
            if len(s) >= 2:
                s = s[1]
                if s.startswith("json"):
                    s = s[4:]
        s = s.strip()

        start = s.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        end = -1
        for i in range(start, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            return None

        candidate = s[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            import re
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None


llm = LLMClient()