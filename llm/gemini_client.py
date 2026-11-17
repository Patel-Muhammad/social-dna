"""Thin wrapper around the current Google Gen AI SDK (`google-genai`)."""
from __future__ import annotations

import time

from core import config

# Tried in order if the primary model is unavailable (404 retired / persistent 5xx).
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash",
                   "gemini-3.6-flash", "gemini-3.5-flash"]
SETUP_HELP = ("Set GEMINI_API_KEY in your .env file (copy .env.example to .env). "
              "Create a key at https://aistudio.google.com/apikey and restart the app.")


class GeminiUnavailableError(RuntimeError):
    """Gemini cannot be used right now; message says why and how to fix it."""


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None,
                 timeout_ms: int = 20000, retries: int = 1, backoff_s: float = 1.5,
                 deadline_s: float = 45.0, cooldown_s: float = 90.0):
        self.api_key = api_key if api_key is not None else config.get_gemini_api_key()
        self.model = model or config.GEMINI_MODEL
        self.models = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        self.timeout_ms, self.retries, self.backoff_s = timeout_ms, retries, backoff_s
        self.deadline_s, self.cooldown_s = deadline_s, cooldown_s
        self._down_until = 0.0
        self._client = client
        self.last_model_used: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) or self._client is not None

    def _sdk(self):
        if self._client is None:
            if not self.api_key:
                raise GeminiUnavailableError("No Gemini API key configured. " + SETUP_HELP)
            from google import genai
            self._client = genai.Client(api_key=self.api_key, http_options={"timeout": self.timeout_ms})
        return self._client

    @property
    def is_down(self) -> bool:
        """True during the cool-down after an outage (avoids making users wait on every question)."""
        return time.monotonic() < self._down_until

    @staticmethod
    def _status(exc: Exception) -> int | None:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        return code if isinstance(code, int) else None

    def _call(self, fn):
        """Run fn(model) with retries on transient errors and fallback on retired models."""
        if self.is_down:
            raise GeminiUnavailableError(
                "Gemini is unavailable right now (it failed a moment ago; retrying shortly). The dashboard still works.")
        last: Exception | None = None
        started = time.monotonic()
        for model in self.models:
            if time.monotonic() - started > self.deadline_s:
                break
            for attempt in range(self.retries + 1):
                try:
                    result = fn(model)
                    self.last_model_used = model
                    return result
                except GeminiUnavailableError:
                    raise
                except Exception as exc:  # SDK raises several error classes
                    last = exc
                    status = self._status(exc)
                    if status in (400, 401, 403):
                        raise GeminiUnavailableError(
                            f"Gemini rejected the request (HTTP {status}). Check that the API key is valid "
                            f"and enabled. {SETUP_HELP}") from exc
                    if status == 404:
                        break  # model retired for this key: try next model
                    if status in (429, 500, 503, 504) or status is None:
                        if attempt < self.retries:
                            time.sleep(self.backoff_s * (attempt + 1))
                            continue
                    break
        self._down_until = time.monotonic() + self.cooldown_s
        raise GeminiUnavailableError(
            f"Gemini is temporarily unavailable ({type(last).__name__}: {str(last)[:160]}). "
            "The dashboard still works; try again in a minute.") from last

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        from google.genai import types
        sdk = self._sdk()

        def run(model: str) -> str:
            cfg = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
            resp = sdk.models.generate_content(model=model, contents=prompt, config=cfg)
            if not (resp.text or "").strip():
                raise RuntimeError("Gemini returned an empty response")
            return resp.text.strip()

        return self._call(run)

    def choose_function(self, prompt: str, declarations: list[dict]) -> tuple[str, dict]:
        """Ask Gemini to pick one declared function (forced function calling). Returns (name, args)."""
        from google.genai import types
        sdk = self._sdk()
        tool = types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=d["name"], description=d["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                                for k, v in d.get("parameters", {}).items()},
                    required=[k for k, v in d.get("parameters", {}).items() if v.get("required")],
                ) if d.get("parameters") else None)
            for d in declarations])
        cfg = types.GenerateContentConfig(
            temperature=0.0, tools=[tool],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")))

        def run(model: str):
            resp = sdk.models.generate_content(model=model, contents=prompt, config=cfg)
            calls = resp.function_calls or []
            if not calls:
                raise RuntimeError("Gemini did not select a function")
            return calls[0].name, dict(calls[0].args or {})

        return self._call(run)
