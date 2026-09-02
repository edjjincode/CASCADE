"""
Model client — one call surface for every stage of the pipeline.

CASCADE talks to the model in exactly one way: a text prompt plus zero or
more images, answered as text. Grounding stage 1 sends no image, stage 2
sends one, stage 3 sends K, and verification sends a single scaffolded
image. Nothing else is needed, so nothing else is here.

Any OpenAI-compatible endpoint works; set CASCADE_BASE_URL and
CASCADE_MODEL. The prompts themselves live as plain text in prompts/ and
are loaded by `load_prompt`, so what the model receives can be read
without running anything.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

from . import config


# A `# ==== NAME ====` line opens a named section of a prompt file.
_SECTION = re.compile(r"^#\s*=+\s*([A-Z][A-Z ]*[A-Z]|[A-Z])\s*=+\s*$")


def load_prompt(name: str, section: str | None = None) -> str:
    """Read a prompt from prompts/ by stem, e.g. load_prompt("stage2_locate").

    Each file opens with a `#` header describing the call, and may split
    the rest into named sections — SYSTEM and USER halves of a call, or a
    RENDERED EXAMPLE showing what one filled-in prompt looked like. The
    header and the example are documentation for the reader; only the
    prompt itself is sent.

    By default this returns the file's first section, which is always the
    prompt. Pass `section` to read another one by name.
    """
    lines = (config.PROMPTS / f"{name}.txt").read_text().splitlines()

    sections: dict[str, list[str]] = {}
    order:    list[str] = []
    current   = sections.setdefault("", [])
    order.append("")
    in_header = True

    for line in lines:
        banner = _SECTION.match(line)
        if banner:
            key = banner.group(1).strip()
            current = sections.setdefault(key, [])
            if key not in order:
                order.append(key)
            in_header = False
        elif in_header and line.startswith("#"):
            continue                      # the file's own documentation
        else:
            in_header = False
            current.append(line)

    if section is not None:
        return "\n".join(sections[section.upper()]).strip()

    for key in order:                     # the first section holding anything
        text = "\n".join(sections[key]).strip()
        if text:
            return text
    return ""


def _encode(path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class Client:
    """A thin, retrying wrapper over a chat-completions endpoint."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, max_tokens: int | None = None):
        from openai import OpenAI          # imported late so `--help` works keyless

        self.model      = model or config.MODEL
        self.max_tokens = max_tokens or config.MAX_TOKENS
        key             = api_key or config.API_KEY
        if not key:
            raise RuntimeError(
                "No API key. Set OPENROUTER_API_KEY (or OPENAI_API_KEY); "
                "see .env.example. Commands that only replay cached results "
                "do not need one."
            )
        self._client = OpenAI(base_url=base_url or config.BASE_URL, api_key=key)

    # ── the single call surface ──────────────────────────────────────
    def ask(self, prompt: str, images: list | None = None,
            retries: int = 2, timeout: int = 120) -> str:
        """Send `prompt` with `images` (paths, in order) and return the text.

        Images follow the prompt in the order given, which is how the
        stage-3 and verification prompts refer to them.
        """
        content = [{"type": "text", "text": prompt}]
        for path in images or []:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode(path)}"},
            })

        params = {
            "model":      self.model,
            "messages":   [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "timeout":    timeout,
        }
        if config.REASONING and _takes_reasoning(self.model):
            params["reasoning_effort"] = config.REASONING

        last = None
        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(**params)
                return response.choices[0].message.content or ""
            except Exception as exc:           # network, rate limit, refusal
                last = exc
                if attempt < retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"model call failed after {retries + 1} attempts: {last}")

    def ask_json(self, prompt: str, images: list | None = None, **kw) -> dict:
        """`ask`, with the reply parsed as JSON."""
        return parse_json(self.ask(prompt, images, **kw))


def _takes_reasoning(model: str) -> bool:
    return any(tag in model.lower() for tag in ("gpt-5", "o1", "o3"))


def parse_json(text: str) -> dict:
    """Pull a JSON object out of a reply, tolerating prose and code fences."""
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-z]*\n|\n```$", "", body).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", body)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object in reply: {text[:300]!r}")
