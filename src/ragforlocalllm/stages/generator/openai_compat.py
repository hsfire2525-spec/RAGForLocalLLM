"""OpenAI互換サーバ（LM Studio / Ollama / vLLM）向けジェネレータ。

``/v1/chat/completions`` を使い、**プロンプト文字列を自前で組まない**。
チャットテンプレートの適用はサーバ側に任せる（モデルごとに異なり、
特に Gemma 系は system ロールを持たないため）。

依存を増やさないため標準ライブラリの urllib を使う。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import Answer, Prompt, TokenUsage


class GeneratorError(RuntimeError):
    """LLMサーバとの通信に失敗した。"""


@register("generator", "openai_compat")
class OpenAICompatGenerator:
    """LM Studio 等の OpenAI 互換 chat completions を呼ぶ。

    決定性のため ``temperature=0`` と ``seed`` 固定を既定とする。
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int = 512,
        top_p: float | None = None,
        timeout: float = 300.0,
        api_key: str = "not-needed",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.api_key = api_key
        self.extra_body = extra_body or {}

    # ------------------------------------------------------------------

    def generate(self, prompt: Prompt, schema: dict[str, Any] | None = None) -> Answer:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in prompt.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if schema is not None:
            # LM Studio は JSON Schema による構造化出力に対応する。
            # GBNF 文法の直接指定は API 経由では利用できない。
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        payload.update(self.extra_body)

        started = time.perf_counter()
        data = self._post("/chat/completions", payload)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise GeneratorError(f"予期しない応答形式です: {json.dumps(data)[:400]}") from exc

        schema_valid: bool | None = None
        if schema is not None:
            try:
                json.loads(text)
                schema_valid = True
            except json.JSONDecodeError:
                schema_valid = False

        usage = data.get("usage") or {}
        return Answer(
            text=text.strip(),
            raw_text=text,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason"),
            schema_valid=schema_valid,
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ),
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """ランレコードに記録するモデル情報。

        サーバ側のロード設定（量子化・コンテキスト長）は API から
        取得できる範囲で残す。記録されない実験結果は再現不能になる。
        """
        info: dict[str, Any] = {
            "type": "openai_compat",
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        try:
            info["server_models"] = self.list_models()
            info["model_loaded"] = self.model in info["server_models"]
        except GeneratorError as exc:
            info["server_error"] = str(exc)
        return info

    def list_models(self) -> list[str]:
        data = self._get("/models")
        return [m["id"] for m in data.get("data", []) if "id" in m]

    def health_check(self) -> None:
        """モデルがサーバにあるか事前確認する。無ければ理由を明示して落とす。"""
        models = self.list_models()
        if self.model not in models:
            raise GeneratorError(
                f"モデル {self.model!r} がサーバにありません。\n"
                f"  利用可能: {', '.join(models) or '(なし)'}\n"
                f"  LM Studio 側でモデルをロードするか、設定の model を変更してください。"
            )

    # ------------------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        return self._request(urllib.request.Request(self.base_url + path, headers=self._headers()))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=body, headers=self._headers(), method="POST"
        )
        return self._request(request)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _request(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise GeneratorError(f"HTTP {exc.code} {self.base_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise GeneratorError(
                f"{self.base_url} に接続できません: {exc.reason}\n"
                "  LM Studio のローカルサーバが起動しているか確認してください。"
            ) from exc
        except TimeoutError as exc:
            raise GeneratorError(
                f"{self.base_url} がタイムアウトしました（{self.timeout}s）"
            ) from exc
