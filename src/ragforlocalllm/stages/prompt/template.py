"""Jinja2 テンプレートによるプロンプト構築とコンテキスト予算の適用。

プロンプトはコードに埋め込まず、テンプレートファイルとして管理する。
このリポジトリで最も頻繁に変更される要素であり、差分が追跡できる形に
しておく必要がある（docs/design/design.md §3.2(8)）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.tokens import make_counter
from ragforlocalllm.core.types import ChatMessage, Prompt, QueryState, ScoredChunk


@register("prompt", "template")
class TemplatePromptBuilder:
    """テンプレート + トークン予算。

    ``context_token_budget`` が指定されている場合、予算に収まるまで
    コンテキストを削る。件数ではなくトークン予算で制御することで、
    チャンクサイズを変える実験と整合する。
    """

    def __init__(
        self,
        path: str,
        context_token_budget: int | None = None,
        overflow_policy: str = "drop_lowest",
        tokenizer: str | None = None,
        instruction_placement: str = "system",
        root: str = "prompts",
    ) -> None:
        if overflow_policy not in ("drop_lowest", "truncate_each"):
            raise ValueError("overflow_policy は drop_lowest か truncate_each です")
        if instruction_placement not in ("system", "user"):
            raise ValueError("instruction_placement は system か user です")

        self.path = path
        self.context_token_budget = context_token_budget
        self.overflow_policy = overflow_policy
        self.instruction_placement = instruction_placement
        self.counter = make_counter(tokenizer)

        template_path = Path(path)
        search_root = template_path.parent if template_path.parent != Path() else Path(root)
        self.env = Environment(
            loader=FileSystemLoader([str(search_root), root, "."]),
            autoescape=select_autoescape(enabled_extensions=(), default=False),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self.template = self.env.get_template(template_path.name)

    def build(self, state: QueryState) -> Prompt:
        contexts, dropped, truncated = self._fit_budget(state.contexts)
        rendered = self.template.render(
            question=state.original_query,
            queries=state.queries,
            contexts=[self._context_view(i, c) for i, c in enumerate(contexts, start=1)],
        )
        system_text, user_text = _split_sections(rendered)

        messages: list[ChatMessage] = []
        if system_text and self.instruction_placement == "system":
            messages.append(ChatMessage(role="system", content=system_text))
            messages.append(ChatMessage(role="user", content=user_text))
        elif system_text:
            # Gemma 系はチャットテンプレートに system ロールを持たないため、
            # 指示を user 側に畳み込む構成も実験軸として選べるようにする。
            messages.append(ChatMessage(role="user", content=f"{system_text}\n\n{user_text}"))
        else:
            messages.append(ChatMessage(role="user", content=user_text))

        total = sum(self.counter.count(m.content) for m in messages)
        return Prompt(
            messages=messages,
            context_chunk_ids=[c.chunk.chunk_id for c in contexts],
            token_estimate=total,
            token_estimate_method=self.counter.method,
            template=self.path,
            n_dropped=dropped,
            n_truncated=truncated,
        )

    # ------------------------------------------------------------------

    def _fit_budget(self, contexts: list[ScoredChunk]) -> tuple[list[ScoredChunk], int, int]:
        if self.context_token_budget is None:
            return list(contexts), 0, 0

        budget = self.context_token_budget
        kept: list[ScoredChunk] = []
        used = 0
        dropped = 0
        truncated = 0

        for item in contexts:
            cost = self.counter.count(item.chunk.text)
            if used + cost <= budget:
                kept.append(item)
                used += cost
                continue
            if self.overflow_policy == "truncate_each":
                remaining = budget - used
                if remaining <= 0:
                    dropped += 1
                    continue
                # 保守的に文字数へ換算して切る
                ratio = remaining / cost
                cut = max(int(len(item.chunk.text) * ratio), 1)
                chunk = item.chunk.model_copy(update={"text": item.chunk.text[:cut]})
                kept.append(
                    ScoredChunk(chunk=chunk, score=item.score, provenance=dict(item.provenance))
                )
                used = budget
                truncated += 1
            else:
                dropped += 1
        return kept, dropped, truncated

    @staticmethod
    def _context_view(number: int, item: ScoredChunk) -> dict[str, Any]:
        chunk = item.chunk
        return {
            "number": number,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "page": chunk.page,
            "section_path": chunk.section_path,
            "score": round(item.score, 4),
        }


SYSTEM_MARKER = "<<<SYSTEM>>>"
USER_MARKER = "<<<USER>>>"


def _split_sections(rendered: str) -> tuple[str, str]:
    """テンプレート出力を system / user に分割する。

    テンプレート側で ``<<<SYSTEM>>>`` / ``<<<USER>>>`` を書くと役割を
    分けられる。マーカーが無い場合は全体を user 扱いにする。
    """
    if SYSTEM_MARKER not in rendered:
        return "", rendered.strip()
    _, rest = rendered.split(SYSTEM_MARKER, 1)
    if USER_MARKER not in rest:
        return rest.strip(), ""
    system_text, user_text = rest.split(USER_MARKER, 1)
    return system_text.strip(), user_text.strip()
