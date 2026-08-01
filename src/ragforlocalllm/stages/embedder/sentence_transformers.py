"""sentence-transformers による埋め込み（multilingual-e5 系を主対象）。

**プレフィックス規約を実装側に閉じ込める。** multilingual-e5 は学習時に
クエリへ ``query: ``、文書へ ``passage: `` を付ける前提であり、これを
外すと検索精度が明確に落ちる。しかも**間違えても例外は出ず、静かに
精度だけが下がる**ため、設定ファイル側に露出させてはいけない
（docs/design/design.md §3.2(3)）。

モデルによって規約が違うため、モデル名から自動判定する。判定できない
モデルではプレフィックスを付けない（``prefix_style`` で明示もできる）。

E5 系はコサイン類似度前提なので、既定で L2 正規化する。正規化した
ベクトルの内積 = コサイン類似度であり、FAISS の ``IndexFlatIP`` と
numpy の全探索がそのまま一致する。
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np

from ragforlocalllm.core.registry import register

PrefixStyle = Literal["e5", "bge", "none", "auto"]

_E5_PATTERN = re.compile(r"(^|[/\-_])e5([\-_]|$)", re.IGNORECASE)
_BGE_PATTERN = re.compile(r"(^|[/\-_])bge([\-_]|$)", re.IGNORECASE)

# BGE 系はクエリ側にのみ指示文を付ける（文書側は素のまま）。
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def detect_prefix_style(model_name: str) -> PrefixStyle:
    """モデル名からプレフィックス規約を推定する。

    判定できない場合は ``none``。**推測で誤ったプレフィックスを付けるより、
    付けないほうが被害が小さい**（誤った指示文は埋め込み空間をずらす）。
    """
    if _E5_PATTERN.search(model_name):
        return "e5"
    if _BGE_PATTERN.search(model_name):
        return "bge"
    return "none"


@register("embedder", "sentence_transformers")
class SentenceTransformerEmbedder:
    """sentence-transformers のモデルで埋め込む。

    Parameters
    ----------
    model:
        HuggingFace のモデルID。
    prefix_style:
        ``auto`` ならモデル名から推定する（既定）。``e5`` / ``bge`` /
        ``none`` で明示もできる。
    device:
        ``None`` なら sentence-transformers に任せる（CUDA があれば CUDA）。
        環境1では埋め込みをCPUに寄せる方針のため ``cpu`` を明示できる
        （docs/design/design.md §6.6）。
    batch_size:
        大きいほど速いがピークメモリが増える。環境1の制約に効く。
    normalize:
        L2正規化。E5 系はコサイン前提なので既定で有効。
    """

    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-base",
        prefix_style: PrefixStyle = "auto",
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if prefix_style not in ("e5", "bge", "none", "auto"):
            raise ValueError("prefix_style は e5 / bge / none / auto のいずれかです")
        self.model_name = model
        self.prefix_style: PrefixStyle = (
            detect_prefix_style(model) if prefix_style == "auto" else prefix_style
        )
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_length = max_seq_length
        self.trust_remote_code = trust_remote_code
        self._model: Any | None = None

    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        """モデルは初回利用時に読み込む。

        **グローバルなシングルトンを持たない**（各コンポーネントが自分の
        モデルを所有する）。環境1でのメモリ配分を後から変えられるように
        するための制約（docs/design/design.md §10.5）。
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - 任意依存
                raise RuntimeError(
                    "sentence-transformers が必要です: uv sync --extra models"
                ) from exc
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
            if self.max_seq_length is not None:
                self._model.max_seq_length = self.max_seq_length
        return self._model

    @property
    def dim(self) -> int:
        model = self._ensure_model()
        # sentence-transformers 5.x で get_sentence_embedding_dimension から改名。
        # 両対応にして、バージョンを上げただけで壊れないようにする。
        getter = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        return int(getter())

    # ------------------------------------------------------------------

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self._with_prefix(t, is_query=True) for t in texts])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed([self._with_prefix(t, is_query=False) for t in texts])

    def _with_prefix(self, text: str, *, is_query: bool) -> str:
        if self.prefix_style == "e5":
            return f"{'query' if is_query else 'passage'}: {text}"
        if self.prefix_style == "bge":
            return f"{_BGE_QUERY_PREFIX}{text}" if is_query else text
        return text

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._ensure_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        """ランレコードに残す情報。

        **プレフィックス規約は必ず記録する。** 同じモデル名でも規約が
        違えば別の実験であり、後から数値を比較できなくなる。
        """
        return {
            "model": self.model_name,
            "prefix_style": self.prefix_style,
            "device": self.device or "auto",
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "max_seq_length": self.max_seq_length,
        }
