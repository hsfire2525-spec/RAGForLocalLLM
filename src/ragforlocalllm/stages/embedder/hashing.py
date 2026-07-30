"""追加依存なしの決定的埋め込み（テスト・疎通確認用）。

文字n-gramのハッシングトリックによる疎ベクトルを密ベクトルに畳む。
モデルのダウンロードもGPUも不要で完全に決定的なため、CI と Phase 0 の
疎通確認に使う。**精度実験には使わない**（意味的な類似性は捉えない）。

実運用の埋め込み器は Phase 2 で追加する（multilingual-e5 系:
``query: `` / ``passage: `` プレフィックス規約を実装側に閉じ込める）。
"""

from __future__ import annotations

import hashlib

import numpy as np

from ragforlocalllm.core.registry import register


@register("embedder", "hashing")
class HashingEmbedder:
    """文字n-gramハッシングによる決定的埋め込み。"""

    def __init__(self, dim: int = 256, ngram: int = 2, normalize: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim は正の整数である必要があります")
        if ngram <= 0:
            raise ValueError("ngram は正の整数である必要があります")
        self._dim = dim
        self.ngram = ngram
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            cleaned = "".join(text.split())
            for i in range(max(len(cleaned) - self.ngram + 1, 0)):
                gram = cleaned[i : i + self.ngram]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "big") % self._dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, slot] += sign
        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            np.divide(out, norms, out=out, where=norms > 0)
        return out
