"""multilingual-e5 系埋め込み器の検証。

**プレフィックス規約が最大の落とし穴。** 付け忘れても例外は出ず、
静かに検索精度だけが落ちる。設定に露出させず実装側で決めているので、
その判定ロジックをテストで固定する（docs/design/design.md §3.2(3)）。

モデルのダウンロードが要る部分は分離し、規約の判定はモデル無しで検証する。
"""

from __future__ import annotations

import pytest

from ragforlocalllm.core.registry import build
from ragforlocalllm.stages.embedder.sentence_transformers import (
    SentenceTransformerEmbedder,
    detect_prefix_style,
)


def test_registered_under_expected_name() -> None:
    embedder = build("embedder", {"type": "sentence_transformers"})
    assert isinstance(embedder, SentenceTransformerEmbedder)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("intfloat/multilingual-e5-base", "e5"),
        ("intfloat/multilingual-e5-large", "e5"),
        ("intfloat/e5-small-v2", "e5"),
        ("BAAI/bge-m3", "bge"),
        ("BAAI/bge-large-ja-v1.5", "bge"),
        ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "none"),
        ("cl-nagoya/ruri-base", "none"),
    ],
)
def test_prefix_style_detection(model: str, expected: str) -> None:
    assert detect_prefix_style(model) == expected


def test_unknown_model_gets_no_prefix() -> None:
    """**推測で誤った指示文を付けるより、付けないほうが被害が小さい。**

    誤ったプレフィックスは埋め込み空間をずらすが、無しなら素の性能が出る。
    """
    assert detect_prefix_style("some-org/mystery-model") == "none"


def test_model_name_containing_e5_as_substring_is_not_matched() -> None:
    """「e5」を含むだけの語で誤検出しない（境界を見ている）。"""
    assert detect_prefix_style("org/base5-model") == "none"
    assert detect_prefix_style("org/rose5") == "none"


# ----------------------------------------------------------------------
# プレフィックスの付与
# ----------------------------------------------------------------------


def test_e5_uses_query_and_passage_prefixes() -> None:
    embedder = SentenceTransformerEmbedder(model="intfloat/multilingual-e5-base")
    assert embedder.prefix_style == "e5"
    assert embedder._with_prefix("承認者は誰か", is_query=True) == "query: 承認者は誰か"
    assert embedder._with_prefix("経営者が承認する", is_query=False) == "passage: 経営者が承認する"


def test_bge_prefixes_only_the_query() -> None:
    embedder = SentenceTransformerEmbedder(model="BAAI/bge-m3")
    assert embedder.prefix_style == "bge"
    assert embedder._with_prefix("本文", is_query=False) == "本文"
    assert embedder._with_prefix("質問", is_query=True).endswith("質問")
    assert embedder._with_prefix("質問", is_query=True) != "質問"


def test_prefix_style_can_be_forced() -> None:
    """自動判定できないモデルでも規約を明示できる。"""
    embedder = SentenceTransformerEmbedder(model="org/custom", prefix_style="e5")
    assert embedder._with_prefix("x", is_query=True) == "query: x"


def test_none_style_leaves_text_untouched() -> None:
    embedder = SentenceTransformerEmbedder(model="org/custom")
    assert embedder._with_prefix("x", is_query=True) == "x"
    assert embedder._with_prefix("x", is_query=False) == "x"


def test_invalid_prefix_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="prefix_style"):
        SentenceTransformerEmbedder(prefix_style="colbert")  # type: ignore[arg-type]


def test_describe_records_the_prefix_style() -> None:
    """同じモデル名でも規約が違えば別の実験。ランレコードに残す。"""
    info = SentenceTransformerEmbedder(model="intfloat/multilingual-e5-base").describe()
    assert info["prefix_style"] == "e5"
    assert info["model"] == "intfloat/multilingual-e5-base"


def test_model_is_not_loaded_until_used() -> None:
    """構築だけでモデルを読み込まない（フットプリント計測が成立しなくなる）。"""
    embedder = SentenceTransformerEmbedder(model="intfloat/multilingual-e5-base")
    assert embedder._model is None


# ----------------------------------------------------------------------
# 実モデルを使う検証
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_query_and_passage_embeddings_differ_for_e5() -> None:
    """プレフィックスが実際に効いていること（同じ文でもベクトルが変わる）。"""
    pytest.importorskip("sentence_transformers")
    import numpy as np

    embedder = SentenceTransformerEmbedder(
        model="intfloat/multilingual-e5-base", device="cpu", batch_size=2
    )
    text = ["情報セキュリティ基本方針"]
    q = embedder.embed_queries(text)
    p = embedder.embed_passages(text)

    assert q.shape == p.shape == (1, embedder.dim)
    assert not np.allclose(q, p)
    # 正規化されていること（内積 = コサイン類似度が成立する前提）
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
