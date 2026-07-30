"""PDFローダーの検証。

評価用コーパス（IPAガイドライン）はリポジトリにコミットできないため、
テストは (a) 実測した現象を再現する合成入力による単体検証と、
(b) その場で生成した小さなPDFによる結合検証の2本立てとする。
コメント中の座標・フォントサイズは実コーパスからの実測値。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragforlocalllm.core.registry import build
from ragforlocalllm.stages.loader.pdf import (
    PyMuPDFLoader,
    _Fragment,
    _is_horizontal,
    _Row,
)


def frag(text: str, y0: float, x0: float, size: float = 10.6, bold: bool = False) -> _Fragment:
    return _Fragment(text=text, y0=y0, x0=x0, x1=x0 + len(text) * size * 0.9, size=size, bold=bold)


def row(text: str, size: float, *, bold: bool = True, page: int = 1) -> _Row:
    return _Row(text=text, page=page, y0=0.0, size=size, bold=bold, in_margin=False)


# ----------------------------------------------------------------------
# 行の再構成
# ----------------------------------------------------------------------


def test_registered_under_expected_names() -> None:
    assert isinstance(build("loader", {"type": "pymupdf"}), PyMuPDFLoader)


def test_fragments_on_same_visual_row_are_joined() -> None:
    """1つの視覚的な行が複数の line に割れる。連結しないと文が分断される。

    実コーパス p16: 「の確保に向けて、経営者は、」(x0=56.7) と
    「（1）に示す「３原則」について認識したうえで、」(x0=178.5) が
    同じ y=112.4 にある。
    """
    loader = PyMuPDFLoader()
    fragments = [
        _Fragment("の確保に向けて、経営者は、", y0=112.4, x0=56.7, x1=178.5, size=10.6, bold=False),
        _Fragment("（1）に示す「３原則」", y0=112.4, x0=178.5, x1=392.2, size=10.6, bold=False),
    ]
    rows = loader._group_rows(fragments, page_no=1)
    assert [r.text for r in rows] == ["の確保に向けて、経営者は、（1）に示す「３原則」"]


def test_nearby_rows_are_grouped_across_bin_boundary() -> None:
    """章番号と章題は y が 1.6pt ずれている。ビン分割では別行になってしまう。

    実コーパス p12: 「1」(18.8pt, y=61.5) と
    「情報セキュリティ対策を怠ることで企業が被る不利益」(15.4pt, y=63.1)。
    """
    loader = PyMuPDFLoader()
    fragments = [
        _Fragment("1", y0=61.5, x0=65.0, x1=76.0, size=18.8, bold=True),
        _Fragment("情報セキュリティ対策", y0=63.1, x0=91.4, x1=250.0, size=15.4, bold=True),
    ]
    rows = loader._group_rows(fragments, page_no=1)
    assert len(rows) == 1
    # 離れた断片の間には空白を入れる（「1情報…」とくっつけない）
    assert rows[0].text == "1 情報セキュリティ対策"
    # 代表サイズは最も文字数の多い断片から取る（番号の 18.8 に引きずられない）
    assert rows[0].size == pytest.approx(15.4)


def test_overlapping_duplicate_draws_are_collapsed() -> None:
    """表紙の袋文字は同じ文字列を僅かにずらして重ね描きしている。"""
    loader = PyMuPDFLoader()
    fragments = [
        _Fragment("中小企業の", y0=100.0, x0=50.0, x1=150.0, size=20.0, bold=True),
        _Fragment("中小企業の", y0=100.4, x0=51.2, x1=151.2, size=20.0, bold=True),
    ]
    rows = loader._group_rows(fragments, page_no=1)
    assert [r.text for r in rows] == ["中小企業の"]


def test_replacements_fix_broken_glyph_mapping() -> None:
    """ToUnicode の誤りで箇条書き記号が「●Ө」として抽出される。"""
    loader = PyMuPDFLoader(replacements={"●Ө": "・"})
    rows = loader._group_rows([frag("●Өログ管理", y0=10.0, x0=50.0)], page_no=1)
    assert rows[0].text == "・ログ管理"


# ----------------------------------------------------------------------
# 見出しの判定
# ----------------------------------------------------------------------


def test_bold_paragraph_is_not_treated_as_headings() -> None:
    """扉ページのリード文は太字14.2ptで、形だけ見ると見出しに合致する。

    見出しとして扱うと、配下の実節がすべて誤った階層にぶら下がる。
    """
    loader = PyMuPDFLoader(heading_run_limit=3)
    rows = [
        row("第1 部 経営者編", 29.8),
        row("経営者編では、情報セキュリティ対策に関し", 14.2),
        row("経営者が認識し", 14.2),
        row("自らの責任で考えなければならない", 14.2),
        row("本文", 10.6, bold=False),
    ]
    flags = loader._heading_flags(rows)
    assert flags == [True, False, False, False, False]


def test_heading_shape_rejects_mid_sentence_and_long_lines() -> None:
    loader = PyMuPDFLoader(heading_max_chars=40)
    assert loader._is_heading_shape(row("（1）金銭の損失", 12.4))
    # 文末が句点・読点の行は本文の折り返し
    assert not loader._is_heading_shape(row("事項について説明します。", 14.2))
    assert not loader._is_heading_shape(row("経営者が認識し、", 14.2))
    # 長すぎる行は見出しではない
    assert not loader._is_heading_shape(row("あ" * 41, 14.2))
    # 太字でなければ見出しではない
    assert not loader._is_heading_shape(row("（1）金銭の損失", 12.4, bold=False))


def test_singleton_heading_size_still_gets_a_level() -> None:
    """章題はフォントサイズが文字数に応じて自動調整される。

    出現回数で足切りすると、1回しか現れないサイズの章題が本文に落ちる。
    """
    loader = PyMuPDFLoader()
    rows = [row("第1 部", 29.8), row("1 章題", 15.4), row("（1）節題", 12.4)]
    levels = loader._heading_levels(rows, [True, True, True])
    assert levels == {29.8: 1, 15.4: 2, 12.4: 3}


# ----------------------------------------------------------------------
# セクションの組み立て
# ----------------------------------------------------------------------


def sections_of(loader: PyMuPDFLoader, rows: list[_Row]) -> list[tuple[str, str]]:
    flags = loader._heading_flags(rows)
    docs = loader._emit_sections(
        Path("doc.pdf"), rows, flags, loader._heading_levels(rows, flags), {}
    )
    return [(str(d.metadata["section_path"]), d.text) for d in docs]


def test_section_path_reflects_heading_hierarchy() -> None:
    loader = PyMuPDFLoader()
    rows = [
        row("第1 部 経営者編", 29.8),
        row("1 経営者が負う責任", 15.4),
        row("（1）法的責任", 12.4),
        row("本文です。", 10.6, bold=False),
    ]
    paths = sections_of(loader, rows)
    assert paths[-1][0] == "第1 部 経営者編 > 1 経営者が負う責任 > （1）法的責任"
    # 見出しは本文の先頭に含める。含めないと検索対象から消える。
    assert paths[-1][1] == "（1）法的責任\n本文です。"


def test_heading_is_excluded_from_text_when_disabled() -> None:
    loader = PyMuPDFLoader(include_heading=False)
    rows = [row("（1）法的責任", 12.4), row("本文です。", 10.6, bold=False)]
    assert sections_of(loader, rows)[-1][1] == "本文です。"


def test_repeated_decorative_label_does_not_capture_later_sections() -> None:
    """「コラム」は 15.6pt で章題(15.4pt)より大きい。

    サイズだけで階層を決めると、コラム以降の節がすべてコラムの配下に
    ぶら下がってしまう。
    """
    loader = PyMuPDFLoader(label_min_pages=3)
    rows = [
        row("1 章題", 15.4, page=1),
        row("コラム", 15.6, page=1),
        row("コラム本文", 10.6, bold=False, page=1),
        row("コラム", 15.6, page=2),
        row("コラム", 15.6, page=3),
        row("（2）節題", 12.4, page=3),
        row("節の本文", 10.6, bold=False, page=3),
    ]
    paths = dict(sections_of(loader, rows))
    # 「（2）節題」はコラムの配下ではなく章題の直下に来る
    assert "1 章題 > （2）節題" in paths


def test_first_body_before_any_heading_is_kept() -> None:
    loader = PyMuPDFLoader()
    rows = [row("前置きの文です。", 10.6, bold=False), row("1 章題", 15.4)]
    paths = sections_of(loader, rows)
    assert paths[0] == ("(前書き)", "前置きの文です。")


# ----------------------------------------------------------------------
# 補助関数
# ----------------------------------------------------------------------


def test_is_horizontal_rejects_rotated_text() -> None:
    """縦書きサイドタブは1文字ずつ行として現れる。dir で落とす。"""
    assert _is_horizontal((1.0, 0.0))
    assert not _is_horizontal((-1.0, 0.0))  # 実コーパスに 112 行
    assert not _is_horizontal((0.0, 1.0))


def test_emit_must_be_valid() -> None:
    with pytest.raises(ValueError, match="emit"):
        PyMuPDFLoader(emit="chapters")


# ----------------------------------------------------------------------
# 結合検証（生成したPDFを使う）
# ----------------------------------------------------------------------


def make_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    for pno in range(3):
        page = doc.new_page(width=516, height=729)
        page.insert_text((30, 15), "Running Header", fontname="helv", fontsize=9)
        if pno == 0:
            # insert_text の y はベースライン。実コーパスの章番号は bbox 上端が
            # y=61.5（余白帯 51pt の外）なので、それに合わせて下げる。
            page.insert_text((60, 90), "1", fontname="hebo", fontsize=18)
            page.insert_text((92, 90), "Chapter One", fontname="hebo", fontsize=15)
        page.insert_text((56, 130), f"Body text on page {pno + 1}.", fontname="helv", fontsize=10)
        page.insert_text((250, 700), str(pno + 1), fontname="helv", fontsize=10)
    doc.save(path)
    doc.close()


def test_load_removes_header_footer_and_keeps_chapter_number(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf)
    docs = PyMuPDFLoader(header_footer_ratio=0.3).load(pdf)
    whole = "\n".join(d.text for d in docs)

    assert "Running Header" not in whole  # 全ページのヘッダは除去される
    assert "Body text on page 1." in whole
    # 余白帯のページ番号は落ちるが、本文領域の章番号は見出しに残る
    assert "1 Chapter One" in whole
    assert any(d.metadata["section_path"] == "1 Chapter One" for d in docs)


def test_load_emit_pages_gives_one_document_per_page(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    make_pdf(pdf)
    docs = PyMuPDFLoader(emit="pages").load(pdf)
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]
