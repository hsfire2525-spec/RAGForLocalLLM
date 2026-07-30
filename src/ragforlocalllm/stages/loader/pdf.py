"""PDF抽出。

**抽出品質が精度の上限を直接決める**ため、この段は実験軸であると同時に
評価の前提でもある（docs/design/design.md §3.2(1)）。

対象コーパス（IPAガイドライン）で実測した具体的な問題と対処:

1. **縦書きサイドタブが1文字ずつ本文に混入する** — ページ端の縦書き
   インデックスが行として現れる。``line["dir"]`` が水平 ``(1, 0)``
   以外になるため、これで除去できる（実測 1,723 文字）
2. **1つの視覚的な行が複数の line に分割される** — 「の確保に向けて、
   経営者は、」「（1）に示す「３原則」について認識したうえで、」のように
   途中で分かれる。y座標でまとめ直さないと文が分断され、gold引用が
   解決できなくなる
3. **読み順が保証されない** — ブロック列の順序は信頼できず、座標で
   ソートし直す必要がある
4. **ヘッダ・フッタ・ページ番号が混入する** — ただし章番号（「1」「2」）も
   単独の行として現れるため、数字だけを見て落とすと見出しが壊れる。
   ページ上下の余白帯に限って除去する
5. 見出しは太字フォントと文字サイズで判別できるが、**太字の本文**
   （扉ページのリード文など）が誤検出される。長さと文末表現で弾く
6. 図表中のラベルが極小フォント（3.4〜4.4pt）で本文行の間に割り込む。
   ``min_font_size`` で落とせるが、情報を捨てる操作なので既定は無効

いずれも放置すると、チャンクにノイズが入り、検索・生成の両方を劣化させる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragforlocalllm.core.registry import register
from ragforlocalllm.core.types import Document

_NUMBER_ONLY = re.compile(r"^\s*[0-9０-９]{1,4}\s*$")
_LEADER_DOTS = re.compile(r"[…‥.．]{3,}")
# 見出しは文の途中で終わらない。読点で終わる行は本文の折り返しとみなす。
_MID_SENTENCE = re.compile(r"[。．\.、，,]\s*$")


@dataclass
class _Fragment:
    """PyMuPDF の1 line。視覚的な行の断片であることが多い。"""

    text: str
    y0: float
    x0: float
    x1: float
    size: float
    bold: bool


@dataclass
class _Row:
    """同じ視覚的な行にある断片をまとめたもの。"""

    text: str
    page: int  # 1始まり
    y0: float
    size: float
    """行の代表フォントサイズ（最も文字数の多い断片のもの）。"""
    bold: bool
    in_margin: bool
    """ページ上下の余白帯にあるか。ヘッダ・フッタ・ページ番号の判定に使う。"""


@dataclass
class _Section:
    """見出しとその配下の本文。"""

    section_path: str
    title: str
    level: int
    page_start: int
    page_end: int
    lines: list[str] = field(default_factory=list)

    def text(self, *, include_heading: bool) -> str:
        body = "\n".join(self.lines).strip()
        if include_heading and self.level > 0:
            return f"{self.title}\n{body}".strip()
        return body


@register("loader", "pymupdf")
class PyMuPDFLoader:
    """PyMuPDF による抽出。座標で行と読み順を再構成し、ノイズを除去する。

    Parameters
    ----------
    emit:
        ``sections`` — 見出し単位で Document を分割する（既定）。
        後段のチャンカーが何であれ ``section_path`` とページが
        チャンクに引き継がれるため、評価アンカーの解決に有利。
        ``pages`` — ページ単位。``single`` — 文書全体を1件。
    heading_min_size:
        この文字サイズ以上かつ太字の行を見出し候補とする。
    heading_max_chars:
        これより長い行は見出しとみなさない（太字の本文を弾く）。
    heading_run_limit:
        同じサイズの見出し候補がこの数だけ連続したら、見出しではなく
        太字の段落とみなす（扉ページのリード文対策）。
    drop_rotated:
        水平でない行（縦書きサイドタブ）を除去する。
    margin_ratio:
        ページ高さのこの割合を上下の余白帯とみなす。ヘッダ・フッタと
        ページ番号の除去はこの帯の中だけで行う。
    header_footer_ratio:
        余白帯において、この割合以上のページに出現する同一行を
        ヘッダ・フッタとみなす。
    min_font_size:
        これ未満のフォントサイズの断片を落とす。図中ラベルの除去に使うが、
        情報を捨てる操作なので既定は無効（0.0）。
    include_heading:
        見出しの文字列を本文の先頭に含める。含めないと見出しが
        メタデータにしか残らず、検索対象から消える。
    label_min_pages:
        この数以上のページに同じ文字列の見出しが現れたら、それは固有の
        節見出しではなく装飾ラベル（「コラム」など）とみなし、最下位の
        階層に落として後続の節を配下に取り込ませない。
    replacements:
        抽出後に適用する文字列置換。PDFのフォントに ToUnicode の
        誤りがあると別の文字として抽出される（本コーパスでは箇条書きの
        記号が ``●Ө`` になる）。**PDF ごとの現象**なので実装に
        埋め込まず、設定で与える。
    skip_pages:
        除外するページ番号（1始まり）。表紙・目次を除く用途。
    """

    def __init__(
        self,
        emit: str = "sections",
        heading_min_size: float = 11.0,
        heading_max_chars: int = 40,
        heading_run_limit: int = 3,
        drop_rotated: bool = True,
        margin_ratio: float = 0.07,
        header_footer_ratio: float = 0.3,
        min_font_size: float = 0.0,
        include_heading: bool = True,
        label_min_pages: int = 3,
        replacements: dict[str, str] | None = None,
        skip_pages: tuple[int, ...] | list[int] = (),
        y_tolerance: float = 4.0,
        x_gap_space: float = 2.0,
        min_section_chars: int = 1,
    ) -> None:
        if emit not in ("sections", "pages", "single"):
            raise ValueError("emit は sections / pages / single のいずれかです")
        self.emit = emit
        self.heading_min_size = heading_min_size
        self.heading_max_chars = heading_max_chars
        self.heading_run_limit = heading_run_limit
        self.drop_rotated = drop_rotated
        self.margin_ratio = margin_ratio
        self.header_footer_ratio = header_footer_ratio
        self.min_font_size = min_font_size
        self.include_heading = include_heading
        self.label_min_pages = label_min_pages
        self.replacements = dict(replacements or {})
        self.skip_pages = set(skip_pages)
        self.y_tolerance = y_tolerance
        self.x_gap_space = x_gap_space
        self.min_section_chars = min_section_chars

    # ------------------------------------------------------------------

    def load(self, path: Path) -> list[Document]:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - 任意依存
            raise RuntimeError("PyMuPDF が必要です: uv sync --extra pdf") from exc

        path = Path(path)
        with fitz.open(path) as doc:
            rows = self._collect_rows(doc)
            n_pages = int(doc.page_count)
            title = doc.metadata.get("title") or path.stem

        rows = self._drop_repeated_margins(rows, n_pages)
        is_heading = self._heading_flags(rows)
        heading_levels = self._heading_levels(rows, is_heading)

        base_meta: dict[str, Any] = {
            "source": str(path),
            "title": title,
            "loader": "pymupdf",
            "n_pages": n_pages,
        }

        if self.emit == "single":
            return [
                Document(
                    doc_id=path.stem,
                    text="\n".join(row.text for row in rows),
                    metadata={**base_meta, "n_lines": len(rows)},
                )
            ]

        if self.emit == "pages":
            return self._emit_pages(path, rows, base_meta)

        return self._emit_sections(path, rows, is_heading, heading_levels, base_meta)

    # ------------------------------------------------------------------
    # 抽出とノイズ除去
    # ------------------------------------------------------------------

    def _collect_rows(self, doc: Any) -> list[_Row]:
        out: list[_Row] = []
        for pno in range(doc.page_count):
            page_no = pno + 1
            if page_no in self.skip_pages:
                continue
            page = doc[pno]
            height = float(page.rect.height)
            top_band = height * self.margin_ratio
            bottom_band = height * (1.0 - self.margin_ratio)

            fragments = self._page_fragments(page)
            for row in self._group_rows(fragments, page_no):
                row.in_margin = row.y0 < top_band or row.y0 > bottom_band
                # ページ番号は余白帯にしかない。章番号（本文上部の「1」）を
                # 守るため、帯の外の数字だけの行は残す。
                if row.in_margin and _NUMBER_ONLY.match(row.text):
                    continue
                out.append(row)
        return out

    def _page_fragments(self, page: Any) -> list[_Fragment]:
        fragments: list[_Fragment] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if self.drop_rotated and not _is_horizontal(line.get("dir", (1.0, 0.0))):
                    continue
                spans = [
                    s
                    for s in line["spans"]
                    if s["text"].strip() and float(s["size"]) >= self.min_font_size
                ]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                bbox = line["bbox"]
                # 代表サイズは最も文字数の多い span から取る。行頭の
                # 装飾記号や添え字にサイズを引きずられないようにする。
                lead = max(spans, key=lambda s: len(s["text"]))
                fragments.append(
                    _Fragment(
                        text=text,
                        y0=float(bbox[1]),
                        x0=float(bbox[0]),
                        x1=float(bbox[2]),
                        size=float(lead["size"]),
                        bold="Bold" in lead["font"],
                    )
                )
        return fragments

    def _group_rows(self, fragments: list[_Fragment], page_no: int) -> list[_Row]:
        """y座標が近い断片を1行にまとめ、x順に連結する。

        PyMuPDF の line は視覚的な行と一致しない。章番号と章題、
        あるいは折り返しのない1行が複数に割れるため、まとめ直さないと
        文が分断され gold 引用が解決できなくなる。
        """
        rows: list[_Row] = []
        for group in _cluster_by_y(fragments, self.y_tolerance):
            group.sort(key=lambda f: f.x0)
            parts: list[str] = []
            prev: _Fragment | None = None
            for frag in group:
                # 表紙の袋文字は同じ文字列が僅かにずれて重ね描きされている。
                # そのまま連結すると「中小企業の中小企業の」になる。
                if (
                    prev is not None
                    and frag.text == prev.text
                    and frag.x0 - prev.x0 < prev.x1 - prev.x0
                ):
                    continue
                if prev is not None and frag.x0 - prev.x1 > self.x_gap_space:
                    parts.append(" ")
                parts.append(frag.text)
                prev = frag
            text = self._apply_replacements("".join(parts).strip())
            if not text:
                continue
            lead = max(group, key=lambda f: len(f.text))
            rows.append(
                _Row(
                    text=text,
                    page=page_no,
                    y0=min(f.y0 for f in group),
                    size=lead.size,
                    bold=lead.bold,
                    in_margin=False,
                )
            )
        return rows

    def _apply_replacements(self, text: str) -> str:
        for src, dst in self.replacements.items():
            text = text.replace(src, dst)
        return text

    def _drop_repeated_margins(self, rows: list[_Row], n_pages: int) -> list[_Row]:
        """余白帯で多くのページに繰り返し現れる行を除去する。

        本文領域を対象にすると、定型的な言い回し（「〜します。」）まで
        巻き添えで落ちうるため、帯の中だけを見る。
        """
        if n_pages <= 2 or self.header_footer_ratio <= 0:
            return rows
        pages_by_text: dict[str, set[int]] = {}
        for row in rows:
            if row.in_margin:
                pages_by_text.setdefault(row.text, set()).add(row.page)
        threshold = max(int(n_pages * self.header_footer_ratio), 2)
        repeated = {text for text, pages in pages_by_text.items() if len(pages) >= threshold}
        return [row for row in rows if not (row.in_margin and row.text in repeated)]

    def _heading_flags(self, rows: list[_Row]) -> list[bool]:
        """各行が見出しかどうかを、前後の文脈も見て決める。"""
        flags = [self._is_heading_shape(row) for row in rows]
        self._unmark_heading_runs(rows, flags)
        return flags

    def _unmark_heading_runs(self, rows: list[_Row], flags: list[bool]) -> None:
        """同じサイズの見出し候補が連続したら、それは段落であって見出しではない。

        部の扉ページのリード文が太字 14.2pt の4行に分かれており、
        形だけ見ると見出しに合致する。放置すると section_path の根が
        「自らの責任で考えなければならない」のような文節になり、
        配下の実節がすべて誤った階層にぶら下がる。
        """
        start = 0
        while start < len(flags):
            if not flags[start]:
                start += 1
                continue
            end = start + 1
            size = round(rows[start].size, 1)
            while end < len(flags) and flags[end] and round(rows[end].size, 1) == size:
                end += 1
            if end - start >= self.heading_run_limit:
                for i in range(start, end):
                    flags[i] = False
            start = end

    def _heading_levels(self, rows: list[_Row], flags: list[bool]) -> dict[float, int]:
        """見出しの文字サイズを大きい順に階層レベルへ割り当てる。

        1回しか現れないサイズも階層に含める。章題はフォントサイズが
        文字数に応じて自動調整されており（15.4 / 17.0 など）、出現回数で
        足切りすると章題そのものが本文に落ちてしまう。
        """
        sizes = {
            round(row.size, 1) for row, is_heading in zip(rows, flags, strict=True) if is_heading
        }
        return {size: level for level, size in enumerate(sorted(sizes, reverse=True), start=1)}

    def _label_titles(self, rows: list[_Row], flags: list[bool]) -> set[str]:
        """複数ページに繰り返し現れる見出し文字列（＝装飾ラベル）。

        本コーパスの「コラム」は 15.6pt で描かれており、章題（15.4pt）
        より大きい。サイズだけで階層を決めると、コラム以降の本文の節が
        すべてコラムの配下にぶら下がってしまう。
        """
        if self.label_min_pages <= 0:
            return set()
        pages_by_title: dict[str, set[int]] = {}
        for row, is_heading in zip(rows, flags, strict=True):
            if is_heading:
                pages_by_title.setdefault(_clean_heading(row.text), set()).add(row.page)
        return {t for t, pages in pages_by_title.items() if len(pages) >= self.label_min_pages}

    def _is_heading_shape(self, row: _Row) -> bool:
        """見出しの「形」をしているか（前後の文脈は見ない）。"""
        if not row.bold or row.size < self.heading_min_size:
            return False
        if len(row.text) > self.heading_max_chars:
            return False
        return not _MID_SENTENCE.search(row.text)

    # ------------------------------------------------------------------
    # Document の組み立て
    # ------------------------------------------------------------------

    def _emit_pages(
        self, path: Path, rows: list[_Row], base_meta: dict[str, Any]
    ) -> list[Document]:
        by_page: dict[int, list[str]] = {}
        for row in rows:
            by_page.setdefault(row.page, []).append(row.text)
        docs = []
        for page, texts in sorted(by_page.items()):
            text = "\n".join(texts).strip()
            if len(text) < self.min_section_chars:
                continue
            docs.append(
                Document(
                    doc_id=f"{path.stem}#p{page:04d}",
                    text=text,
                    metadata={**base_meta, "page": page, "page_start": page, "page_end": page},
                )
            )
        return docs

    def _emit_sections(
        self,
        path: Path,
        rows: list[_Row],
        flags: list[bool],
        heading_levels: dict[float, int],
        base_meta: dict[str, Any],
    ) -> list[Document]:
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        current: _Section | None = None
        labels = self._label_titles(rows, flags)
        label_level = max(heading_levels.values(), default=0) + 1

        for row, is_heading in zip(rows, flags, strict=True):
            level = heading_levels.get(round(row.size, 1)) if is_heading else None
            if level is not None and _clean_heading(row.text) in labels:
                # 装飾ラベルは最下位に置き、後続の節を配下に取り込ませない
                level = label_level
            if level is None:
                if current is None:
                    # 最初の見出しより前の本文（表紙・前書き等）
                    current = _Section(
                        section_path="(前書き)",
                        title="(前書き)",
                        level=0,
                        page_start=row.page,
                        page_end=row.page,
                    )
                current.lines.append(row.text)
                current.page_end = row.page
                continue

            if current is not None:
                sections.append(current)

            title = _clean_heading(row.text)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = _Section(
                section_path=" > ".join(t for _, t in stack),
                title=title,
                level=level,
                page_start=row.page,
                page_end=row.page,
            )

        if current is not None:
            sections.append(current)

        docs: list[Document] = []
        for i, section in enumerate(sections):
            text = section.text(include_heading=self.include_heading)
            if len(text) < self.min_section_chars:
                continue
            docs.append(
                Document(
                    doc_id=f"{path.stem}#s{i:04d}",
                    text=text,
                    metadata={
                        **base_meta,
                        "section_path": section.section_path,
                        "heading": section.title,
                        "heading_level": section.level,
                        "page": section.page_start,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                    },
                )
            )
        return docs


@register("loader", "pypdf")
class PyPDFLoader:
    """pypdf による最小限の抽出。ノイズ除去も見出し検出も行わない。

    比較用のベースライン。**縦書きタブやヘッダが混入したまま**になるため、
    引用解決率とチャンク品質の下限を測る参照点として使う。
    """

    def __init__(self, emit: str = "pages") -> None:
        if emit not in ("pages", "single"):
            raise ValueError("pypdf ローダーは emit=pages / single のみ対応します")
        self.emit = emit

    def load(self, path: Path) -> list[Document]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - 任意依存
            raise RuntimeError("pypdf が必要です: uv sync --extra pdf") from exc

        path = Path(path)
        reader = PdfReader(str(path))
        base_meta: dict[str, Any] = {
            "source": str(path),
            "title": path.stem,
            "loader": "pypdf",
            "n_pages": len(reader.pages),
        }

        if self.emit == "single":
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return [Document(doc_id=path.stem, text=text, metadata=dict(base_meta))]

        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    doc_id=f"{path.stem}#p{i:04d}",
                    text=text,
                    metadata={**base_meta, "page": i, "page_start": i, "page_end": i},
                )
            )
        return docs


# ----------------------------------------------------------------------


def _cluster_by_y(fragments: list[_Fragment], tolerance: float) -> list[list[_Fragment]]:
    """y0 が ``tolerance`` 以内で連なる断片をまとめる。

    ``round(y / tolerance)`` によるビン分割では、境界をまたぐだけの
    近接した断片（章番号 y=61.5 と章題 y=63.1）が別行になってしまう。
    """
    groups: list[list[_Fragment]] = []
    for frag in sorted(fragments, key=lambda f: (f.y0, f.x0)):
        if groups and frag.y0 - groups[-1][0].y0 <= tolerance:
            groups[-1].append(frag)
        else:
            groups.append([frag])
    return groups


def _is_horizontal(direction: tuple[float, float] | list[float]) -> bool:
    """水平方向のテキストか。

    IPAガイドラインではページ端の縦書きインデックスが1文字ずつ行として
    現れる。``dir`` が ``(1, 0)`` 以外のものを除去することで落とせる。
    """
    return round(float(direction[0]), 1) == 1.0 and abs(float(direction[1])) < 0.1


def _clean_heading(text: str) -> str:
    """目次のリーダー線やページ番号を見出しから落とす。"""
    cleaned = _LEADER_DOTS.split(text)[0]
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text.strip()
