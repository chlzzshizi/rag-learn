#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切块：从「500 字硬切」到「按结构切」。

## 为什么要换 —— ⚠️ 先纠正一个曾经的错误前提

写这个模块时的理由是「**547 块里 96.7% 的切口落在词中间**」。**那个数字是错的**，
它是 `scripts/inspect_cuts.py` 的旧版测出来的假象，两个 bug 叠出来：

1. 旧版分析的是**原始文件**，而块是从 `"# 文件：x.md\\n\\n" + 正文`（多 57 字符）切的
2. 旧版用**块长累加**反推偏移（`offsets_of`），它假设块首尾相接 —— 而 RCT 的
   `_join_docs` 会 strip 每一片，累计误差越滚越大

平均行长约 55 字符，于是约 57 字符的偏移误差**恰好把偏移打到行中间**。
修好之后实测：**flat 的切口有 99.1% 落在行边界**，不是 3.3%。

**所以换切法的真实理由只有一个：结构被撕碎。** 这一条是量出来的、站得住的：

- 设计文档最大的 2275 字符代码块被切成 5 段，其中 2 段带**落单的 ```**
- bug 文档那张 49 行 / 7594 字符的表格被切成 20 段，**只有第 1 段带表头和 `|---|`**，
  其余 19 段是无表头的数据行 —— 表头丢了的数据行，人和模型都读不出是什么

硬指标对照（`inspect_cuts.py`，538 块 vs 547 块）：落单围栏 **0 vs 22**、
无表头表格块 **1 vs 20**；代价是超 500 字的块 **7 vs 0**、切在行中间 **2.1% vs 0.9%**。
**是取舍，不是全面胜出** —— README 里照实记着。

## 三段式

    ① 按标题切     自己写的围栏感知扫描（**见下面「为什么不用现成的库」**）
    ② 二次切分     超过 size 的段再切，且**不撕开围栏和表格**
    ③ 合块         小片贪心合并到接近 size

**每一步都是必需的，不是加复杂度**（都有实测数据支撑）：

- 不按标题切：块没有上下文，一块是「4 | 待出厂 | 洗涤完成」，读完不知道这是哪一节
- 不二次切分：设计文档最大 H2 = 20,014 字符、bug 文档最大 H2 = **28,892** 字符
- 不合块：bug 文档有 254 个 H3、中位仅 292 字符、24 个 <100，只按标题切 = 77% 的块不足 500，
  把「少而大」换成「多而碎」；**实测不合块 4~5/16，合块 8~9/16，差 3~4 题**

## 为什么不用现成的 MarkdownHeaderTextSplitter

因为**它会破坏正文**。`.venv/Lib/site-packages/langchain_text_splitters/markdown.py:164`
对每一行做 `stripped_line = line.strip()`，然后把**剥过的行**拼进 `page_content`：

    原文  '    Label: "1"'      →  MHS 出来的正文  'Label: "1"'

这份语料里满是缩进的 YAML / SQL / 代码块，**缩进一丢就不是原文了**，
「块 = 语料的连续切片」这条性质直接不成立（而本项目的评测正是建立在这条上的）。
所以只留它两个正确的想法，自己写：

- **代码块内的 `#` 不是标题**（bug 文档里有 11 行 shell 注释以 `#` 开头；
  用纯正则扫会得到 318 行，其中 11 行是假的）
- 标题进路径、进正文之外的 metadata，正文保持逐字

## 偏移（offset）是怎么来的 —— 这里有个坑

`scripts/inspect_cuts.py` 原来用「块长累加」反推切口偏移（`offsets_of`），
那个法子假设块与块首尾相接。这套切法有三处破坏它：**注入的章节前缀、
合块时插入的 `\\n`、被摘掉的标题行**。所以偏移由这里直出，下游不许反推：

    每个块只记**它正文的前 60 个字符**当锚，用单调游标在原文里往后找。
    合块后的整块文本在原文里根本不存在，但它的前 60 字符一定在。

用法：

    from chunking import CHUNKERS, chunk_markdown, chunk_flat
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0

# 中文没有空格，默认的 " " 分隔符基本用不上；显式加上中文标点，让它在句子/分句的边界上找落脚点。
SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]

HEAD_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
TABLE_RE = re.compile(r"^\s*\|")
# 表格的分隔行：| --- | :---: | 之类，只有 | - : 空格
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")

# 锚的长度。60 字符足够独特，又短到不会跨出「片内部」的范围（见 _anchor）
ANCHOR_LEN = 60


@dataclass
class Piece:
    """一块。`text` 是**正文**，不含注入的章节前缀。"""

    text: str
    offset: int                      # 在**带头部的语料文本**里的起始偏移
    path: dict[int, str] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        """`h1 > h2 > h3` 形式的章节路径。键是层级数字，升序正好等于层级序。"""
        return " > ".join(self.path[k] for k in sorted(self.path))

    def render(self, inject: bool = True) -> str:
        """最终进索引/进 prompt 的文本。`inject=False` 时逐字等于语料切片。"""
        h = self.heading
        return f"[章节] {h}\n{self.text}" if (inject and h) else self.text

    @property
    def lone_fences(self) -> int:
        """落单的围栏标记行数。**抓「代码块被切开又没补回 ```」**，奇数就是有落单。"""
        return sum(1 for ln in self.text.split("\n") if FENCE_RE.match(ln))


def _rct(size: int, overlap: int = 0):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        length_function=len, separators=SEPARATORS)


def _anchor(text: str) -> str:
    """块的定位锚 = 正文**前 60 个字符**（跳掉行首空白）。

    三条约束逼出来的取法：

    1. **不能拿整块找**。合块后的文本在原文里根本不存在（合块插了额外的 `\\n`），
       补回表头的表格片段也一样。
    2. **不能拿整行找**。基线里有 0.9% 的切口落在行中间（`chunk_flat` 的分隔符
       最后一级是 `""`，会硬切字符），那时候「第一行」是半截行，按行匹配必然落空。
    3. **前 60 字符一定安全**。RCT 的 `_join_docs` 会 `strip()` 每一片，
       但只动片首片尾 —— 前 60 字符要么是整块（短块），要么落在片内部，逐字不动。

    切口在行中间时锚就是那半截行，仍然能精确定位，只是偏移不落在行首而已
    —— 对「这一刀切在哪」的判定没有影响。
    """
    return text.lstrip()[:ANCHOR_LEN]


def _locate(text: str, anchors: list[str]) -> list[int]:
    """每个锚在 text 里的偏移。锚按原文顺序给，游标只往前走。

    游标每次只前进到「锚起点 + 1」而不是锚的末尾 —— 因为合块块的正文长度
    大于它在原文里的跨度，按长度推会冲过头。
    """
    out, cursor = [], 0
    for a in anchors:
        at = text.find(a, cursor)
        if at < 0:
            raise ValueError(f"锚在原文里找不到，切块器有 bug：{a[:60]!r}")
        out.append(at)
        cursor = at + 1
    return out


# ---------------------------------------------------------------- ① 按标题切


def md_sections(text: str) -> list[tuple[str, dict[int, str]]]:
    """按标题切成 (正文, 章节路径)。正文是**原文的连续切片，逐字不动**。

    **标题行留在正文里**（`buf.append(line)`），不摘出去。理由有两条，
    第二条是实测的：

    1. 摘掉标题行，正文就不再是原文的连续切片了，而且一块从半截内容开始、
       不注入章节路径的话根本不知道自己属于哪一节
    2. 实测**摘出去比留下差**：不注入时 k4 6/16 → 7/16，注入时 k8 8/16 → 9/16

    唯一要防的是**代码块内的 `#`**（bug 文档里 11 行 shell 注释以 `#` 开头）。
    ``` 和 ~~~ 两种围栏都认，而且围栏内不再判标题。
    """
    out: list[tuple[str, dict[int, str]]] = []
    buf: list[str] = []
    path: dict[int, str] = {}
    in_fence: str | None = None

    def flush() -> None:
        body = "\n".join(buf).strip("\n")
        if body.strip():
            out.append((body, dict(path)))
        buf.clear()

    for line in text.split("\n"):
        s = line.lstrip()
        if in_fence is not None:
            buf.append(line)
            if s.startswith(in_fence):
                in_fence = None
            continue
        if s.startswith("```") or s.startswith("~~~"):
            in_fence = s[:3]
            buf.append(line)
            continue
        m = HEAD_RE.match(s)
        if m:
            flush()
            lvl = len(m.group(1))
            path[lvl] = m.group(2).strip()
            for k in [k for k in path if k > lvl]:   # 换到浅层就砍掉更深的
                del path[k]
            buf.append(line)                         # ← 标题行进正文
            continue
        buf.append(line)
    flush()
    return out


# ---------------------------------------------------------------- ② 结构安全二次切分
#
# 目的：二次切分**不许撕开围栏和表格**。做法是先把正文切成"原子块"
# （围栏块 / 表格 / 普通文字），普通文字交给 RCT，围栏和表格各自按自己的规则切，
# 并且**把语义上必需的头部补回去**（围栏补 ```，表格补表头行）。


def _atoms(lines: list[str]) -> list[tuple[str, list[str]]]:
    """把行序列切成原子块：围栏代码块 / 表格 / 普通文字。"""
    out: list[tuple[str, list[str]]] = []
    buf: list[str] = []
    kind = "text"
    fence: str | None = None

    def flush() -> None:
        if buf:
            out.append((kind, list(buf)))
            buf.clear()

    for line in lines:
        if fence is not None:                      # 围栏内：一切照收，直到闭合
            buf.append(line)
            if line.lstrip().startswith(fence):
                fence = None
                flush()
            continue
        m = FENCE_RE.match(line)
        if m:                                      # 围栏开：上一段收尾，换个原子块
            flush()
            kind, fence = "fence", m.group(1)
            buf.append(line)
            continue
        if TABLE_RE.match(line):
            if kind != "table":
                flush()
                kind = "table"
            buf.append(line)
            continue
        if kind == "table":                        # 表格结束
            flush()
            kind = "text"
        buf.append(line)
    flush()
    return out


def _greedy_break(lines: list[str], budget: int) -> list[list[str]]:
    """按行打包到 budget 以内，**优先在空行处断开**（空行是代码里天然的语义缝）。"""
    out: list[list[str]] = []
    start = 0
    while start < len(lines):
        used, end, last_blank = 0, start, None
        while end < len(lines):
            add = len(lines[end]) + 1
            if used + add > budget and end > start:
                break
            used += add
            if not lines[end].strip():
                last_blank = end
            end += 1
        if end >= len(lines):
            out.append(lines[start:])
            break
        cut = (last_blank + 1) if (last_blank is not None and last_blank >= start) else end
        cut = cut if cut > start else start + 1
        out.append(lines[start:cut])
        start = cut
    return out


def _split_fence(atom: list[str], size: int) -> list[tuple[str, str]]:
    """切开过大的围栏代码块，**每片都补回 ``` 和语言标记**。返回 (文本, 锚)。"""
    whole = "\n".join(atom)
    if len(atom) < 3:
        return [(whole, _anchor(whole))]
    open_line, close_line = atom[0], atom[-1]
    body = atom[1:-1]
    # 留出围栏两行 + 两个换行的位置
    budget = max(size - len(open_line) - len(close_line) - 2, 1)
    out = []
    for i, g in enumerate(_greedy_break(body, budget)):
        # 第 0 片的锚**不能只取围栏行**：`\`\`\`` 只有 3 个字符，原文里到处都是，
        # `_locate` 会匹配到别的那道围栏，offset 就落到别的节里去了（实测 2 块中招）。
        # 取「围栏行 + 首行正文」—— 它在原文里仍是连续的，但几乎不会重复。
        anchor_src = "\n".join([open_line, g[0]]) if (i == 0 and g) else \
                     open_line if i == 0 else "\n".join(g)
        out.append(("\n".join([open_line, *g, close_line]), _anchor(anchor_src)))
    return out


def _split_table(atom: list[str], size: int) -> list[tuple[str, str]]:
    """切开过大的表格，**每片都重复表头 + 分隔行**。返回 (文本, 锚)。

    这是 C 改动的重点：表头一旦丢了，剩下的数据行读不出是什么
    —— 实测原来那张 7594 字符的表格被切成 20 段，19 段没有表头。
    """
    whole = "\n".join(atom)
    if len(atom) < 3 or not TABLE_SEP_RE.match(atom[1]):
        return [(whole, _anchor(whole))]           # 不像表格，不硬来
    head, sep, rows = atom[0], atom[1], atom[2:]
    budget = max(size - len(head) - len(sep) - 2, 1)
    out = []
    for i, g in enumerate(_greedy_break(rows, budget)):
        # 同理：表头行可能跟别处重复，取「表头 + 分隔行 + 首条数据行」
        anchor_src = "\n".join([head, sep, g[0]]) if (i == 0 and g) else \
                     head if i == 0 else "\n".join(g)
        out.append(("\n".join([head, sep, *g]), _anchor(anchor_src)))
    return out


def _split_body(text: str, size: int, structure: bool) -> list[tuple[str, str]]:
    """二次切分一个超长 section。返回 (文本, 锚) 列表。"""
    if not structure:
        return [(s, _anchor(s)) for s in _rct(size).split_text(text)]
    out: list[tuple[str, str]] = []
    for kind, lines in _atoms(text.split("\n")):
        whole = "\n".join(lines)
        if len(whole) <= size:
            out.append((whole, _anchor(whole)))
        elif kind == "fence":
            out.extend(_split_fence(lines, size))
        elif kind == "table":
            out.extend(_split_table(lines, size))
        else:
            # 普通文字交给 RCT —— 它本来就是干这个的，中文分隔符也是为它调的
            out.extend((s, _anchor(s)) for s in _rct(size).split_text(whole))
    return out


# ---------------------------------------------------------------- 三种切法


def chunk_flat(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[Piece]:
    """**现在的切法**（整文件 RCT），留作 `--chunker flat` 的对照组。

    块数应当与换切法之前完全一致（实测 547 块），旧数字靠它复现。
    """
    parts = _rct(size, overlap).split_text(text)
    return [Piece(t, o) for t, o in zip(parts, _locate(text, [_anchor(p) for p in parts]))]


def chunk_markdown(text: str, size: int = CHUNK_SIZE,
                   inject: bool = True, structure: bool = True) -> list[Piece]:
    """按结构切。`structure=False` 退回「二次切分也用 RCT」的版本（C 改动的对照组）。"""
    # ① 按标题切 → ② 段内二次切分
    parts: list[tuple[str, dict[int, str], str]] = []
    for body, path in md_sections(text):
        subs = ([(body, _anchor(body))] if len(body) <= size
                else _split_body(body, size, structure))
        parts.extend((t, path, a) for t, a in subs)

    # ③ 合块。**跨段界合并** —— 段内合并实测 6/16，差 2 题
    merged: list[tuple[str, dict[int, str], str]] = []
    buf, bpath, banchor = "", {}, ""
    for t, path, anchor in parts:
        cand = f"{buf}\n{t}" if buf else t
        if len(cand) <= size or not buf:
            if not buf:
                bpath, banchor = path, anchor
            buf = cand
        else:
            merged.append((buf, bpath, banchor))
            buf, bpath, banchor = t, path, anchor
    if buf:
        merged.append((buf, bpath, banchor))

    offsets = _locate(text, [a for _t, _p, a in merged])
    return [Piece(t, o, p) for (t, p, _a), o in zip(merged, offsets)]


# 名字必须短 —— 它会被写进 data/lc_index/build.json 并四处打印。
CHUNKERS = {"md": chunk_markdown, "flat": chunk_flat}

# 切法的说明，给 --chunker 的 help 和索引自述共用
CHUNKER_DOC = {
    "md": "按标题切 + 结构安全二次切分 + 合块（默认）",
    "flat": "整文件 RecursiveCharacterTextSplitter 500/0（旧切法，对照用）",
}


def pieces_of(text: str, chunker: str = "md", size: int = CHUNK_SIZE,
              inject: bool = True, structure: bool = True) -> list[Piece]:
    """一份语料 → Piece 列表（`render()` 之后才是最终文本）。"""
    fn = CHUNKERS[chunker]
    if chunker == "flat":
        return fn(text, size)
    return fn(text, size, inject=inject, structure=structure)


def split_corpus(text: str, chunker: str = "md", size: int = CHUNK_SIZE,
                 inject: bool = True, structure: bool = True) -> list[str]:
    """一份语料 → 最终块文本列表。`rag.build_index` 和免费筛共用这一条路。"""
    return [p.render(inject) for p in pieces_of(text, chunker, size, inject, structure)]
