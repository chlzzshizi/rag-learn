#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切口体检：不靠肉眼，把每个切口"落在什么位置"分类统计出来。

## 它量的是切块，不是检索

这是本仓库里**唯一不依赖那 16 道题的硬指标** —— 切口落在哪，跟题库无关。
所以换切法时，"分数涨没涨"可以吵，"切口还切不切在词中间"没得吵。

## 偏移从哪来

**从切块器直出**（`chunking.Piece.offset`），不反推。原来的 `offsets_of()`
用「块长累加」反推，那个法子假设块与块首尾相接 —— 而这三处都会破坏它：
注入的章节前缀、合块时插入的 `\n`、以及块本身就跨了段界。

## 分类的轴是「行边界 vs 行中间」，不是「切在哪一段里面」

    切在行边界  —— 干净（表格在**行边界**处断开是**正确**切法，配合补回表头就是好块）
    切在行中间  —— 真硬伤（中文里往往就是切在词中间）

早先的版本把「落在表格区间内」一律算硬伤，**结果把正确的行边界切法也判成了硬伤
—— 指标错了，结论就会反**。所以另按「切口在不在围栏/表格区间里」细分成三列，
只作归因用，不参与判定。

## 另外四个不依赖偏移的量

    落单围栏块      块里 ``` 出现奇数次 —— 代码块被劈开又没补回围栏
    无表头表格块    块以 | 开头、第二行不是 |---| —— 数据行丢了表头，读不出每列是什么
    跨节块          块跨了节界，`[章节]` 前缀只描述得到块首
    偏移自洽失败    offset 处的真实所属节 ≠ 块记录的 path —— **非 0 就说明锚定位有 bug**

前两个是「结构被撕碎」的直接证据，也是换切法的真实理由。**后两个是本脚本的自检**：
`跨节块` 是 `[章节]` 注入这条设计的已知代价（照实报出来），
`偏移自洽失败` 一旦非 0，上面按偏移算的切口分类就都不作数了。

用法：

    .venv/Scripts/python.exe scripts/inspect_cuts.py                 # md 和 flat 都查
    .venv/Scripts/python.exe scripts/inspect_cuts.py --chunker md
    .venv/Scripts/python.exe scripts/inspect_cuts.py --detail        # 打印切坏的位置
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chunking  # noqa: E402
import rag  # noqa: E402


def find_fenced_ranges(text: str) -> list[tuple[int, int]]:
    """所有围栏代码块（``` 或 ~~~ 包起来的）的字符区间。"""
    ranges, pos, open_at, token = [], 0, None, None
    for line in text.splitlines(keepends=True):
        s = line.lstrip()
        tok = s[:3] if (s.startswith("```") or s.startswith("~~~")) else None
        if tok is not None:
            if open_at is None:
                open_at, token = pos, tok
            elif tok == token:                      # 同类围栏才算闭合
                ranges.append((open_at, pos + len(line)))
                open_at = None
        pos += len(line)
    if open_at is not None:
        ranges.append((open_at, len(text)))
    return ranges


def find_table_ranges(text: str) -> list[tuple[int, int]]:
    """所有 markdown 表格（连续以 | 开头的行）的字符区间。"""
    ranges, pos, start = [], 0, None
    for line in text.splitlines(keepends=True):
        is_row = line.lstrip().startswith("|")
        if is_row and start is None:
            start = pos
        elif not is_row and start is not None:
            if pos - start > 1:
                ranges.append((start, pos))
            start = None
        pos += len(line)
    if start is not None:
        ranges.append((start, len(text)))
    return ranges


def in_ranges(offset: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a < offset < b for a, b in ranges)


def _headings_at(text: str) -> list[tuple[int, int, str]]:
    """(偏移, 层级, 标题) —— **围栏内的 `#` 不算标题**（bug 文档有 11 行 shell 注释）。"""
    fences = find_fenced_ranges(text)
    out, pos = [], 0
    for line in text.split("\n"):
        if not in_ranges(pos, fences):
            m = chunking.HEAD_RE.match(line.lstrip())
            if m:
                out.append((pos, len(m.group(1)), m.group(2).strip()))
        pos += len(line) + 1
    return out


def _path_at(hs: list[tuple[int, int, str]], off: int) -> dict[int, str]:
    p: dict[int, str] = {}
    for o, lvl, t in hs:
        if o > off:
            break
        p[lvl] = t
        for k in [k for k in p if k > lvl]:
            del p[k]
    return p


def spans_sections(piece) -> bool:
    """块**跨了节**吗 —— 即 `[章节]` 前缀描述不到块尾。

    前缀（`Piece.path`）取的是块的**首片**所属的那一节。而合块是跨节界的，
    所以一个块很可能「开头在 §2.1、结尾已经进了 §3」，前缀却只说 §2.1。

    **判在块自己的正文上，不用偏移** —— 偏移万一算错，这个指标照样准。

    ⚠️ 状态必须**用 `piece.path` 起头**。块里常常只有 `### 1.2` 而没有祖先
    `## 一、`，从零搭路径会缺级，跟完整路径一比就全成了「跨节」——
    那样量出来的是「块里有没有祖先标题」，不是「跨没跨节」。
    """
    if not piece.path:          # flat 切法不记 path，跳过（否则全线误报）
        return False
    cur = dict(piece.path)
    for _o, lvl, t in _headings_at(piece.text):
        cur[lvl] = t
        for k in [k for k in cur if k > lvl]:
            del cur[k]
        if cur != piece.path:   # 走到别的节里去了，前缀管不到这里
            return True
    return False


def headerless_table_blocks(pieces: list) -> int:
    """**开头是表格行、却没有表头的块**。

    这是表格被切坏的**直接**证据，而且不依赖偏移算得对不对：
    一块以 `|` 开头，而它的第二行不是 `|---|---|` 这种分隔行 —— 说明它是一条
    没有表头的数据行，读它的人和模型都不知道每列是什么。
    """
    n = 0
    for p in pieces:
        lines = p.text.split("\n")
        if (len(lines) >= 2 and lines[0].lstrip().startswith("|")
                and not chunking.TABLE_SEP_RE.match(lines[1])):
            n += 1
    return n


def analyze(label: str, text: str, pieces: list, detail: bool) -> dict:
    """切口的分类。**分类的轴是「切在行边界还是行中间」，不是「切在哪一段里面」。**

    之所以按行边界分：表格在**行边界**处断开是**正确**切法（配合补回表头就是好块），
    **行中间**断开才是真硬伤。早先的版本把「落在表格区间内」一律算硬伤，
    结果把正确的行边界切法也判成了硬伤 —— 指标错了，结论就会反。
    """
    fences = find_fenced_ranges(text)
    tables = find_table_ranges(text)
    starts = [p.offset for p in pieces[1:]]      # 跳过第 0 块：它不是切口

    def at_line_start(off: int) -> bool:
        """切口是不是落在**行的内容起点**上。

        ⚠️ 不能只判 `text[off-1] == "\\n"`：**RCT 的 `_join_docs` 会 strip 片首**，
        而嵌在列表里的表格/代码是**缩进**的（`    | 4 | 待出厂 |`），
        于是偏移落在内容的第一个字符上、前面还留着缩进空格。
        那种情况切的就是行边界，只是偏了几格 —— 判成「切在行中间」是假阳性。
        """
        if off == 0:
            return True
        nl = text.rfind("\n", 0, off)
        return not text[nl + 1:off].strip()

    mid: list[tuple[int, str]] = []              # 切在行中间的（真硬伤）
    at_line = 0
    for off in starts:
        in_f = in_ranges(off, fences)
        in_t = in_ranges(off, tables)
        if at_line_start(off):
            at_line += 1
            continue
        mid.append((off, "代码块内" if in_f else "表格行内" if in_t else "普通正文"))

    bucket = {kk: sum(1 for _o, k in mid if k == kk)
              for kk in ("普通正文", "代码块内", "表格行内")}
    n = max(len(starts), 1)
    lengths = [len(p.text) for p in pieces]
    lone = sum(1 for p in pieces if p.lone_fences % 2)   # ``` 出现奇数次 = 劈开了没补回
    headerless = headerless_table_blocks(pieces)
    over = sum(1 for x in lengths if x > chunking.CHUNK_SIZE)
    clean = sum(1 for p in pieces
                if p.lone_fences % 2 == 0 and len(p.text) <= chunking.CHUNK_SIZE)
    cross = sum(1 for p in pieces if spans_sections(p))
    # 偏移自洽：每个块记的 offset 处的真实所属节，必须等于它记的 path。
    # 这是一个**内部一致性断言** —— 它一旦不为 0，说明锚定位错了，
    # 上面按偏移算的切口分类也就跟着不可信。（实测踩过：围栏被切开时第 0 片的
    # 锚退化成了裸 ```，`find` 匹配到别的那道围栏，2 个块的 offset 落到了别的节。）
    hs = _headings_at(text)
    bad_off = sum(1 for p in pieces
                  if p.path and _path_at(hs, p.offset) != p.path)

    print(f"\n{'=' * 74}")
    print(f"{label}")
    print("=" * 74)
    print(f"  {len(pieces)} 块，{len(starts)} 个切口；块长 {min(lengths)} ~ {max(lengths)}，"
          f"中位 {sorted(lengths)[len(lengths) // 2]}，平均 {sum(lengths) / len(lengths):.0f}")
    print(f"    切在行边界（干净）  {at_line:>4} 个  ({at_line / n * 100:5.1f}%)")
    print(f"    切在行中间（硬伤）  {len(mid):>4} 个  ({len(mid) / n * 100:5.1f}%)"
          f"  其中 正文 {bucket['普通正文']} / 代码块内 {bucket['代码块内']}"
          f" / 表格内 {bucket['表格行内']}")
    print(f"  → 落单围栏的块：{lone} 个     ← 代码块被劈开又没补回 ```")
    print(f"  → **无表头表格块**：{headerless} 个     ← 数据行丢了表头，读不出每列是什么")
    print(f"  → 超 {chunking.CHUNK_SIZE} 字的块：{over} 个")
    print(f"  → **干净块占比**（不超长 且 无落单围栏）："
          f"{clean}/{len(pieces)} = {clean / len(pieces) * 100:.1f}%")
    if any(p.path for p in pieces):
        print(f"  → **跨节块**：{cross} 个 ({cross / len(pieces) * 100:.1f}%)"
              f"     ← 合块跨了节界，`[章节]` 前缀只描述得到块首、描述不到块尾")
        print(f"  → 偏移自洽失败：{bad_off} 个"
              f"{'（0 = 偏移落点与记录的章节一致）' if bad_off == 0 else '  ⚠ **锚定位有 bug，上面的切口分类不可信**'}")

    if detail:
        for p in pieces:
            if spans_sections(p):
                print(f"\n    跨节块（{len(p.text)} 字）前缀 {p.heading}")
                print(f"      首行 {p.text.splitlines()[0][:50]!r}")
                print(f"      末行 {p.text.splitlines()[-1][:50]!r}")

    if detail:
        for off, kind in mid:
            print(f"\n    偏移 {off:,} 切在【{kind}】的行中间")
            print(f"      上一块结尾 …{text[max(0, off - 40):off]!r}")
            print(f"      下一块开头 {text[off:off + 40]!r}…")

    return {"chunks": len(pieces), "cuts": len(starts), "at_line": at_line,
            "mid": len(mid), "mid_正文": bucket["普通正文"],
            "mid_代码块": bucket["代码块内"], "mid_表格": bucket["表格行内"],
            "lone": lone, "headerless": headerless, "over": over, "clean": clean,
            "cross": cross, "bad_off": bad_off}


def main() -> None:
    ap = argparse.ArgumentParser(description="切口体检")
    ap.add_argument("--chunker", choices=[*chunking.CHUNKERS, "both"], default="both")
    ap.add_argument("--size", type=int, default=chunking.CHUNK_SIZE)
    ap.add_argument("--inject", action="store_true", default=True,
                    help="块文本带 [章节] 前缀（默认带；切口位置不受它影响）")
    ap.add_argument("--no-structure", dest="structure", action="store_false",
                    help="二次切分不用结构安全版（对照）")
    ap.add_argument("--detail", action="store_true", help="打印切在代码块/表格内部的位置")
    args = ap.parse_args()

    names = list(chunking.CHUNKERS) if args.chunker == "both" else [args.chunker]
    corpora = list(rag.corpus_texts())

    print(f"语料 {len(corpora)} 份，合计 {sum(len(t) for _n, t in corpora):,} 字符")
    for name, text in corpora:
        f, tb = find_fenced_ranges(text), find_table_ranges(text)
        print(f"  {name}：{len(f)} 个围栏代码块（{sum(b - a for a, b in f):,} 字符）"
              f" + {len(tb)} 张表格（{sum(b - a for a, b in tb):,} 字符）"
              f" = 占 {(sum(b - a for a, b in f) + sum(b - a for a, b in tb)) / len(text) * 100:.1f}%")

    # 两份语料各自体检，再汇总。**必须分开算切口** —— 块号跨文件连续，
    # 但偏移是各文件自己的坐标，混在一起算会得出假切口。
    totals: dict[str, dict] = {}
    per_file: dict[str, list[list]] = {}
    for chunker in names:
        per_file[chunker] = []
        for name, text in corpora:
            ps = chunking.pieces_of(text, chunker, size=args.size,
                                    inject=args.inject, structure=args.structure)
            per_file[chunker].append(ps)

    KEYS = ["chunks", "cuts", "at_line", "mid", "mid_正文", "mid_代码块",
            "mid_表格", "lone", "headerless", "over", "clean", "cross", "bad_off"]
    for chunker in names:
        agg = dict.fromkeys(KEYS, 0)
        for (name, text), ps in zip(corpora, per_file[chunker]):
            r = analyze(f"{chunker}  {name}", text, ps, args.detail)
            for k in KEYS:
                agg[k] += r[k]
        totals[chunker] = agg

    if len(names) == 2:
        a, b = (totals[n] for n in names)
        print(f"\n{'=' * 74}")
        print(f"汇总对比（{names[0]} → {names[1]}）")
        print("=" * 74)
        for k, label in (("chunks", "块数"), ("cuts", "切口数"),
                         ("at_line", "切在行边界"), ("mid", "切在行中间"),
                         ("lone", "落单围栏块"), ("headerless", "无表头表格块"),
                         ("over", f"超 {args.size} 字块"), ("clean", "干净块"),
                         ("cross", "跨节块"), ("bad_off", "偏移自洽失败")):
            print(f"  {label:<12} {a[k]:>5}  →  {b[k]:>5}   ({b[k] - a[k]:+d})")
        print(f"\n  切在行中间占比  {a['mid'] / max(a['cuts'], 1) * 100:>5.1f}%  →  "
              f"{b['mid'] / max(b['cuts'], 1) * 100:>5.1f}%")
        print(f"  干净块占比      {a['clean'] / a['chunks'] * 100:>5.1f}%  →  "
              f"{b['clean'] / b['chunks'] * 100:>5.1f}%")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
