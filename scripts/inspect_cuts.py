#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切口体检：不靠肉眼，把每个切口"落在什么位置"分类统计出来。

第 1 步肉眼翻切口是应该做的，但肉眼只能看出"读不通"，
看不出"这个切口落在代码块内部"——后者要跨行才能判断，人翻 148 行必漏。

这个脚本做三件事：
  1. 证明全切了（偏移连续、无缝隙、无重叠、总长相等）
  2. 清点语料里"能被切坏的结构"有多少（代码块、表格）
  3. 给每个切口分类：落在行首 / 句中 / 代码块内 / 表格行内

用法：
    .venv/Scripts/python.exe scripts/inspect_cuts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from step1_chunk import CORPUS, CHUNK_SIZE, chunk_text, load_text  # noqa: E402


def find_fenced_ranges(text: str) -> list[tuple[int, int]]:
    """返回所有围栏代码块（``` 包起来的部分）的字符区间。

    做法：逐行扫，遇到以 ``` 开头的行就翻转状态。奇数个开围栏意味着
    在块内。markdown 里嵌套围栏很少见，这个近似足够用。
    """
    ranges = []
    pos = 0
    open_at = None
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if open_at is None:
                open_at = pos
            else:
                ranges.append((open_at, pos + len(line)))
                open_at = None
        pos += len(line)
    if open_at is not None:            # 围栏没闭合，一路到文件尾
        ranges.append((open_at, len(text)))
    return ranges


def find_table_ranges(text: str) -> list[tuple[int, int]]:
    """返回所有 markdown 表格的字符区间（连续的以 | 开头的行算一张表）。"""
    ranges = []
    pos = 0
    start = None
    for line in text.splitlines(keepends=True):
        is_row = line.lstrip().startswith("|")
        if is_row and start is None:
            start = pos
        elif not is_row and start is not None:
            if pos - start > 1:        # 单行不算表
                ranges.append((start, pos))
            start = None
        pos += len(line)
    if start is not None:
        ranges.append((start, len(text)))
    return ranges


def in_ranges(offset: int, ranges: list[tuple[int, int]]) -> bool:
    """offset 是不是落在某个区间**内部**（贴边不算）。"""
    return any(a < offset < b for a, b in ranges)


def main() -> None:
    text = load_text(CORPUS)
    chunks = chunk_text(text, CHUNK_SIZE)

    # ---------- 1. 全切了吗 ----------
    print("=" * 66)
    print("1. 覆盖检查 —— 全文都进块了吗")
    print("=" * 66)

    contiguous = all(chunks[i]["end"] == chunks[i + 1]["start"] for i in range(len(chunks) - 1))
    total = sum(c["end"] - c["start"] for c in chunks)
    checks = [
        ("第一块从 0 开始", chunks[0]["start"] == 0),
        ("最后一块到文件尾", chunks[-1]["end"] == len(text)),
        ("块与块首尾相接，无缝隙", contiguous),
        ("总长 == 文件字符数", total == len(text)),
    ]
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"\n  0 → {chunks[-1]['end']:,} 连续覆盖 {len(text):,} 字符，"
          f"切成 {len(chunks)} 块，一块不多一块不少。")
    print("  **没有任何字符被跳过。** 不重叠的固定切块，这两件事是等价的。")

    # ---------- 2. 语料里有什么可切的 ----------
    print()
    print("=" * 66)
    print('2. 语料结构清点 —— 有什么东西是"会被切坏"的')
    print("=" * 66)

    fences = find_fenced_ranges(text)
    tables = find_table_ranges(text)

    print(f"  围栏代码块  {len(fences):>4} 处，共 {sum(b - a for a, b in fences):>6,} 字符")
    print(f"  markdown 表格 {len(tables):>4} 张，共 {sum(b - a for a, b in tables):>6,} 字符")

    fence_lines = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("```"))
    table_lines = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("|"))
    lines = text.count("\n") + 1
    print(f"\n  （交叉核对：以 ``` 开头的行 {fence_lines} 行；以 | 开头的行 {table_lines} 行；全文 {lines:,} 行）")
    print(f"  合计占全文 {(sum(b - a for a, b in fences) + sum(b - a for a, b in tables)) / len(text) * 100:.1f}%")

    # ---------- 3. 每个切口落在哪 ----------
    print()
    print("=" * 66)
    print("3. 切口分类 —— 148 个切口各自落在什么位置")
    print("=" * 66)

    buckets = {"代码块内": [], "表格行内": [], "行首(切在换行处)": [], "句中(切在行中间)": []}
    for c in chunks[1:]:                      # 第一个切口之前是块 0 的开头，跳过
        p = c["start"]
        if in_ranges(p, fences):
            buckets["代码块内"].append(p)
        elif in_ranges(p, tables):
            buckets["表格行内"].append(p)
        elif text[p - 1] == "\n":
            buckets["行首(切在换行处)"].append(p)
        else:
            buckets["句中(切在行中间)"].append(p)

    n_cuts = len(chunks) - 1
    for name, hits in buckets.items():
        pct = len(hits) / n_cuts * 100
        print(f"  {name:<18} {len(hits):>4} 个  ({pct:5.1f}%)")

    print()
    print("  行首那几个是**运气**，不是设计 —— 500 是个整数，文档的行长不是，")
    print("  所以切口落在哪儿完全取决于前面所有行的累计长度，纯巧合。")

    # ---------- 4. 最硬的那几个 ----------
    print()
    print("=" * 66)
    print("4. 最硬的切口（切口落在代码块/表格内部）")
    print("=" * 66)

    hard = [c for c in chunks[1:] if in_ranges(c["start"], fences) or in_ranges(c["start"], tables)]
    if not hard:
        print("  一个都没有。")
    else:
        for c in hard:
            p = c["start"]
            kind = "代码块" if in_ranges(p, fences) else "表格"
            print(f"\n  ── 切口 #{c['i']}  偏移 {p:,}  落在【{kind}】内部 ──")
            print(f"     上一块结尾 ...{text[p - 46:p]!r}")
            print(f"     下一块开头 {text[p:p + 46]!r}...")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
