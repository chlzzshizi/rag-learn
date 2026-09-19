#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切口体检：不靠肉眼，把每个切口"落在什么位置"分类统计出来。

原来这里 `from step1_chunk import ...`，2026-09-19 手写版被删掉后就跑不起来了。
现在自包含，并且能查两种切法：

    fixed      固定长度切（手写版的做法），每 500 字符一刀，不看内容
    recursive  LangChain 的 RecursiveCharacterTextSplitter，从大到小找边界

同样参数下对比这两个，就是"用 LangChain 到底值不值"的硬证据。

用法：

    .venv/Scripts/python.exe scripts/inspect_cuts.py                # 两种都查
    .venv/Scripts/python.exe scripts/inspect_cuts.py --splitter fixed
    .venv/Scripts/python.exe scripts/inspect_cuts.py --detail       # 列出切坏的
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(r"C:\Users\33705\Desktop\项目\2026-07-04-yunxi-redesign.md")
DEFAULT_SIZE = 500


def load_text(path: Path) -> str:
    if not path.exists():
        sys.exit(f"找不到语料：{path}")
    return path.read_text(encoding="utf-8")


def split_fixed(text: str, size: int) -> list[str]:
    """手写版的做法：range(0, len, size) 硬切，不看内容。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


def split_recursive(text: str, size: int) -> list[str]:
    """LangChain 的做法：按 separators 从大到小找落脚点，找不到才硬切。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=0,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_text(text)


def find_fenced_ranges(text: str) -> list[tuple[int, int]]:
    """所有围栏代码块（``` 包起来的）的字符区间。"""
    ranges, pos, open_at = [], 0, None
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            if open_at is None:
                open_at = pos
            else:
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


def offsets_of(chunks: list[str]) -> list[int]:
    """每块的起始偏移（第 0 块是 0，不算切口）。"""
    out, pos = [], 0
    for c in chunks:
        out.append(pos)
        pos += len(c)
    return out


def analyze(label: str, text: str, chunks: list[str], detail: bool) -> dict:
    fences = find_fenced_ranges(text)
    tables = find_table_ranges(text)
    starts = offsets_of(chunks)[1:]              # 跳过第 0 块

    buckets = {"代码块内": [], "表格行内": [], "行首": [], "句中": []}
    for p in starts:
        if in_ranges(p, fences):
            buckets["代码块内"].append(p)
        elif in_ranges(p, tables):
            buckets["表格行内"].append(p)
        elif p > 0 and text[p - 1] == "\n":
            buckets["行首"].append(p)
        else:
            buckets["句中"].append(p)

    n = len(starts)
    lengths = [len(c) for c in chunks]
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print("=" * 70)
    print(f"  {len(chunks)} 块，{n} 个切口；块长 {min(lengths)} ~ {max(lengths)}，"
          f"平均 {sum(lengths) / len(lengths):.0f}")
    print()
    for name in ("行首", "句中", "代码块内", "表格行内"):
        hits = buckets[name]
        print(f"    {name:<10} {len(hits):>4} 个  ({len(hits) / n * 100:5.1f}%)")

    bad = len(buckets["代码块内"]) + len(buckets["表格行内"])
    print(f"\n  → 切在代码块/表格内部（硬伤）：{bad} 个，占 {bad / n * 100:.1f}%")
    print(f"  → 落在换行处（干净）：{len(buckets['行首'])} 个，"
          f"占 {len(buckets['行首']) / n * 100:.1f}%")

    if detail:
        for name in ("代码块内", "表格行内"):
            for p in buckets[name]:
                kind = "代码块" if in_ranges(p, fences) else "表格"
                print(f"\n    偏移 {p:,} 落在【{kind}】内部")
                print(f"      上一块结尾 …{text[max(0, p - 40):p]!r}")
                print(f"      下一块开头 {text[p:p + 40]!r}…")

    return {"chunks": len(chunks), "cuts": n, **{k: len(v) for k, v in buckets.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="切口体检")
    ap.add_argument("--splitter", choices=["fixed", "recursive", "both"], default="both")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    ap.add_argument("--detail", action="store_true", help="列出切在代码块/表格内部的切口")
    args = ap.parse_args()

    text = load_text(CORPUS)
    fences = find_fenced_ranges(text)
    tables = find_table_ranges(text)
    print(f"语料 {CORPUS.name}：{len(text):,} 字符")
    print(f"  {len(fences)} 个围栏代码块（{sum(b - a for a, b in fences):,} 字符）"
          f" + {len(tables)} 张表格（{sum(b - a for a, b in tables):,} 字符）"
          f" = 占全文 {(sum(b - a for a, b in fences) + sum(b - a for a, b in tables)) / len(text) * 100:.1f}%")

    results = {}
    if args.splitter in ("fixed", "both"):
        results["fixed"] = analyze(f"固定长度切块（手写版做法）  size={args.size}",
                                   text, split_fixed(text, args.size), args.detail)
    if args.splitter in ("recursive", "both"):
        results["recursive"] = analyze(f"RecursiveCharacterTextSplitter（LangChain）  size={args.size}",
                                       text, split_recursive(text, args.size), args.detail)

    if len(results) == 2:
        f, r = results["fixed"], results["recursive"]
        print(f"\n{'=' * 70}")
        print("对比")
        print("=" * 70)
        print(f"  块数        {f['chunks']:>4}  →  {r['chunks']:>4}   "
              f"({r['chunks'] - f['chunks']:+d})")
        for k in ("行首", "句中", "代码块内", "表格行内"):
            print(f"  {k:<10} {f[k]:>4}  →  {r[k]:>4}   ({r[k] - f[k]:+d})")
        fb = f["代码块内"] + f["表格行内"]
        rb = r["代码块内"] + r["表格行内"]
        print(f"\n  硬伤占比    {fb / f['cuts'] * 100:>5.1f}%  →  {rb / r['cuts'] * 100:>5.1f}%")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
