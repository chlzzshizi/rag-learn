#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""切块方案免费筛：只用 BM25，**一块都不编码**。

## 为什么能免费

`hybrid.BM25Index` 要的是 `{块号: 文本}` 这个字典，**它不碰 FAISS、不加载模型**。
所以换一种切法、重新打分，代价就是几秒的 jieba 分词。

## 为什么必须筛

切法一换，稠密索引就得重建（CPU 上几分钟）。而切法是**组合**出来的
（按不按标题切 × 合不合块 × 注不注入章节路径 × 要不要结构安全），
盲试等于拿编码时间当筛子。先把组合在 BM25 上摊开，再付一次编码钱。

## 这个筛子能分辨什么、不能分辨什么

**能**：量级差别。比如「按标题切但**不合块**」实测 4~5/16，合块 8~9/16
—— 差 3~4 题，机制也清楚（naive 切法会产出 16,588 字符的单块，
在 BM25 的长度归一化下永远赢不了）。

**不能**：±1 题。16 道题里 1 题 = 6.25%，而同一套设计跑两次就能差出 1 题来。
**所以别拿这个表去挑「最好看的那一行」**，它只能排除明显更差的组合。

用法：

    .venv/Scripts/python.exe scripts/eval_chunker_screen.py
    .venv/Scripts/python.exe scripts/eval_chunker_screen.py --size 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import chunking  # noqa: E402
import rag  # noqa: E402
from eval_cases import BUG_CASES, CASES  # noqa: E402
from hybrid import BM25Index  # noqa: E402


def all_cases() -> list[dict]:
    for c in CASES:
        c.setdefault("题库", "设计文档")
    for c in BUG_CASES:
        c.setdefault("题库", "bug文档")
    return CASES + BUG_CASES


def build(chunker: str, size: int, inject: bool, structure: bool):
    """两份语料 → ({全局块号: 文本}, 落单围栏块数)。

    **块号全局连续**，跟 `rag.build_index` 一个规矩（两份语料不各从 0 开始）。
    """
    chunks, lone, i = {}, 0, 0
    for _name, text in rag.corpus_texts():
        for p in chunking.pieces_of(text, chunker, size=size,
                                    inject=inject, structure=structure):
            chunks[i] = p.render(inject)
            lone += p.lone_fences % 2
            i += 1
    return chunks, lone


def ok_of(case: dict, got: set[int], chunks: dict[int, str]) -> bool:
    stat = []
    for g in case["groups"]:
        hit = [a for a in g if any(a in chunks[j] for j in got)]
        stat.append((len(hit), len(g)))
    return (any(h == n for h, n in stat) if case["mode"] == "any"
            else all(h == n for h, n in stat))


def screen(label: str, chunks: dict[int, str], lone: int, cases: list[dict],
           ks: list[int]) -> None:
    bm25 = BM25Index(chunks)
    sz = sorted(len(v) for v in chunks.values())
    line = (f"{label:<30} n={len(chunks):>4} med={sz[len(sz) // 2]:>4} "
            f"max={max(sz):>4} 落单围栏={lone:>3}")
    for k in ks:
        res = [(c, ok_of(c, set(bm25.search(c["q"], k=k)), chunks)) for c in cases]
        cells = [sum(x for c, x in res if c["题库"] == lib and c["type"] == t)
                 for lib in ("设计文档", "bug文档") for t in ("集中", "散落")]
        line += f"  k{k}={sum(x for _c, x in res):>2}/16{cells}"
    print(line, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="切块方案免费筛（BM25，不编码）")
    ap.add_argument("--size", type=int, default=chunking.CHUNK_SIZE)
    ap.add_argument("-k", type=int, nargs="*", default=[4, 8])
    args = ap.parse_args()

    cases = all_cases()
    print(f"语料 {len(list(rag.corpus_texts()))} 份，题目 {len(cases)} 道；"
          f"size={args.size}")
    print("分格顺序 [设计/集中, 设计/散落, bug/集中, bug/散落]\n")

    VARIANTS = [
        ("flat 旧切法（基线）", "flat", False, True),
        ("md 不注入（只有 A）", "md", False, True),
        ("md 注入（A+B）", "md", True, True),
        ("md 不注入 + 二次切分不用结构", "md", False, False),
        ("md 注入 + 二次切分不用结构", "md", True, False),
    ]
    for label, chunker, inject, structure in VARIANTS:
        chunks, lone = build(chunker, args.size, inject, structure)
        screen(label, chunks, lone, cases, args.k)

    print("\n⚠ 只能排除量级差别（3~4 题），**分辨不了 ±1 题**（1 题 = 6.25%）。")
    print("  挑方案要靠：切口硬伤率 + 落单围栏数 + 机制，别挑表里最好看的那行。")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
