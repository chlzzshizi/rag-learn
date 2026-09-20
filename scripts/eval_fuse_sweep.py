#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""融合参数敏感性：三路对照那个结论，对 RRF 的两个旋钮有多敏感？

## 为什么要跑这个

`eval_retrieval.py --retriever all` 量出 **BM25 单路比等权融合还好**。
但那是钉死在 k=60、等权、深度 20 下的一个点。**万一 k=60 是巧合呢？**
在 16 道题上调参数是最容易骗自己的地方，所以这里把整个格子摊开看，
而不是挑一个最好看的报出去。

## 网格

    权重 (dense, bm25) ∈ {(0,1) 只用BM25, (1,1) 等权, (1,3), (1,10), (1,0) 只用dense}
    RRF 常数 k         ∈ {1, 5, 20, 60}

`(0,1)` / `(1,0)` 是网格的两个端点，也就是两条单臂 —— 它们给出"融合到底
有没有超过最好的那条单臂"。

**每题的候选只编码一次**（深度 50），之后所有格子都在这批候选上重排，
所以整个网格是一次编码的代价。

用法：

    .venv/Scripts/python.exe scripts/eval_fuse_sweep.py
    .venv/Scripts/python.exe scripts/eval_fuse_sweep.py --depth 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cases import BUG_CASES, CASES  # noqa: E402
from hybrid import rrf_fuse  # noqa: E402

WEIGHT_GRID = [(0, 1), (1, 1), (1, 3), (1, 10), (1, 0)]
K_GRID = [1, 5, 20, 60]
WEIGHT_LABEL = {(0, 1): "(0,1) 只用BM25", (1, 1): "(1,1) 等权",
                (1, 3): "(1,3)", (1, 10): "(1,10)", (1, 0): "(1,0) 只用dense"}


def all_cases() -> list[dict]:
    for c in CASES:
        c.setdefault("题库", "设计文档")
    for c in BUG_CASES:
        c.setdefault("题库", "bug文档")
    return CASES + BUG_CASES


def ok_of(case: dict, got: set[int], chunks: dict[int, str]) -> bool:
    stat = []
    for g in case["groups"]:
        hit = [a for a in g if any(a in chunks[i] for i in got)]
        stat.append((len(hit), len(g)))
    return (any(h == n for h, n in stat) if case["mode"] == "any"
            else all(h == n for h, n in stat))


def main() -> None:
    ap = argparse.ArgumentParser(description="融合参数敏感性")
    ap.add_argument("--depth", type=int, default=50,
                    help="每路取多少候选后融合（默认 50）")
    ap.add_argument("-k", type=int, default=4, help="最终要几块（默认 4）")
    args = ap.parse_args()

    import rag
    from hybrid import BM25Index

    emb = rag.get_embeddings()
    store = rag.load_index(emb)
    chunks = {}
    for _vi, did in store.index_to_docstore_id.items():
        d = store.docstore.search(did)
        chunks[d.metadata["i"]] = d.page_content

    bm25 = BM25Index(chunks)
    cases = all_cases()

    print(f"编码 {len(cases)} 题 × 2 路（各取 {args.depth} 深度）…", flush=True)
    arms = []
    for c in cases:
        dense = [d.metadata["i"] for d, _s in
                 store.similarity_search_with_score(c["q"], k=args.depth)]
        arms.append((dense, bm25.search(c["q"], k=args.depth)))

    # ---- 格子 ----
    rows = []
    for w in WEIGHT_GRID:
        for kk in K_GRID:
            if w == (0, 1) and kk != K_GRID[0]:
                continue    # 单臂与 k 无关，只算一次
            if w == (1, 0) and kk != K_GRID[0]:
                continue
            res = []
            for c, (dense, sparse) in zip(cases, arms):
                if w == (0, 1):
                    got = set(sparse[:args.k])
                elif w == (1, 0):
                    got = set(dense[:args.k])
                else:
                    got = set(rrf_fuse([dense, sparse], k=kk,
                                       weights=list(w))[:args.k])
                res.append(ok_of(c, got, chunks))
            rows.append((w, kk, res))

    # ---- 打印 ----
    print("\n" + "=" * 78)
    print(f"融合参数敏感性（{len(cases)} 题；每路候选深度 {args.depth}，"
          f"融合后截到 {args.k} 块）")
    print("=" * 78)

    def cell(res, 题库, t):
        idx = [i for i, c in enumerate(cases)
               if c["题库"] == 题库 and c["type"] == t]
        if not idx:
            return "  —  "
        return f"{sum(res[i] for i in idx) / len(idx) * 100:>4.0f}%"

    cols = [("设计文档", "集中"), ("设计文档", "散落"),
            ("bug文档", "集中"), ("bug文档", "散落")]
    head = f"{'权重 (dense,bm25)':<20}{'RRF k':>6}  " + "".join(
        f"{题库[-2:]}/{t:>2}  " for 题库, t in cols) + " 合计"
    print(head)
    print("-" * len(head))
    for w, kk, res in rows:
        tag = WEIGHT_LABEL[w] if w in ((0, 1), (1, 0)) else WEIGHT_LABEL[w]
        ktag = "  —" if w in ((0, 1), (1, 0)) else f"{kk:>4}"
        line = f"{tag:<20}{ktag:>6}  " + "".join(
            f"{cell(res, q, t):>8}" for q, t in cols)
        line += f"{sum(res) / len(res) * 100:>5.0f}%"
        print(line)

    print("\n  （单臂只列一行 —— 它跟 RRF 的 k 无关）")
    print(f"  最好的一条单臂就是对照线：融合必须要**超过它**才算有用。")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
