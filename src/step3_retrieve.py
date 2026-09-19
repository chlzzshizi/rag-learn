#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 3 步：检索。

问题进来 → 编成向量 → 和库里 148 个向量比相似度 → 取 top-k。

核心就一行：

    scores = vectors @ q

因为第 2 步已经把库向量归一化了、这里也会把问题向量归一化，
**两个单位向量的点积就等于余弦相似度**（|a||b|cosθ，|a|=|b|=1）。
所以不需要真的去算余弦 —— 这也是第 2 步为什么死磕 float32 的原因：
范数不精确等于 1，"点积 = 余弦" 就只是近似成立。

用法：

    .venv/Scripts/python.exe src/step3_retrieve.py "订单状态有哪些状态"
    .venv/Scripts/python.exe src/step3_retrieve.py            # 跑内置的几道题
    .venv/Scripts/python.exe src/step3_retrieve.py "..." --k 3 --instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_chunk import CORPUS  # noqa: E402
from step2_encode import MODEL_DIR, QUERY_INSTRUCTION, load_model, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# README 第 3 步给的那道验收题。
DEFAULT_QUESTIONS = [
    "订单状态有哪些状态",
    "抢券的时候怎么防止超卖",
    "精洗的加价规则是怎么算的",
]


def load_index() -> tuple[list[dict], "object", dict]:
    """读回第 2 步的产物，并检查它是不是过期了。

    这个检查看着多余，其实是增量更新的地基：manifest 里存了语料的 sha256，
    语料一改这里就对不上 —— 与其拿着过期向量给出似是而非的答案，
    不如直接拒绝服务。
    """
    import numpy as np

    for name in ("chunks.jsonl", "vectors.npy", "manifest.json"):
        if not (DATA / name).exists():
            sys.exit(f"缺少 data/{name} —— 先跑：python src/step2_encode.py")

    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in
              (DATA / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    vectors = np.load(DATA / "vectors.npy")

    now_sha = sha256_file(CORPUS)
    if now_sha != manifest["source_sha256"]:
        print("⚠ 语料已经改过了，索引是旧的。")
        print(f"  索引建于 {manifest['created_at']}，语料已从 "
              f"{manifest['source_chars']:,} 字符变成 "
              f"{len(CORPUS.read_text(encoding='utf-8')):,} 字符。")
        print("  先重跑：python src/step2_encode.py\n")

    assert vectors.shape[0] == len(chunks), "向量条数和块数对不上"
    return chunks, vectors, manifest


def search(query: str, chunks: list[dict], vectors, model, k: int, instruct: bool):
    """一趟检索。返回 [(分数, chunk), ...]，按分数降序。"""
    import numpy as np

    q_text = (QUERY_INSTRUCTION + query) if instruct else query

    # 查询端也必须 float32 + normalize —— 跟建库时保持一致。
    # 两边用了不同的精度，比出来的分数就没有意义了。
    q = model.encode([q_text], batch_size=1, convert_to_numpy=True,
                     normalize_embeddings=True)[0].astype(np.float32)

    scores = vectors @ q                       # 点积 == 余弦，前提两边都是单位向量
    order = np.argsort(-scores)[:k]            # 降序取前 k
    return [(float(scores[i]), chunks[i]) for i in order]


def show(query: str, hits: list[tuple[float, dict]], k: int) -> None:
    print(f"\n{'=' * 74}")
    print(f"问：{query}")
    print("=" * 74)
    prev = None
    for rank, (score, c) in enumerate(hits, 1):
        gap = "" if prev is None else f"  (比上一名低 {prev - score:.4f})"
        prev = score
        # 块里全是换行，取前 70 字拍扁成一行预览
        preview = c["text"][:70].replace("\n", "⏎")
        print(f"  {rank}. 分数 {score:.4f}{gap}")
        print(f"     块 #{c['i']}  偏移 {c['start']:,}-{c['end']:,}  {preview}…")


def main() -> None:
    ap = argparse.ArgumentParser(description="第 3 步：检索")
    ap.add_argument("question", nargs="*", help="问题；不填就跑内置的几道")
    ap.add_argument("--k", type=int, default=5, help="取前几名（默认 5）")
    ap.add_argument("--instruct", action="store_true",
                    help="给查询加指令前缀（README 里的 v2 对比项）")
    args = ap.parse_args()

    chunks, vectors, manifest = load_index()
    print(f"索引：{len(chunks)} 块 / {manifest['embedding_dim']} 维 / "
          f"{manifest['model']} / 建于 {manifest['created_at']}")
    print(f"查询指令前缀：{'开' if args.instruct else '关'}")

    model = load_model()
    questions = args.question or DEFAULT_QUESTIONS

    for q in questions:
        show(q, search(q, chunks, vectors, model, args.k, args.instruct), args.k)

    print(f"\n{'=' * 74}")
    print("看什么：")
    print("  * 第一名是不是真的那一节？不是的话，先回去翻第 1 步的切口 ——")
    print("    90% 的检索不中是切块切烂了，不是检索代码错了。")
    print("  * 名次之间的分数落差。前几名咬得很紧说明它们其实在讲同一件事，")
    print("    或者说明块切得太碎、一个主题散在好几块里。")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
