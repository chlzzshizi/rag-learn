#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 5 步：检索评测。10 道题，5 道答案集中、5 道答案散落。

**只测检索，不调 LLM。** 两个理由：
  1. 切块坏了，坏在"召回的块里有没有那句话"，那是检索的事；生成会把这个变量搅浑
  2. 不花钱、不用网络、几秒出结果

## 标准答案怎么定

每题给若干**组**锚点，每组是一份**完整答案**：

    mode="any"  集中题：任意一组凑齐 → 答得出
    mode="all"  散落题：每一组都要凑齐 → 才答得出

**为什么要有"多组"**：这份文档把同一件事在好几个地方各说一遍 ——
定义处一次、代码处一次、§11.8「面试四连问」再逐字答一次。
第一版我给每题只记了一组锚点，结果检索召回了另一处的等价答案，被我判成"噪音"。
那是**标准答案不认账**，不是检索失败。所以能让多组就得让。

**span 是"最少几块能凑齐一组"，不是"含锚点的所有块"。**
用精确集合覆盖算（块少，直接穷举）。差别在哪：`blacklist:token:` 同时出现在
第 81 和第 106 块，但只要你手里有 81，106 就是多余的 —— 按"含锚点的块"算会虚增。

## 三层自检（防止把锚点的错算到检索头上）

    锚点打错字   → 语料里根本没有 → 我写错了
    锚点跨块     → 语料里有，但没有一块完整包含 → **切块把它劈开了**
    锚点不唯一   → 落在多块 → 正常，但要如实报出来

用法：

    .venv/Scripts/python.exe scripts/eval_retrieval.py
    .venv/Scripts/python.exe scripts/eval_retrieval.py -k 4 --verbose
    .venv/Scripts/python.exe scripts/eval_retrieval.py --sweep
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "scripts"))
from eval_cases import BUG_CASES, CASES  # noqa: E402  （题库和答案级评测共用一份，别分家）


def all_cases() -> list[dict]:
    """两份题库合起来，都打上「题库」标签。

    **新题必须过 audit() 这一关** —— 不审就上，等于把锚点自己的错
    （打错字、跨块被劈开）算到检索头上。bug 文档那 6 道是新加的，
    锚点虽然已逐字验过存在，但"有没有被切块劈开"只有这里的 audit 能查。

    汇总也**必须分题库报**：bug 文档对设计文档那 10 道题含零条标准答案、
    却含大量同词（`403` 70 次、`token` 62 次）—— 合在一起报，
    读到的差值里混着"候选池从 214 涨到 547"这个纯干扰效应。
    """
    for c in CASES:
        c.setdefault("题库", "设计文档")
    for c in BUG_CASES:
        c.setdefault("题库", "bug文档")
    return CASES + BUG_CASES


def chunks_from_store(store) -> dict[int, str]:
    out = {}
    for _vec_idx, doc_id in store.index_to_docstore_id.items():
        d = store.docstore.search(doc_id)
        out[d.metadata["i"]] = d.page_content
    return out


def min_cover(anchor_chunks: list[list[int]]) -> list[int]:
    """最少几块能凑齐这组锚点。块少，直接穷举精确解。"""
    universe = sorted({i for s in anchor_chunks for i in s})
    for r in range(1, len(universe) + 1):
        for combo in combinations(universe, r):
            pick = set(combo)
            if all(pick & set(s) for s in anchor_chunks):
                return list(combo)
    return universe


def audit(corpus: str, chunks: dict[int, str]) -> list[tuple]:
    print("\n" + "=" * 76)
    print("标准答案自检")
    print("=" * 76)
    problems = []

    for case in all_cases():
        print(f"\n[{case['题库']}/{case['type']}/{case['mode']}] {case['q']}")
        spans = []
        for gi, group in enumerate(case["groups"], 1):
            per_anchor = []
            for a in group:
                if a not in corpus:
                    print(f"   ❌ {a!r}  ← 语料里根本没有，**锚点打错了**")
                    problems.append(("typo", case["q"], a))
                    per_anchor.append([])
                    continue
                hits = sorted(i for i, t in chunks.items() if a in t)
                if not hits:
                    print(f"   💥 {a!r}  ← 没有一块完整包含，**被切块劈开了**")
                    problems.append(("split", case["q"], a))
                else:
                    print(f"   ✓ 第{hits if len(hits) < 4 else str(hits[:3]) + '…'}块"
                          f"  {a[:38]!r}")
                per_anchor.append(hits)
            cover = min_cover(per_anchor) if per_anchor else []
            spans.append(cover)
            print(f"      → 第 {gi} 组最少要 {len(cover)} 块：{cover}")

        if case["mode"] == "any":
            best = min(spans, key=len) if spans else []
            need = len(best)
        else:
            need = sum(len(s) for s in spans)
        case["_span"] = sorted({i for s in spans for i in s})
        case["_need"] = need
        print(f"   ⇒ 答得出最少需要 k ≥ {need}"
              + ("（任一等价答案组）" if case["mode"] == "any" else "（每组都要）"))

    if problems:
        print("\n" + "!" * 76)
        print("锚点有问题，先修，别把账算到检索头上：")
        for kind, q, a in problems:
            print(f"   [{kind}] {a!r}  ← {q}")
        print("!" * 76)
    return problems


def evaluate(store, chunks: dict[int, str], k: int) -> list[dict]:
    out = []
    for case in all_cases():
        got = {d.metadata["i"] for d, _s in store.similarity_search_with_score(case["q"], k=k)}
        got &= set(chunks)

        gstat = []
        for group in case["groups"]:
            hit = [a for a in group if any(a in chunks[i] for i in got)]
            gstat.append((len(hit), len(group)))

        if case["mode"] == "any":
            ok = any(h == n for h, n in gstat)
        else:
            ok = all(h == n for h, n in gstat)

        out.append(dict(case=case, k=k, ok=ok, gstat=gstat, got=sorted(got),
                        need=case["_need"], span=case["_span"]))
    return out


def report(results: list[dict], k: int, chunks: dict, verbose: bool) -> None:
    print("\n" + "=" * 76)
    print(f"检索评测  k={k}")
    print("=" * 76)

    for r in results:
        c = r["case"]
        mark = "✅" if r["ok"] else ("🟡" if any(h for h, _ in r["gstat"]) else "❌")
        short = "／".join(f"{h}/{n}" for h, n in r["gstat"])
        print(f"\n{mark} [{c['type']}] {c['q']}")
        print(f"   答案最少要 {r['need']} 块"
              + ("" if r["need"] > k else f"，k={k} 够装")
              + (f"  ⚠ **k={k} 装不下，结构上做不到**" if r["need"] > k else ""))
        print(f"   各组锚点命中：{short}   （{c['mode']} 语义）")
        print(f"   召回：{r['got']}")
        if verbose:
            for i in r["got"]:
                tag = "★答案区" if i in r["span"] else "  噪音"
                print(f"      {tag}  第 {i:>3} 块  {chunks[i][:48]}".replace("\n", "⏎"))

    # ---- 汇总：**分题库报，不给跨库合计** ----
    #
    # 为什么不合计：bug 文档那 332 块对设计文档的 10 道题含**零条标准答案**，
    # 却是主题相近的干扰块（`403` 70 次、`token` 62 次）。候选池从 214 涨到 547
    # 本身就是对检索的惩罚。合在一起给一个"总命中率"，读到的差值里
    # 混着这个纯干扰效应 —— 而那跟"检索好不好"不是一回事。
    print("\n" + "=" * 76)
    print(f"汇总（k={k}）")
    print("=" * 76)
    print(f"{'题库':<8}{'类型':<6}{'题数':>5}{'答得出':>8}{'命中率':>9}{'答案最少需块':>14}")
    for 题库 in ("设计文档", "bug文档"):
        for t in ("集中", "散落"):
            rs = [r for r in results
                  if r["case"]["题库"] == 题库 and r["case"]["type"] == t]
            if not rs:
                continue
            n_ok = sum(1 for r in rs if r["ok"])
            avg_need = sum(r["need"] for r in rs) / len(rs)
            print(f"{题库:<8}{t:<6}{len(rs):>5}{n_ok:>8}"
                  f"{n_ok / len(rs) * 100:>8.0f}%{avg_need:>13.1f}")

    print("\n  归因 —— 没答得出的题，卡在哪：")
    for r in results:
        if r["ok"]:
            continue
        if r["need"] > k:
            why = f"答案要 {r['need']} 块 > k={k}，位置不够装（**结构上做不到**）"
        elif any(h for h, _ in r["gstat"]):
            why = f"只拿到部分答案（{r['gstat']}），**排序没把剩下的排进前 {k}**"
        else:
            why = f"一片答案都没召到，**排序完全没排上**"
        print(f"     [{r['case']['type']}] {r['case']['q'][:26]}…  ← {why}")


def main() -> None:
    ap = argparse.ArgumentParser(description="第 5 步：检索评测")
    ap.add_argument("-k", type=int, default=4, help="召回几块（默认 4，跟 rag.py 一致）")
    ap.add_argument("--sweep", action="store_true", help="额外跑 k=1/2/8/16 看趋势")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--instruct", action="store_true",
                    help="查询端加 Qwen3-Embedding 的 instruction 前缀（文档端不加）")
    args = ap.parse_args()

    import rag

    emb = rag.get_embeddings()
    if args.instruct:
        # Qwen3-Embedding 的查询端 instruction。文档端**不加** —— 索引本身不用重建，
        # 只换查询向量，这也是这个前缀容易漏掉的原因：漏了照样能跑，只是更差。
        emb.query_encode_kwargs = {
            "prompt": "Instruct: Given a web search query, retrieve relevant "
                      "passages that answer the query\nQuery: "
        }
        print("查询端已加 instruction 前缀")
    store = rag.load_index(emb)
    chunks = chunks_from_store(store)
    parts = list(rag.corpus_texts())
    corpus = "\n\n".join(t for _n, t in parts)
    print(f"语料 {' + '.join(n for n, _t in parts)}：{len(corpus):,} 字符"
          f" → 索引 {len(chunks)} 块"
          f"（chunk_size={rag.CHUNK_SIZE}, overlap={rag.CHUNK_OVERLAP}）")

    problems = audit(corpus, chunks)

    for k in sorted({args.k} | ({1, 2, 8, 16} if args.sweep else set())):
        report(evaluate(store, chunks, k), k, chunks, args.verbose)

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
