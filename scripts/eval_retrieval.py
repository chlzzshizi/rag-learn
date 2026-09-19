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

# ── 10 道题 ───────────────────────────────────────────────────────────────
# 锚点必须是**句子级**的，不能是关键词。第一版我用了 'DECR' / '分叉点' 这种词，
# 结果 'DECR' 落在 8 块里 —— 量出来的是词频不是答案位置，5 道"集中题"全被
# 算成散落。锚点收到只在该答案处出现的整句后，落点才是真的。
CASES = [
    # ── 集中：一份答案就够 ──
    dict(type="集中", mode="any",
         q="订单状态码 1-7 分别叫什么？网单和门店单在哪个状态分叉？",
         groups=[["| 4 | 待出厂 | 洗涤完成，等待下一步 |",
                  "| 6 | 派送中 | 快递配送中 | 仅网单 |"]]),

    dict(type="集中", mode="any",
         q="精洗的价格是怎么定出来的？有没有例外？",
         groups=[["精洗价不用手动填，后端根据", "精洗价直接手动填写"]]),

    dict(type="集中", mode="any",
         q="抢券为什么不用 MySQL 直接扣库存？",
         groups=[["MySQL 行锁能解决超卖", "Redis 单线程 + 原子 DECR"],       # §11.3
                 ["行锁能防超卖但要排队等锁", "Redis 单线程 + DECR 原子操作"]]),  # §11.8

    dict(type="集中", mode="any",
         q="快递单号有长度上限吗？多少？按什么口径数？",
         groups=[["不超过 50 个字符（按 **code point** 数",
                  "按 **code point** 数，不是 `length()`"]]),

    dict(type="集中", mode="any",
         q="订单表为什么不用外键？引用完整性靠什么保证？",
         groups=[["逻辑关系」，不是外键约束", "引用完整性由代码负责"]]),

    # ── 散落：每一片都要 ──
    dict(type="散落", mode="all",
         q="折扣券从发放到核销：谁能创建、谁能用、核销记录写在哪？",
         groups=[["店长创建折扣券", "管理员 → 403「管理员不参与发券」"],   # §11.6 谁能建
                 ["门店单和网单都能用**（2026-09-12"],                    # §5.8 谁能用
                 ["UPDATE coupon_grabs SET used = 1, used_time = NOW()"]]),  # §5.8 核销

    dict(type="散落", mode="all",
         q="顾客有哪两种来源？系统怎么区分「设过密码」和「没设过密码」？",
         groups=[["hasPassword()：门店单顾客从没设过密码"],               # §三 域
                 ["**NULL**（线上确实不知道他是谁）"],                     # §4.3 表
                 ["注册即登录，**只要手机号+密码**"]]),                     # §11.6

    dict(type="散落", mode="all",
         q="一个 token 会因为哪些原因失效？",
         groups=[["登出时 Token 加入 Redis 黑名单"],                      # §6.2
                 ["auth:staff:invalidAfter:<staffId> = 当前毫秒"],         # §6.2
                 ["canLogin()：只有明确 status=1 才放行"]]),               # §三 域

    dict(type="散落", mode="all",
         q="项目里 Redis 存了哪些东西？各自的键名是什么？",
         groups=[["coupon:stock:{couponId}", "coupon:grabbed:{couponId}"],        # §11.5
                 ["blacklist:token:", "auth:staff:invalidAfter:<staffId> = 当前毫秒"],  # §6.2
                 ["Redis 库存只是 MySQL 的**加速副本**"]]),                # §11.10

    dict(type="散落", mode="all",
         q="「管理员不参与业务」具体体现在哪几件事上？",
         groups=[["不参与订单操作，也不能改价"],                            # §4.2
                 ["管理员 403 只剩三处"],                                  # §4.2
                 ["管理员 → 403「管理员不参与发券」"]]),                     # §11.6
]


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

    for case in CASES:
        print(f"\n[{case['type']}/{case['mode']}] {case['q']}")
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
    for case in CASES:
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

    # ---- 汇总 ----
    print("\n" + "=" * 76)
    print(f"汇总（k={k}）")
    print("=" * 76)
    print(f"{'类型':<6}{'题数':>5}{'答得出':>8}{'命中率':>9}{'答案最少需块':>14}")
    for t in ("集中", "散落"):
        rs = [r for r in results if r["case"]["type"] == t]
        n_ok = sum(1 for r in rs if r["ok"])
        avg_need = sum(r["need"] for r in rs) / len(rs)
        print(f"{t:<6}{len(rs):>5}{n_ok:>8}{n_ok / len(rs) * 100:>8.0f}%{avg_need:>13.1f}")
    n_ok = sum(1 for r in results if r["ok"])
    avg_need = sum(r["need"] for r in results) / len(results)
    print(f"{'合计':<6}{len(results):>5}{n_ok:>8}{n_ok / len(results) * 100:>8.0f}%"
          f"{avg_need:>13.1f}")

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
    corpus = rag.CORPUS.read_text(encoding="utf-8")
    print(f"语料 {rag.CORPUS.name}：{len(corpus):,} 字符 → 索引 {len(chunks)} 块"
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
