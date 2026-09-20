#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 6 步的「逐题胜负」判决器 —— 把那个 1:11:4 变成可复现的。

## 为什么要有这个脚本

第 6 步的头条数字（**RAG 胜 1、长上下文胜 11、平 4**）**当时是人工读 report 数出来的**，
仓库里没有任何脚本能重算它。后果很直接：**换一版检索重跑，新旧数字没法用同一条规则比。**

所以这里把判据写死成代码。判据只有一条，是从 README 抄下来的：

    每题每臂   delta = 3-gram 覆盖 − 跨题蒙中地板
    两臂相比   delta 差 > 10 分才算「胜」，否则「平」

**不看均值**：单题摆幅实测最大 62 分，n=5 时一题翻面就动均值 20 分。

## 为什么必须减掉地板

长上下文臂话多（平均 1083 字 vs RAG 的 276 字），**话多本身就更容易蒙中锚点**。
`gram_null` 就是拿本题答案去对**别题**的锚点，量出「纯靠话多能得多少」。
不减它，等于在比谁写得多。

它的第一次实际使用（2026-09-20 晚）就抓到了上面那个 `i+1` 地板问题 —— 见
`recompute_null()` 的文档。

用法：

    .venv/Scripts/python.exe scripts/eval_verdict.py                    # 最新一次 dump
    .venv/Scripts/python.exe scripts/eval_verdict.py data/answers_xxx.jsonl
    .venv/Scripts/python.exe scripts/eval_verdict.py --baseline long --arms bm25,hybrid

臂名是 `dense` / `bm25` / `hybrid` / `long`。**旧的 `rag` 就是 `dense`** ——
第 7 步之前只有稠密一路，那时 `arm` 字段写的是 `"rag"`，同一个东西。
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 「胜」的门槛，单位是百分点。README 写的是「> 10 分」。
WIN_MARGIN = 0.10
CELLS = [("设计文档", "集中"), ("设计文档", "散落"), ("bug文档", "集中"), ("bug文档", "散落")]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recompute_null(rows: list[dict]) -> dict[tuple[str, str], float]:
    """按**当前**规则重算跨题蒙中地板，**不用 dump 里存的那个**。

    ⚠️ 这一条是 2026-09-20 查出来的必要修复。当时 dump 里存的地板是**旧的
    `i+1` 规则**（拿相邻下一题当对照），而 `eval_answer.py` 后来改成了「取半程
    之外的题」—— **改完没有重跑**，所以仓库里所有 dump 的 `gram_null` 都是旧规则的。

    实测差别很大：「支付报 500」那题的 long 臂地板，旧规则 **0.922**（相邻的 Bug 2
    原文被抄进去了），半程规则只有 **0.065**。差 0.86 —— 足以把一题的胜负判反。

    地板**能离线重算**（答案和锚点都在），所以这里现算，不信任存货。
    返回 {题: 地板}（地板是**每题一个**，不随臂变 —— 它是"拿本题答案去对别题锚点"，
    跟臂无关，dump 里按行存只是冗余）。
    """
    import sys
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from eval_answer import judge_group, _mean
    from eval_cases import CASES, BUG_CASES

    cases = [dict(c) for c in CASES + BUG_CASES]
    byq = {c["q"]: c for c in cases}
    order = [c["q"] for c in cases]
    n = len(order)
    out: dict[tuple[str, str], float] = {}
    for r in rows:
        q = r["q"]
        if q not in byq:
            continue
        try:
            i = order.index(q)
        except ValueError:
            continue
        other = byq[order[(i + n // 2) % n]]          # 半程之外，不是相邻
        per = [judge_group(r["answer"], g) for g in other["groups"]]
        out[(q, r["arm"])] = _mean([p["gram"] for p in per]) or 0.0
    return out


def deltas(rows: list[dict], use_stored: bool = False) -> dict[tuple[str, str], float]:
    """(题, 臂) → 减掉地板之后的 3-gram 覆盖。"""
    nulls = {} if use_stored else recompute_null(rows)
    out = {}
    for r in rows:
        g = r["score"].get("gram")
        if g is None:
            continue
        n_ = r.get("gram_null") if use_stored else nulls.get((r["q"], r["arm"]))
        if n_ is None:
            continue
        out[(r["q"], r["arm"])] = g - n_
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="第 6 步逐题胜负判决")
    ap.add_argument("dump", nargs="?", help="answers_*.jsonl；不给就用最新的一个")
    ap.add_argument("--baseline", default="long", help="跟谁比（默认 long）")
    ap.add_argument("--arms", help="只比这几臂（逗号分隔）；默认除 baseline 外全部")
    ap.add_argument("--margin", type=float, default=WIN_MARGIN,
                    help=f"「胜」的门槛，单位百分点（默认 {WIN_MARGIN * 100:.0f}）")
    ap.add_argument("--raw", action="store_true",
                    help="不减地板，直接用裸 3-gram（**这是错的做法**，只为复现旧数字留着）")
    ap.add_argument("--stored-null", action="store_true",
                    help="用地板在 dump 里存的值，不重算（旧 dump 存的是已废弃的 i+1 规则）")
    args = ap.parse_args()

    path = Path(args.dump) if args.dump else max(
        (Path(p) for p in glob.glob(str(ROOT / "data" / "answers_*.jsonl"))),
        key=lambda p: p.stat().st_mtime, default=None)
    if path is None or not path.exists():
        sys.exit("找不到 dump。先跑 scripts/eval_answer.py 生成一个。")

    rows = load(path)
    if args.raw:
        d = {(r["q"], r["arm"]): r["score"]["gram"]
             for r in rows if r["score"].get("gram") is not None}
    else:
        d = deltas(rows, use_stored=args.stored_null)

    arms = sorted({r["arm"] for r in rows})
    if args.arms:
        others = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        others = [a for a in arms if a != args.baseline]
    if args.baseline not in arms:
        sys.exit(f"baseline={args.baseline!r} 不在 dump 里（有 {arms}）")

    meta = {r["q"]: r for r in rows}
    qs = [q for q in dict.fromkeys(r["q"] for r in rows) if (q, args.baseline) in d]
    print(f"\n判决依据：{path.name}")
    print(f"  判据 delta = 3-gram − 跨题蒙中地板，两臂差 > {args.margin * 100:.0f} 分才算「胜」")
    print(f"  基准臂 = {args.baseline}；对照臂 = {others}")

    for arm in others:
        qs_a = [q for q in qs if (q, arm) in d]
        if not qs_a:
            continue
        print("\n" + "=" * 78)
        print(f"{arm}  VS  {args.baseline}   （{len(qs_a)} 题）")
        print("=" * 78)
        print(f"{'题库':<8}{'题型':<6}{'题数':>5}{arm + ' 胜':>12}{args.baseline + ' 胜':>12}{'平':>6}")
        tot = {arm: 0, args.baseline: 0, "平": 0}
        for 题库, t in CELLS:
            sel = [q for q in qs_a
                   if meta[q]["题库"] == 题库 and meta[q]["type"] == t]
            if not sel:
                continue
            c = {arm: 0, args.baseline: 0, "平": 0}
            for q in sel:
                diff = d[(q, arm)] - d[(q, args.baseline)]
                key = arm if diff > args.margin else args.baseline if diff < -args.margin else "平"
                c[key] += 1
            for k in c:
                tot[k] += c[k]
            print(f"{题库:<8}{t:<6}{len(sel):>5}{c[arm]:>12}{c[args.baseline]:>12}{c['平']:>6}")
        n = sum(tot.values())
        print(f"{'合计':<8}{'':<6}{n:>5}{tot[arm]:>12}{tot[args.baseline]:>12}{tot['平']:>6}")

        # 均值只作参考 —— 判据不看它，这里列出来是为了显示它**没有分辨力**
        print(f"\n  （参考，不用来判胜负）均值 delta："
              f"{arm} {sum(d[(q, arm)] for q in qs_a) / len(qs_a) * 100:.1f}  vs  "
              f"{args.baseline} {sum(d[(q, args.baseline)] for q in qs_a) / len(qs_a) * 100:.1f}")
        print(f"  逐题差 > {args.margin * 100:.0f} 分的：")
        shown = 0
        for q in qs_a:
            diff = d[(q, arm)] - d[(q, args.baseline)]
            if abs(diff) <= args.margin:
                continue
            shown += 1
            who = arm if diff > 0 else args.baseline
            print(f"    {diff * 100:+6.0f} 分  {who:<8} [{meta[q]['题库']}/{meta[q]['type']}]"
                  f" {q[:36]}…")
        if not shown:
            print("    （没有 —— 全部落在门槛内）")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
