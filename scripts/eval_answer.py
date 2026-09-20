#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 6 步：答案级评测 —— RAG vs 长上下文，同语料、同题、同模型、同 prompt。

    .venv/Scripts/python.exe scripts/eval_answer.py                 # 跑两臂
    .venv/Scripts/python.exe scripts/eval_answer.py --arms long     # 只跑一臂
    .venv/Scripts/python.exe scripts/eval_answer.py -k 4 --limit 2  # 先小样试试

## 跟第 5 步的区别

第 5 步评的是**检索**（那句答案有没有被召回的块盖住），不调 LLM。
这一步评的是**答案**（模型答对没有）。两个不同的东西，
所以别拿第 5 步的数跟这里的数直接对。

## 两臂的唯一差别是「资料」

    两臂共用 `longctx.build_messages()` —— 指令和拼装方式只有那一处定义
    RAG 臂：资料 = 检索到的 k 块，形如 `【第 N 块】\\n<500字片段>`
    长上下文臂：资料 = 整个语料（带 `# 文件：` 头的完整 markdown）

**「两臂 prompt 逐字节相同」是结构上保证的，不靠人记得同步改两处。**

⚠️ 但**资料的「形式」也变了**，不只是内容：RAG 臂拿到的是被切碎的片段，
标题和上下文被切掉、句子被劈开；长上下文臂拿到的是完整的 markdown。
这是**呈现方式上的劣势**，不是检索能力的差距。写结论时必须说清，
不能把"切块导致读不懂"算成"检索没用"。

## 判分：五个指标 + 三个对照数。⚠️ 它们**不是**相互抵消的

    strict       锚点整句**逐字**出现在答案里
    strict_norm  去标点空白后逐字出现（markdown 重锚点的唯一出路）
    strict_fuzzy 容许 1 个字的插入/删除/替换后出现
    ident        正则抽出的标识符/数字 命中率
    gram         锚点字符 3-gram 覆盖比例

### ⚠️⚠️ 最重要的一条：`strict` 量的不是「答对没有」，是「有没有照着原文抄」

这是**实测**撞出来的，不是推演 —— 「精洗价怎么定」那题，RAG 臂答得完全正确：

    锚点  「精洗价直接手动填写」（语料原文）
    答案  「精洗价**就**直接手动填写」   ← 只多一个「就」
    → strict 0%，strict_norm 0%

而长上下文臂拿 **100%**，因为它逐字复述了原文（还带引号和 §编号）。

**手里原文多的那一臂天然占便宜**：RAG 臂上下文里只有 ≤4 块（约 2K 字），
长上下文臂有 198K 字，复述就白拿分。所以：

  * **别拿 `strict` 当主指标** —— 它是抄袭探测器，方向系统性偏向长上下文臂
  * `strict_fuzzy` 的存在就是为了让"答对但改了字"不至于被判 0
  * **主指标看 `gram`**（下面两个对照数是用来给它挤水分的）

另外两条同向偏差：

  * `gram` 分母固定（锚点的 3-gram 数），**答案越长越容易蒙中**
  * `ident` 量的是"答案里有没有出现语料里存在的 token"，**看到的资料越多越容易撞上**；
    而且这份语料一大半锚点是纯中文，**抽不出标识符**，该指标大面积作废（报告里标「—」）

所以**「长上下文赢」这个结果自带三份同向偏差**，不能当结论直接读。
抵消它们的是下面两个零假设和拒答统计：

    copy_ratio   答案的 3-gram 有多大比例能在语料里找到 → 量化"复述"
    gram_null    拿第 i 题的答案去对第 i+1 题的锚点 → 蒙中的地板，随话多而涨
    refusal      「资料里没有提到」的出现次数 → RAG 的"错"可能是**有原则的拒答**

**同时报答案长度** —— 不报的话"话多 = 蒙中多"会被当成"答得好"。

⚠️ 这些指标是**排序工具，不是真值**。真值在落盘的原始答案里，
**人读一遍比任何指标都可靠**。跑完请至少抽读几条。

## 两个实测踩到的坑（写在这里防复发）

1. **纯正则抽标识符会废掉一半锚点**：10 个锚点里 5 个抽出空集
   （`| 4 | 待出厂 | 洗涤完成 |`、`不参与订单操作，也不能改价` …全是中文），
   而且 `登出时 Token 加入 Redis 黑名单` 只抽出 `['Token','Redis']` ——
   **漏掉「黑名单」，那才是关键**。所以不能只用这一个指标。
2. **3-gram 有跨词碎片**（`厂洗涤`、`填后端`），永远不会被命中。
   噪声来自锚点本身、两臂分母相同，压的是绝对值不是差值 —— 但仍要用答案长度校一下。

## ⚠️ 一个必须自己防住的统计假象

**语料翻倍本身就是对 RAG 臂的惩罚，跟上下文长度无关。**
bug 文档对原来 10 道设计文档的题含**零条标准答案**，却含大量同词
（`403` 出现 70 次、`token` 62 次、`Redis` 17 次）。候选池从 214 块涨到 547 块，
多出来的 332 块是**主题相近的干扰块** —— 对词面级检索器是最坏情况。
所以本报告**分「设计文档 10 题 / bug 文档 6 题」两块报，绝不给一个合计头条数字**，
并额外报 RAG 臂的**干扰块占比**（top-k 里有几块来自 bug 文档）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cases import BUG_CASES, CASES  # noqa: E402

DATA = ROOT / "data"

# ── 判分用的规则（写死在这里，人和机器都看得见）──────────────────────────

# 标识符 / 数字 / 反引号内的东西。**这是规则，不是判断** ——
# 换谁来跑都抽出同一批 token。
# 已知弱点：`\d+` 会把 403 / 2026 / 50 这种高频数字也算进来，
# 所以这个指标更接近"有没有聊到这个话题"，不等于"答没答对"。见 gram_null。
IDENT_RE = re.compile(r"`([^`]+)`|[A-Za-z_][A-Za-z0-9_]{1,}|\d+")
# 判分前把标点和空白去掉，免得 `**NULL**` 和 `NULL` 因为星号对不上
PUNCT_RE = re.compile(r"[\s|*`（）()，。、：；「」【】\[\]{}<>~]+")
# 拒答。prompt 规则 2 明确要求资料里没有就说「资料里没有提到」——
# 那是**合规行为**，不是答错。不单独统计的话 RAG 臂会被白白扣分。
REFUSAL_RE = re.compile(r"资料里没有提到|资料中没有提到|资料里没有|未提到|没有相关")

METRICS = ("strict", "strict_norm", "strict_fuzzy", "ident", "gram")


def is_refusal(answer: str) -> bool:
    """**纯**拒答：整段回答基本就只有「资料里没有提到」。

    ⚠️ 早先写成"出现这个词就算拒答"，**实测立刻误判**：RAG 臂答完精洗价的
    正常规则和例外之后，收尾说「资料里没有提到除…之外的其他例外」——
    那是**按规则 2 如实划定边界**，却被记成弃答。一个答对的答案被记成弃权，
    汇总里直接多出一条"RAG 拒答 1 次"的假结论。

    所以加长度条件：真拒答很短，划边界的句子挂在长答案的尾巴上。
    （阈值 40 是规则，写死在这里，人和机器都看得见。）
    """
    if not REFUSAL_RE.search(answer):
        return False
    return len(norm(answer)) <= 40


def fuzzy_in(needle: str, hay: str, k: int = 1) -> bool:
    """needle 是否以 **≤k 个编辑**（插入/删除/替换）出现在 hay 里。

    ⚠️ 加这个指标是因为实测抓到一个**答对却判 0** 的案例：
        锚点  「精洗价直接手动填写」
        答案  「精洗价**就**直接手动填写」   ← 多一个「就」
    `strict` 和 `strict_norm` 双双判 0，而那个答案是**完全正确**的。

    换来的教训：**`strict` 量的不是「答对没有」，是「有没有照着原文抄」。**
    手里原文多的那一臂天然占便宜（长上下文臂甚至带引号+§编号地整句复述，
    拿满 100%）。所以不能拿 strict 当主指标，需要一个容许改写的最小指标。

    短锚点（<8 字）不做模糊 —— 太容易撞上，那就不是判分了。
    """
    n = len(needle)
    if needle in hay:
        return True
    if k < 1 or n < 8:
        return False
    # 插入/删除：锚点去掉任一个字之后仍是 hay 的子串
    if any(needle[:i] + needle[i + 1:] in hay for i in range(n)):
        return True
    # 替换：hay 中某个等长窗口只差一个字
    for j in range(len(hay) - n + 1):
        w = hay[j:j + n]
        if w[0] != needle[0] and w[-1] != needle[-1]:
            continue                     # 首尾都对不上，不可能只差一个
        diff = 0
        for a, b in zip(needle, w):
            if a != b:
                diff += 1
                if diff > k:
                    break
        else:
            return True
    return False


def idents(anchor: str) -> list[str]:
    """机械抽出锚点里的标识符/数字。全中文的锚点会返回空列表 —— 这是已知缺陷。"""
    out, seen = [], set()
    for m in IDENT_RE.finditer(anchor):
        t = (m.group(1) or m.group(0)).strip()
        if len(t) < 2 or t.isdigit() and len(t) > 6:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def ngrams(anchor: str, n: int = 3) -> list[str]:
    """字符 n-gram。对中文有效（不用分词），代价是混进跨词碎片。"""
    s = PUNCT_RE.sub("", anchor)
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def norm(text: str) -> str:
    """判分用的规范化：去标点空白。**只用于 strict_norm / gram / 命中判断** ——
    `strict` 必须用原文逐字比，规范化了就不叫"逐字"了。"""
    return PUNCT_RE.sub("", text)


def judge_group(answer: str, group: list[str]) -> dict:
    """一组锚点的四项覆盖。每组是一份完整答案。"""
    na = norm(answer)
    hits = dict(strict=0, strict_norm=0, strict_fuzzy=0, ident=None, gram=None)
    i_all = i_hit = g_all = g_hit = 0
    detail = []

    for a in group:
        st = a in answer                    # 原文逐字
        sn = norm(a) in na                  # 去标点后逐字
        sf = sn or fuzzy_in(norm(a), na)    # 容许 1 个字的插入/删除/替换
        hits["strict"] += st
        hits["strict_norm"] += sn
        hits["strict_fuzzy"] += sf

        ids = idents(a)
        gi = sum(1 for t in ids if t in answer or t in na)
        i_all += len(ids)
        i_hit += gi

        gs = ngrams(a)
        gg = sum(1 for gr in gs if gr in na)
        g_all += len(gs)
        g_hit += gg

        detail.append(dict(anchor=a[:40], ident=ids, gram_hit=gg, gram_all=len(gs),
                           strict=st, strict_norm=sn, strict_fuzzy=sf))

    n = len(group)
    return dict(
        strict=hits["strict"] / n,
        strict_norm=hits["strict_norm"] / n,
        strict_fuzzy=hits["strict_fuzzy"] / n,
        # None = 这组锚点一个标识符都抽不出（全中文）→ 该题这个指标作废
        ident=(i_hit / i_all) if i_all else None,
        gram=(g_hit / g_all) if g_all else 0.0,
        detail=detail,
    )


def judge(answer: str, case: dict) -> dict:
    """按 any / all 语义把各组并起来。

    any（集中题）：任一组凑齐即可 → 取**最好**的那组
    all（散落题）：每组都要凑齐   → 取**最差**的那组（最弱一环）

    ⚠️ **每个指标各自聚合，不共用"获胜组"。**
    早先的写法是先用 (strict, gram) 选出一个获胜组、返回那一组的整个 dict，
    结果 `ident` 是"获胜组的 ident"，而两臂的获胜组可能不是同一组
    → 汇总时两臂的 ident 平均在**不同的题目子集**上，分母都不一样。
    现在四个指标各自在**同一批组**上做 any→max / all→min，
    并且只要有一组抽不出标识符，整题的 ident 就记为 None（两臂同时作废），
    保证两臂的题目集合始终相同。
    """
    per = [judge_group(answer, g) for g in case["groups"]]
    pick = max if case["mode"] == "any" else min

    out = {}
    for m in METRICS:
        vals = [g[m] for g in per if g[m] is not None]
        # 有一组作废 → 整题作废，否则两臂分母会漂
        out[m] = None if len(vals) != len(per) else (pick(vals) if vals else None)

    return dict(out, per_group=per, chars=len(answer),
                refusal=is_refusal(answer))


# ── 零假设：两个用来抵消"偏向长上下文"的对照数 ─────────────────────────

def build_corpus_ngrams(corpus: str) -> set[str]:
    """语料里所有字符 3-gram。用来量答案有多少是**从资料里抄的**。"""
    s = norm(corpus)
    return {s[i:i + 3] for i in range(len(s) - 2)}


def copy_ratio(answer: str, corpus_grams: set[str]) -> float:
    """答案自己的 3-gram 有多大比例能在语料里找到。

    长上下文臂资料多，**复述就能拿分**。这个数上去了，
    strict / gram 的高分就要打折看。这是 R2 偏差的直接读数。
    """
    gs = ngrams(answer)
    if not gs:
        return 0.0
    return sum(1 for g in gs if g in corpus_grams) / len(gs)


# ── 两臂 ────────────────────────────────────────────────────────────────

def run_rag(question: str, store, corpus: str, k: int, llm):
    import rag
    import longctx

    hits = store.similarity_search(question, k=k)
    material = rag.format_docs(hits)
    msg = llm.invoke(longctx.build_messages(material, question))
    return (msg.content, longctx.usage_of(msg),
            [(d.metadata["i"], d.metadata.get("source", "?")) for d in hits])


def run_long(question: str, store, corpus: str, k: int, llm):
    import longctx

    msg = llm.invoke(longctx.build_messages(corpus, question))
    return msg.content, longctx.usage_of(msg), None


ARMS = {
    "rag": ("RAG（检索 top-k）", run_rag),
    "long": ("长上下文（整份语料）", run_long),
}


# ── 汇总 ────────────────────────────────────────────────────────────────

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(v, pct=True):
    if v is None:
        return "—"
    return f"{v * 100:.0f}%" if pct else f"{v:.0f}"


def summarize(rows: list[dict], filter_fn, label: str) -> None:
    """一张小表：四种切分（题型 / 题库）都用它，免得代码写四遍。"""
    sel = [(q, r) for q, r in rows if filter_fn(q)]
    if not sel:
        return
    print(f"\n  {label}（{len(sel)} 题）")
    print(f"    {'臂':<22}{'整句':>7}{'规整句':>8}{'容错句':>8}{'标识符':>8}{'3-gram':>8}"
          f"{'字数':>7}{'复述率':>8}{'拒答':>6}")
    for arm in ARMS:
        rs = [r[arm] for _q, r in sel if arm in r]
        if not rs:
            continue
        idv = _mean([x["score"]["ident"] for x in rs])
        nid = sum(1 for x in rs if x["score"]["ident"] is not None)
        ref = sum(1 for x in rs if x["score"]["refusal"])
        print(f"    {ARMS[arm][0]:<22}"
              f"{_fmt(_mean([x['score']['strict'] for x in rs])):>7}"
              f"{_fmt(_mean([x['score']['strict_norm'] for x in rs])):>8}"
              f"{_fmt(_mean([x['score']['strict_fuzzy'] for x in rs])):>8}"
              f"{_fmt(idv):>8}"
              f"{_fmt(_mean([x['score']['gram'] for x in rs])):>8}"
              f"{_mean([x['score']['chars'] for x in rs]):>7.0f}"
              f"{_fmt(_mean([x['copy'] for x in rs])):>8}"
              f"{ref:>4}/{len(rs)}")
        if nid != len(rs):
            print(f"      ↑ 标识符只统计了 {nid}/{len(rs)} 题"
                  f"（其余题的锚点全是中文，抽不出标识符）")


def report(rows: list[dict], k: int, nulls: dict) -> None:
    print("\n" + "=" * 84)
    print("逐题")
    print("=" * 84)
    for q, by_arm in rows:
        print(f"\n[{q['题库']}/{q['type']}/{q['mode']}] {q['q']}")
        for arm, r in by_arm.items():
            s = r["score"]
            print(f"   {ARMS[arm][0]:<20}"
                  f" 整句 {_fmt(s['strict']):>4} 规整句 {_fmt(s['strict_norm']):>4}"
                  f" 容错句 {_fmt(s['strict_fuzzy']):>4}"
                  f" 标识符 {_fmt(s['ident']):>4} 3-gram {_fmt(s['gram']):>4}"
                  f"  {s['chars']:>4}字 复述 {_fmt(r['copy']):>4}"
                  + ("  ⛔拒答" if s["refusal"] else ""))
            if r["hits"]:
                src = [x[1] for x in r["hits"]]
                nbug = sum(1 for x in src if "bug" in x)
                print(f"   {'':<20} 召回 {[x[0] for x in r['hits']]}"
                      f"  （其中 {nbug}/{len(src)} 块来自 bug 文档）")

    # ---- 汇总：分题库、分题型，**不给合计头条数字** ----
    print("\n" + "=" * 84)
    print(f"汇总（k={k}）")
    print("=" * 84)
    print("  ⚠ 四个指标全都偏向长上下文臂（详见证 docstring），别单看一列。")
    for 题库 in ("设计文档", "bug文档"):
        for t in ("集中", "散落"):
            summarize(rows, lambda q, a=题库, b=t: q["题库"] == a and q["type"] == b,
                      f"{题库} × {t}")

    # ---- 零假设 ----
    print("\n" + "=" * 84)
    print("抵消偏差用的对照数")
    print("=" * 84)
    print(f"  {'臂':<22}{'跨题蒙中(gram_null)':>22}{'平均复述率':>14}")
    for arm in ARMS:
        rs = [r[arm] for _q, r in rows if arm in r]
        if not rs:
            continue
        print(f"  {ARMS[arm][0]:<22}"
              f"{_fmt(_mean([x['gram_null'] for x in rs])):>22}"
              f"{_fmt(_mean([x['copy'] for x in rs])):>14}")
    print("    跨题蒙中 = 拿本题答案去对**别题**的锚点，纯蒙的地板；随话多而涨。")
    print("    复述率 = 答案的 3-gram 能在语料里找到的比例；高 = 在抄资料，")
    print("            那么 strict / gram 的高分要打折看。")

    # ---- 拒答 ----
    ref = {arm: sum(1 for _q, r in rows if arm in r and r[arm]["score"]["refusal"])
           for arm in ARMS}
    if any(ref.values()):
        print(f"\n  拒答（回答「资料里没有提到」）："
              + "，".join(f"{ARMS[a][0]} {n} 次" for a, n in ref.items()))
        print("    ⚠ prompt 规则 2 要求资料没有就说没提到 —— 拒答是**合规**，不是答错。")
        for q, by_arm in rows:
            for arm, r in by_arm.items():
                if r["score"]["refusal"]:
                    print(f"    [{arm}] {q['题库']}｜{q['q'][:36]}…")

    # ---- 打架检测：只在**能影响结论**的分歧上报 ----
    # 早先阈值用 max-min>0.5，而 strict 常常≈0、gram≈0.6，
    # 几乎每题都"打架" → 狼来了，没人看。改成真分歧才报。
    print("\n  指标真分歧（一项说行、一项说不行，最值得人工看；已落盘）:")
    any_bad = False
    for q, by_arm in rows:
        for arm, r in by_arm.items():
            s = r["score"]
            vals = [s[m] for m in ("strict_fuzzy", "gram") if s[m] is not None]
            if vals and max(vals) >= 0.5 and min(vals) <= 0.2:
                any_bad = True
                print(f"    [{arm}] {q['q'][:32]}…"
                      f"  容错句 {_fmt(s['strict_fuzzy'])} / 3-gram {_fmt(s['gram'])}")
    if not any_bad:
        print("    （没有 —— 指标一致，结论可以放心些）")


def main() -> None:
    ap = argparse.ArgumentParser(description="第 6 步：答案级评测")
    ap.add_argument("--arms", default="rag,long", help="跑哪些臂，逗号分隔")
    ap.add_argument("-k", type=int, default=4, help="RAG 臂召回几块（默认 4）")
    ap.add_argument("--limit", type=int, help="只跑前 N 道（先小样试）")
    args = ap.parse_args()

    import rag
    import longctx
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")   # import rag 不会加载 .env

    for c in CASES:
        c["题库"] = "设计文档"
    for c in BUG_CASES:
        c["题库"] = "bug文档"
    questions = CASES + BUG_CASES
    if args.limit:
        questions = questions[:args.limit]

    corpus = longctx.load_corpus()
    print(f"语料 {len(corpus):,} 字符；{len(questions)} 道题；臂 = {args.arms}")

    emb = rag.get_embeddings()
    store = rag.load_index(emb)
    print(rag.describe_index(getattr(store, "chunker_meta", None)))
    llm = rag.get_llm()

    # ---- S1 语料完整性断言（阻塞级）----
    # 索引如果是旧的（只有设计文档），Arm A 物理上看不到 bug 文档，
    # 而 Arm B 看得到 —— 那比的就不是检索策略，是"两臂语料不同"，全盘作废。
    chunks = {d.metadata["i"]: d.metadata.get("source", "?")
              for d in (store.docstore.search(i) for i in store.index_to_docstore_id.values())}
    srcs = sorted({s for s in chunks.values()})
    print(f"索引 {len(chunks)} 块；来源 {srcs}")
    if len(srcs) < 2:
        sys.exit("✋ 索引里只有一份语料 —— 多半是 --rebuild 没跑完/被杀，\n"
                 "   而旧索引还留在原地，**不会报错**。先重建：\n"
                 "   .venv/Scripts/python.exe src/rag.py --rebuild \"任意问题\"")
    if "bug" not in " ".join(srcs):
        sys.exit(f"✋ 索引里没有 bug 文档（来源={srcs}），先重建索引。")

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    corpus_grams = build_corpus_ngrams(corpus)
    rows, cost = [], []
    for qi, q in enumerate(questions, 1):
        by_arm = {}
        for arm in arms:
            t0 = time.time()
            text, usage, hits = ARMS[arm][1](q["q"], store, corpus, args.k, llm)
            by_arm[arm] = dict(
                answer=text, usage=usage, hits=hits, secs=time.time() - t0,
                score=judge(text, q), copy=copy_ratio(text, corpus_grams))
            if usage:
                cost.append(usage)
            print(f"  [{qi}/{len(questions)}] {arm:<5} {time.time() - t0:5.1f}s"
                  f"  {q['q'][:30]}…")
        rows.append((q, by_arm))

    # ---- S4 跨题蒙中地板：拿本题答案去对**别题**的锚点 ----
    #
    # ⚠️ 必须用**远处**的题，不能用相邻的下一题 —— 这是实测踩出来的：
    # 一开始用 `(i+1)`，「支付报 500」(Bug 1) 那道题的长上下文臂地板竟高达 **92%**，
    # 因为它的答案里连 Bug 2 的原文都抄进去了（相邻两条 bug 互相引用、
    # 同属一个模块）。于是 `gram - null` 把 **100% 的正确答案压成 8%**，
    # 凭空造出一个"RAG 赢了"的假结论。
    # 相邻题不是随机对照，是**高度相关**的对照。
    # 取半程之外的题后，地板从 0~92% 收到 0~18%。
    for i, (_q, by_arm) in enumerate(rows):
        other = rows[(i + len(rows) // 2) % len(rows)][0]
        for arm, r in by_arm.items():
            per = [judge_group(r["answer"], g) for g in other["groups"]]
            r["gram_null"] = _mean([p["gram"] for p in per]) or 0.0

    # ---- S2 prompt 真的送到了吗：长上下文臂的输入应恒定 ----
    lh = [r["long"]["usage"].get("input") for _q, r in rows
          if "long" in r and r["long"]["usage"].get("input")]
    if lh:
        print(f"\n  长上下文臂输入 tokens：{min(lh):,} ~ {max(lh):,}"
              f"（{len(lh)} 次调用）")
        if max(lh) - min(lh) > 2000:
            print("  ⚠ 输入长度在变 —— 前缀没固定住，或 prompt 被截断了，"
                  "缓存和'整份语料都送到了'两个前提都不成立")

    # ---- 落盘：这是最终防线，人得能读到原文 ----
    DATA.mkdir(exist_ok=True)
    dump = DATA / f"answers_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    with dump.open("w", encoding="utf-8") as f:
        for q, by_arm in rows:
            for arm, r in by_arm.items():
                f.write(json.dumps(dict(
                    q=q["q"], 题库=q["题库"], type=q["type"], mode=q["mode"], arm=arm,
                    answer=r["answer"], score=r["score"], usage=r["usage"],
                    hits=r["hits"], copy=r["copy"], gram_null=r["gram_null"],
                ), ensure_ascii=False) + "\n")

    report(rows, args.k, {})

    # ---- 用量：缓存到底命中没有，不能靠猜。**分臂报** ----
    # 两臂的缓存行为根本不同（长上下文臂前缀固定，RAG 臂每问都变），
    # 混在一起报会把 RAG 臂正常的 0% 当成故障。
    print()
    for arm in ("long", "rag"):
        us = [r[arm]["usage"] for _q, r in rows if arm in r]
        if not us:
            continue
        inp = sum(u.get("input") or 0 for u in us)
        out = sum(u.get("output") or 0 for u in us)
        hit = sum(u.get("cache_hit") or 0 for u in us)
        line = f"  {ARMS[arm][0]}：输入 {inp:,} / 输出 {out:,}"
        if hit:
            line += f"；**缓存命中 {hit:,}（{hit / inp * 100:.1f}%）**"
        elif arm == "long":
            line += "\n  ⚠ **一处缓存都没命中** —— 前缀没固定住，成本估算不成立，先查这个"
        else:
            # ⚠ 别写"RAG 臂 0% 是正常的" —— 实测小样里 RAG 臂拿到过 80.7% 命中，
            # 说明检索块每问都变并**没有**阻止前缀复用（共同的那段指令头命中了）。
            # 只报事实，不替它编机制。
            line += "；缓存 0%（本次没命中）"
        print(line)
    print(f"\n  原始答案：{dump.relative_to(ROOT)}  ← 请抽读几条，别只看分数")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
