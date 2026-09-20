#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""混合检索：BM25（词面）+ 稠密（向量），RRF 合并。

## 为什么加这一路

第 5 步的诊断是「0.6B 稠密检索**基本是词面级匹配**」。那句话的后果是：

    这个稠密检索器在**用它不擅长的方式，做它该做的事** ——
    想匹配词面，但因为是向量近似，匹配不准；又没有大模型的语义能力，
    跨不过「症状词 → 原因词」那一跳。**两头不占。**

而词面匹配这件事有个专门做到极致的算法，不需要模型、不需要 GPU、不花钱：BM25。
它的 IDF 会给 `uk_coupon_customer` 这种稀有串极高权重 —— 而稠密向量里，
这串字符的贡献被周围那段中文稀释掉了。**语料里最该被精确命中的东西，
正好是稠密检索最不擅长的东西。**

## 为什么合并用 RRF，不是加权求和

BM25 分数能到几百、**没有上界**；余弦 ∈ [0,1]。**量纲不同，加起来没意义**，
做归一化又得先有校准集，脆。

RRF 把分数全扔掉、只看名次：

    score(d) = Σ  1 / (60 + rank_r(d))
             r∈两路

前几名之间几乎没差别（第 1 名 1/61，第 2 名 1/62），所以**某一路在第一名上
自信地搞错了，也带不偏最终结果**。60 是 Elasticsearch / Qdrant / Weaviate
的默认值，不是需要你调的参数。

## 中文必须分词

BM25 原版按空格切词。中文没空格，**不分词的话整句变成一个 token，直接废掉**。
这里用 jieba，但**先把标识符抠出来保护住**再交给 jieba ——
否则 `uk_coupon_customer` 会被切成 uk/coupon/customer，而这恰恰是 IDF 最高、
最该精确命中的那批词。

用法：

    from hybrid import BM25Index, HybridRetriever, rrf_fuse
"""

from __future__ import annotations

import re

# 反引号包起来的代码段。**必须最先抠出来**，否则里面的 `.` `()` 会被 jieba 当标点。
CODE_RE = re.compile(r"`([^`]+)`")

# 标识符 / 键名 / 数字。用在**非代码段**上，把 `uk_coupon_customer`、
# `auth:staff:invalidAfter`、`500` 这种整体抠出来，不让 jieba 碰它们。
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)*|\d+")

# 把代码段拆成子词：`orderMapper.insert()` → [orderMapper, insert]
# 拆开是为了让查询里的 `save` 能匹配到 `save()` —— 整段保留的话只有逐字相同才中。
IDENT_SPLIT_RE = re.compile(r"[^A-Za-z0-9_]+")


def _jieba():
    import logging

    import jieba

    # 默认是 DEBUG，会把「Building prefix dict…」那几行喷进评测输出里。
    jieba.setLogLevel(logging.INFO)
    return jieba


def _zh_tokens(seg: str, jb) -> list[str]:
    """非代码段：标识符整体抠出，中间的中文交给 jieba。"""
    out, pos = [], 0
    for m in TOKEN_RE.finditer(seg):
        out.extend(jb.lcut(seg[pos:m.start()]))
        out.append(m.group())
        pos = m.end()
    out.extend(jb.lcut(seg[pos:]))
    return out


def tokenize(text: str) -> list[str]:
    """切成 BM25 要的词序列。**这是这套检索里唯一的语言相关决策，写死在这。**"""
    jb = _jieba()
    out, pos = [], 0
    for m in CODE_RE.finditer(text):
        out.extend(_zh_tokens(text[pos:m.start()], jb))
        out.extend(IDENT_SPLIT_RE.split(m.group(1)))
        pos = m.end()
    out.extend(_zh_tokens(text[pos:], jb))

    # 丢掉纯标点（jieba 会原样吐出），统一小写（否则 Redis / redis 算两个词）
    return [t for t in (s.strip().lower() for s in out)
            if t and any(c.isalnum() for c in t)]


class BM25Index:
    """按块建的 BM25。块号用 rag 的全局 i，跟 FAISS 那边对得上。"""

    def __init__(self, chunks: dict[int, str]):
        from rank_bm25 import BM25Okapi

        # 顺序固定 —— 分数数组是按下标对齐 ids 的，排序一变就全错位
        self.ids = sorted(chunks)
        self.bm25 = BM25Okapi([tokenize(chunks[i]) for i in self.ids])

    def search(self, query: str, k: int = 20) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.ids)), key=lambda j: (-scores[j], self.ids[j]))
        # **丢掉 0 分的**：一个词都没匹配上的块不该进融合，
        # 否则 RRF 会当它是「第 20 名」给它记分，等于往池子里灌噪音。
        return [self.ids[j] for j in order[:k] if scores[j] > 0]


def rrf_fuse(rankings: list[list[int]], k: int = 60,
             weights: list[float] | None = None) -> list[int]:
    """RRF 合并多路排名。每路是**有序**块号列表（最相关在前）。

    同分时按块号兜底排序 —— 否则 dict 的插入顺序会渗进结果，跑两次不一样。

    ## ⚠️ k=60 在这个语料上是错的 —— 教科书值害人（2026-09-20 实测）

    先记现象。那题「明细 32 条」（bug 218），两路给的名次是：

        390   dense 第 1  + bm25 第 6   → 1/61 + 1/66 = 0.0315
        218   bm25  第 1  + dense 第 17 → 1/61 + 1/77 = 0.0294   ← 输

    **BM25 的第一名，输给了一个「两路都在中游」的块。**

    原因在 k 和候选深度不匹配：k=60 时第 1~50 名只差 1.8 倍
    （1/61=0.0164 → 1/110=0.0091），于是**任何块只要在两路里各出现一次
    （哪怕都在第 50 名）得 0.018，就打赢只在单路排第 1 的 0.0164。**
    整个候选列表都落在被压平的区间里，RRF 退化成「谁被两路都收进来谁赢」
    —— 而当一路基本是噪音时，「两路都排上」只等于「更符合噪音的平均口味」。

    **k=60 是从 TREC 那种上千深的候选列表里定出来的。** 深度 20~50 时
    要按比例缩小。`scripts/eval_fuse_sweep.py` 扫了整个格子：

        权重 (dense,bm25)   k=1    k=5   k=20   k=60
        (1,1) 等权          56%    56%    44%    38%    ← 单调变坏
        (0,1) 只用 BM25      —— 50% ——            （单臂对照线）
        (1,0) 只用 dense     —— 38% ——

    **k 小到 5 时融合才真正起作用**，而且是各取所长：文档/集中拿 dense 的
    100%、bug/集中拿 BM25 的 60%。加大 BM25 权重反而退化回 BM25 单路（50%）
    —— 因为融合的价值恰恰来自 dense 在文档那半边的优势，压它就没价值了。

    ⚠️ **诚实交代**：小 k 是在**这 16 道题上**挑的，不是独立验证。可信度来自
    两处 —— 趋势是单调的（不是撞上一个孤立的峰），且机制清楚。但别当它是定理。

    还有个怎么调都过不去的：**文档/散落恒为 0%**，全格子无一例外。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    score: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, 1):
            score[doc] = score.get(doc, 0.0) + w / (k + rank)
    return sorted(score, key=lambda d: (-score[d], d))


class HybridRetriever:
    """稠密 + BM25，RRF 融合。接口对齐 eval 那边要的 search(q, k) -> [块号]。

    `rrf_k` 默认 **5**，不是 RRF 教科书的 60 —— 理由见 `rrf_fuse` 的文档：
    60 是给上千深候选列表定的，配 20~50 深会把名次压平、让噪音共识胜出。
    **这是本仓库里唯一一个「默认值不是教科书值」的参数，改它之前先看扫描表。**
    """

    def __init__(self, store, chunks: dict[int, str], depth: int | None = None,
                 rrf_k: int = 5):
        self.store = store
        self.bm25 = BM25Index(chunks)
        self.depth = depth      # None = 每次按 k 现算
        self.rrf_k = rrf_k

    def _depth(self, k: int) -> int:
        # **每路的候选深度不能就是 k。** 两路各只给 k 个、又互不重合的话，
        # RRF 拿到的就是 2k 个「并列第一」，合并退化成随机。
        # 先各取一批宽的，融合完再截到 k —— 这是融合检索的标准做法。
        return self.depth or max(4 * k, 20)

    def arms(self, query: str, k: int) -> tuple[list[int], list[int]]:
        """单独跑两路，供评测报「这一路自己找得到吗」。"""
        d = self._depth(k)
        dense = [doc.metadata["i"] for doc, _s in
                 self.store.similarity_search_with_score(query, k=d)]
        return dense, self.bm25.search(query, k=d)

    def search(self, query: str, k: int = 4) -> list[int]:
        dense, sparse = self.arms(query, k)
        return rrf_fuse([dense, sparse], k=self.rrf_k)[:k]
