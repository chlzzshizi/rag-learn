#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""长上下文问答：**不检索**，整个语料直接塞进 prompt。

    .venv/Scripts/python.exe src/longctx.py "订单状态有哪些状态"
    .venv/Scripts/python.exe src/longctx.py            # 跑内置的几道
    .venv/Scripts/python.exe src/longctx.py --stats    # 只看语料规模，不调 API

## 为什么要单独一个文件

第 5 步量出 RAG「集中题 100%、散落题 0%」，由此推出「该上多路召回」。
但那个推论建立在一个错误前提上：我以为上下文窗口是 128K。实际
`deepseek-flash` / `deepseek-v4-pro` 是 **1M**（V4 起；128K 是上一代 V3.2 的数）。

量了一下真实规模：

    设计文档        73,785 字符    41,139 tokens
    bug-record.md  124,866 字符    72,368 tokens
    ─────────────────────────────────────────────
    合计           198,651 字符   113,507 tokens   = 1M 的 11.4%

**塞得下。** 所以真正该先回答的不是「怎么修检索」，而是「这个语料上检索带来了什么」。
这个文件是那条对照臂。

## 两个必须遵守的约束

**① 不用 `.format()`，也不用 ChatPromptTemplate 去渲染语料。**
语料里有 **147 处花括号**（设计文档 53 + bug 文档 94；`{couponId}`、`{staffId}`、
`{ ... }`），任何把语料当模板变量传的写法都会 `KeyError: 'couponId'`。
这里走 messages 列表 + 纯字符串拼接，**不经过任何模板引擎**。

> 这个数**跟着语料走**，改动语料后重新数一遍再改这里：
> `.venv/Scripts/python.exe -c "import sys;sys.path.insert(0,'src');import rag;
> print(sum(t.count('{') for _n,t in rag.corpus_texts()))"`

**② 固定的放前面，问题放最后。**
DeepSeek 的上下文缓存要求「从第 0 个 token 起前缀逐字节相同」才算命中，
命中价约为未命中的 1/50。语料是这个固定前缀，问题放最后，所以
**同一份语料问第二个问题时就已经全部命中缓存了**。
一旦把问题挪到语料前面，前缀每天每问都变，缓存全废 —— 成本差 50 倍。

⚠️ 和 `rag.py` 一样，`import rag` **不会**加载 `.env`（那只在 `rag.main()` 里做），
所以这里显式 `load_dotenv`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── 两臂共用的指令 ────────────────────────────────────────────────────────
# 这段必须和 RAG 臂**逐字节相同**，否则比的就不是检索策略了。
#
# 注意这里**没有** rag.py 那条「回答完写上来源：第 X 块」——
# 长上下文臂没有「第 X 块」可写，留着它两臂就不可比了。
# 代价是 RAG 臂的输出会和 `rag.py` 直接跑出来的不同；没有损失，
# 因为那些输出从来没被评分过（第 5 步评的是检索，不是答案）。
INSTRUCTIONS = """你是一个只根据给定资料回答问题的助手。

规则：
1. 只使用下面「资料」里的内容回答，不要用你自己的知识补充。
2. 资料里没有的，直接回答「资料里没有提到」，不要猜。
"""


def build_messages(material: str, question: str) -> list[dict]:
    """把「资料 + 问题」拼成 messages。

    **两个臂都调这个函数** —— 长上下文臂传整个语料，RAG 臂传检索到的块。
    指令和拼接方式只有这一处定义，「两臂 prompt 相同」就是结构上保证的，
    不靠人记得同步改两个地方。
    """
    return [
        {"role": "system", "content": f"{INSTRUCTIONS}\n资料：\n{material}"},
        {"role": "user", "content": question},
    ]


def load_corpus() -> str:
    """整份语料，顺序固定。走 `rag.corpus_texts()` —— 和索引同源，
    保证两臂看到的字符串逐字节相同（含那个 `# 文件：` 头）。"""
    import rag

    return "\n\n".join(text for _name, text in rag.corpus_texts())


def usage_of(msg) -> dict:
    """从回复里挖出 token 用量，特别是缓存命中数。

    不同版本/网关放的字段不一样，两处都找，找不到就返回空 ——
    **宁可报不出来，也不要编一个数**。
    """
    out = {}
    meta = getattr(msg, "usage_metadata", None) or {}
    out["input"] = meta.get("input_tokens")
    out["output"] = meta.get("output_tokens")
    details = meta.get("input_token_details") or {}
    if "cache_read" in details:
        out["cache_hit"] = details["cache_read"]

    raw = getattr(msg, "response_metadata", None) or {}
    for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
        if k in raw:
            out[k.replace("prompt_cache_", "")] = raw[k]
    return {k: v for k, v in out.items() if v is not None}


def answer(question: str, corpus: str, llm=None):
    """问一句，返回 (答案文本, 用量 dict)。"""
    import rag

    if llm is None:
        llm = rag.get_llm()
    msg = llm.invoke(build_messages(corpus, question))
    return msg.content, usage_of(msg)


def main() -> None:
    import rag
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser(description="长上下文问答（不检索）")
    ap.add_argument("question", nargs="*", help="问题；不填就跑内置的几道")
    ap.add_argument("--stats", action="store_true", help="只看语料规模，不调 API")
    args = ap.parse_args()

    corpus = load_corpus()

    if args.stats:
        # 用真实分词器数，不按字节估 —— 中文一字 2-3 字节，按字节估会差一倍
        from transformers import AutoTokenizer

        tk = AutoTokenizer.from_pretrained(str(rag.MODEL_DIR))
        n = len(tk.encode(corpus))
        print(f"语料 {len(corpus):,} 字符 → {n:,} tokens")
        for name, text in rag.corpus_texts():
            print(f"    {name}：{len(text):,} 字符")
        return

    load_dotenv(ROOT / ".env")  # import rag 不会加载 .env，必须显式来一次

    llm = rag.get_llm()
    for q in (args.question or rag.QUESTIONS):
        text, u = answer(q, corpus, llm)
        print(f"\n{'=' * 72}\n问：{q}\n{'=' * 72}")
        print(text)
        if u:
            hit, inp = u.get("cache_hit"), u.get("input")
            line = f"\n[用量] 输入 {inp} / 输出 {u.get('output')}"
            if hit is not None and inp:
                line += f"；缓存命中 {hit}（{hit / inp * 100:.1f}%）"
            print(line)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
