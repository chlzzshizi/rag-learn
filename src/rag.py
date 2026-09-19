#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangChain 版 RAG —— 切块 / 编码 / 检索 / 生成 全流程。

跟之前手写版的对照（手写版在 git 1963514，要翻：git show 1963514:src/step1_chunk.py）：

    手写版                          LangChain 版
    ------------------------------  ------------------------------------
    step1_chunk.py  自己写切片循环   RecursiveCharacterTextSplitter
    step2_encode.py 自己调模型       HuggingFaceEmbeddings + FAISS.from_documents
    step3_retrieve.py 自己算点积     vectorstore.as_retriever()
    （没写）                         ChatPromptTemplate | ChatOpenAI | StrOutputParser

用法：

    .venv/Scripts/python.exe src/rag.py                        # 跑内置的几道题
    .venv/Scripts/python.exe src/rag.py "订单状态有哪些状态"
    .venv/Scripts/python.exe src/rag.py "..." --k 3 --show    # 顺便打印召回的原文
    .venv/Scripts/python.exe src/rag.py "..." --rebuild       # 强制重建索引

索引存在 data/lc_index/，第二次跑会直接读，不用重新编码（建一次约 2 分钟）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "lc_index"
CORPUS = Path(r"C:\Users\33705\Desktop\项目\2026-07-04-yunxi-redesign.md")
MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"

# 跟手写版一样的 500 / 0，好直接对比。
# 注意：这个 splitter **不会**在固定位置切，它按 separators 从大到小找边界
# （段落 → 换行 → 空格 → 字符）。所以同样是 500，切出来的块数会和手写版不同
# —— 那个差异就是"语义切块 vs 固定切块"的全部内容。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0

QUESTIONS = [
    "订单状态有哪些状态",
    "抢券的时候怎么防止超卖",
]


def get_embeddings():
    """本地 Qwen3 模型包成 LangChain 的 Embeddings 接口。

    两个参数要跟手写版保持一致，否则检索分数没意义：
      * dtype=float32 —— 不传的话 sentence-transformers 会读成 bfloat16，
        归一化范数在 0.998~1.002 之间飘（详见 README「技术选型」）
      * normalize_embeddings=True —— 归一化后点积才等于余弦
    """
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    if not MODEL_DIR.exists():
        sys.exit(f"找不到模型：{MODEL_DIR}\n先跑：python scripts/fetch_model.py")

    return HuggingFaceEmbeddings(
        model_name=str(MODEL_DIR),
        # ⚠ 这里要嵌两层，不是笔误。langchain_huggingface/embeddings/huggingface.py:98
        # 是这么调的：  model_cls(self.model_name, cache_folder=..., **self.model_kwargs)
        # 也就是说传进来的 model_kwargs 会被**展开成 SentenceTransformer 的直接参数**。
        # 而 SentenceTransformer 不接受 dtype 这个直接参数 —— 它要的是
        # model_kwargs={"dtype": ...}。所以外面这层 key 必须叫 "model_kwargs"。
        # 写错的表现是：TypeError: ... unexpected keyword argument 'dtype'
        model_kwargs={"model_kwargs": {"dtype": torch.float32}, "device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        # 本地模型，别去 HuggingFace 上查有没有更新（这台机器上 HF 连不通）
        cache_folder=str(MODEL_DIR.parent),
    )


def build_index(embeddings, verbose=True):
    """读文档 → 切块 → 编码 → 存 FAISS。"""
    import torch  # noqa: F401  （确认 torch 在，报错更早更清楚）
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not CORPUS.exists():
        sys.exit(f"找不到语料：{CORPUS}")

    text = CORPUS.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # 中文没有空格，默认的 " " 分隔符基本用不上；显式加上中文标点，
        # 让它在句子/分句的边界上找落脚点。
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    docs = splitter.create_documents([text])
    for i, d in enumerate(docs):
        d.metadata["source"] = CORPUS.name
        d.metadata["i"] = i

    if verbose:
        print(f"切块  {len(text):,} 字符 → {len(docs)} 块"
              f"（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
        print(f"      手写固定切块同样参数是 148 块 —— 差多少就是边界选择的差别")
        print("编码中（第一次约 2 分钟）…")

    store = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    if verbose:
        print(f"      索引已存到 {INDEX_DIR.relative_to(ROOT)}/")
    return store, docs


def load_index(embeddings):
    from langchain_community.vectorstores import FAISS

    # allow_dangerous_deserialization：FAISS 索引是 pickle，加载等于执行代码。
    # 这里加载的是**我们自己刚生成的**文件，所以可以开；从网上下来的索引绝不能开。
    return FAISS.load_local(str(INDEX_DIR), embeddings,
                            allow_dangerous_deserialization=True)


def get_llm():
    """DeepSeek。它是 OpenAI 兼容接口，所以直接借 ChatOpenAI 用。"""
    from langchain_openai import ChatOpenAI

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("没读到 DEEPSEEK_API_KEY —— 检查项目根目录的 .env")
    return ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )


PROMPT = """你是一个只根据给定资料回答问题的助手。

规则：
1. 只使用下面「资料」里的内容回答，不要用你自己的知识补充。
2. 资料里没有的，直接回答「资料里没有提到」，不要猜。
3. 回答完在最后一行写上「来源：第 X 块」，X 是你实际用到的那一块的编号。

资料：
{context}

问题：{question}
"""


def format_docs(docs):
    return "\n\n".join(f"【第 {d.metadata['i']} 块】\n{d.page_content}" for d in docs)


def ask(question, retriever, chain, k, show_sources):
    # 1. 检索
    hits = retriever.invoke(question)

    print(f"\n{'=' * 72}")
    print(f"问：{question}")
    print("=" * 72)
    print(f"召回 {len(hits)} 块：")
    for rank, d in enumerate(hits, 1):
        preview = d.page_content[:56].replace("\n", "⏎")
        print(f"  {rank}. 第 {d.metadata['i']} 块  {len(d.page_content):>4} 字  {preview}…")

    if show_sources:
        for rank, d in enumerate(hits, 1):
            print(f"\n--- 第 {d.metadata['i']} 块全文 ---")
            print(d.page_content)

    # 2. 生成
    print("\n答：")
    answer = chain.invoke({"context": format_docs(hits), "question": question})
    print(answer)


def main():
    ap = argparse.ArgumentParser(description="LangChain 版 RAG")
    ap.add_argument("question", nargs="*", help="问题；不填就跑内置的几道")
    ap.add_argument("-k", type=int, default=4, help="召回几块（默认 4）")
    ap.add_argument("--rebuild", action="store_true", help="强制重建索引")
    ap.add_argument("--show", action="store_true", help="打印召回的原文全文")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    import torch
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    embeddings = get_embeddings()

    if args.rebuild or not INDEX_DIR.exists():
        store, _ = build_index(embeddings)
    else:
        print(f"读已有索引 {INDEX_DIR.relative_to(ROOT)}/（要重建加 --rebuild）")
        store = load_index(embeddings)

    retriever = store.as_retriever(search_kwargs={"k": args.k})
    llm = get_llm()
    chain = ChatPromptTemplate.from_template(PROMPT) | llm | StrOutputParser()

    for q in (args.question or QUESTIONS):
        ask(q, retriever, chain, args.k, args.show)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
