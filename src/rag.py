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
    .venv/Scripts/python.exe src/rag.py "..." --rebuild --chunker flat   # 用旧切法建

索引存在 data/lc_index/，第二次跑会直接读，不用重新编码（建一次约 2 分钟）。

**索引会把自己的来历写进 `data/lc_index/build.json`**（切法、块数、语料指纹）。
加了 `--chunker` 之后这不是可选项：谁跑一次 `--chunker flat` 建索引，之后所有评测
都会读到 flat 建的东西却以为看的是新的。三个评测脚本都会把这行自述打出来。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from chunking import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKER_DOC, CHUNKERS, pieces_of

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INDEX_DIR = DATA / "lc_index"
BUILD_INFO = INDEX_DIR / "build.json"
# 语料是**多份**了（设计文档 + bug 记录）。顺序固定 —— 索引里的块号 i 依赖它，
# 顺序一变块号全变，对着旧块号记的笔记就废了。
#
# 两份都是 markdown，所以切块那套对两边都成立，不需要按扩展名分派切块器。
CORPUSES = [
    Path(r"C:\Users\33705\Desktop\项目\2026-07-04-yunxi-redesign.md"),
    Path(r"D:\study\yunxi-server\docs\bug-record.md"),
]
MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"

# 切块参数和切法都住在 chunking.py。这里再导一次**只是为了让老调用点不炸**
# （eval_retrieval.py 里引了 rag.CHUNK_SIZE / rag.CHUNK_OVERLAP）。
DEFAULT_CHUNKER = "md"

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


def corpus_texts():
    """产出 (文件名, 带标题的全文)。**索引和长上下文两条路都走这里** ——

    两边拿到的字符串必须逐字节相同，否则对照实验比的就不是检索策略了。
    """
    for src in CORPUSES:
        if not src.exists():
            sys.exit(f"找不到语料：{src}")
        yield src.name, f"# 文件：{src.name}\n\n" + src.read_text(encoding="utf-8")


def build_index(embeddings, chunker=DEFAULT_CHUNKER, verbose=True):
    """读文档 → 切块 → 编码 → 存 FAISS。切法见 chunking.py。"""
    import torch  # noqa: F401  （确认 torch 在，报错更早更清楚）
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    # 逐份切，块号 i **全局连续**（跨文件不重置）。理由是检索评测拿 i 当块的身份，
    # 重置的话两份语料会出现两个「第 3 块」，对不上号。
    docs, total_chars, corpus_meta = [], 0, []
    for name, text in corpus_texts():
        total_chars += len(text)
        corpus_meta.append({
            "name": name, "chars": len(text),
            # 语料一改块号全变，留个指纹好判断「这份索引是哪版语料建的」
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
        for p in pieces_of(text, chunker):
            docs.append(Document(page_content=p.render(True),
                                 metadata={"source": name}))

    for i, d in enumerate(docs):
        d.metadata["i"] = i

    if verbose:
        print(f"切块  {total_chars:,} 字符（{len(CORPUSES)} 份）→ {len(docs)} 块"
              f"（切法={chunker}：{CHUNKER_DOC[chunker]}）")
        for src in CORPUSES:
            n = sum(1 for d in docs if d.metadata["source"] == src.name)
            print(f"      {src.name}：{n} 块")
        print("编码中（第一次约 2 分钟）…")

    store = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))

    # ---- 索引自述 ----
    # **不写这个文件，早晚会把两步的数字搅在一起**：加了 --chunker 之后，
    # 谁跑一次 --chunker flat，之后所有评测都会读到 flat 建的索引却以为看的是新的。
    info = {"chunker": chunker, "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP,
            "block_count": len(docs), "created_at": datetime.now().isoformat(timespec="seconds"),
            "corpus": corpus_meta}
    BUILD_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    if verbose:
        print(f"      索引已存到 {INDEX_DIR.relative_to(ROOT)}/（自述在 build.json）")
    return store, docs


def load_index(embeddings):
    from langchain_community.vectorstores import FAISS

    # allow_dangerous_deserialization：FAISS 索引是 pickle，加载等于执行代码。
    # 这里加载的是**我们自己刚生成的**文件，所以可以开；从网上下来的索引绝不能开。
    store = FAISS.load_local(str(INDEX_DIR), embeddings,
                             allow_dangerous_deserialization=True)
    # 把自述挂上去，让评测脚本能打出来「这份索引是用什么切法建的」
    try:
        store.chunker_meta = json.loads(BUILD_INFO.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        store.chunker_meta = None
    return store


def describe_index(meta) -> str:
    """一行索引自述，三个评测脚本共用。**没有自述就大声说没有。**"""
    if not meta:
        return ("⚠ data/lc_index/build.json 不存在 —— 这是**老索引**或手工拷来的，"
                "不知道它是用什么切法建的。重新建一次就有自述了。")
    corp = " + ".join(f"{c['name']}({c['chars']:,}字)" for c in meta.get("corpus", []))
    return (f"索引自述：切法={meta['chunker']}（{CHUNKER_DOC.get(meta['chunker'], '?')}）"
            f"  size={meta['chunk_size']} overlap={meta['overlap']}"
            f"  {meta['block_count']} 块  建于 {meta['created_at']}\n"
            f"          语料：{corp}")


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
    ap.add_argument("--chunker", default=DEFAULT_CHUNKER, choices=list(CHUNKERS),
                    help="切法（只在 --rebuild 时生效）。"
                         + "；".join(f"{k}={v}" for k, v in CHUNKER_DOC.items()))
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    import torch
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    embeddings = get_embeddings()

    if args.rebuild or not INDEX_DIR.exists():
        store, _ = build_index(embeddings, chunker=args.chunker)
    else:
        print(f"读已有索引 {INDEX_DIR.relative_to(ROOT)}/（要重建加 --rebuild）")
        store = load_index(embeddings)
        print(describe_index(getattr(store, "chunker_meta", None)))

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
