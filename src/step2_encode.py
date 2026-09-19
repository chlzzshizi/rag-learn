#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 2 步：编码落盘。

把第 1 步切出来的块，一块一块送进 embedding 模型，变成向量存下来。

这一步是整个 RAG 里唯一"不可逆的压缩"——一块 500 字的文本进去，
出来是 1024 个数。**进去之后就再也拿不回来了**，所以：

  * 原文要**另外存一份**（chunks.jsonl），检索命中后要拿原文喂给生成模型。
    向量只能用来"找"，不能用来"读"。
  * 编码质量取决于第 1 步切得好不好。块切烂了，向量就是烂块的向量，
    后面检索再准也没用 —— 这就是第 1 步那句话的意思：
    "检索的问题 90% 出在切块"。

产物（都在 data/，已 git 忽略）：

    chunks.jsonl    每行一块：i / start / end / text / chunk_hash / source_file
    vectors.npy     float32 矩阵，形状 (N, 1024)
    manifest.json   这次索引的元信息：语料指纹、模型、维度、块数

跑法：

    .venv/Scripts/python.exe src/step2_encode.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step1_chunk import CORPUS, CHUNK_SIZE, chunk_text, load_text  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 和 .env 里 EMBEDDING_MODEL 一致。这里直接写死相对路径，省得引 dotenv——
# 这一步用不到任何密钥，能少一个依赖就少一个。
MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"

# 查询端要加指令前缀，文档端不加（见 README 第 2 步的说明）。
# v1 先不加，第 3 步拿它做对比实验。写在这儿是为了别忘了这回事。
QUERY_INSTRUCTION = "Instruct: 给定用户问题，检索能回答该问题的文档\nQuery: "


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model():
    """加载模型。**必须显式 float32**，理由见 README「技术选型」那一节。

    不传 model_kwargs 的话 SentenceTransformer 会按模型自带的 dtype 读成
    bfloat16 —— 尾数只有 8 位，归一化后范数在 0.998~1.002 之间飘，
    "归一化后余弦 = 点积" 就只是近似成立。实测 6 个样本里 2 个排序会翻。
    代价是慢一倍，你 148 个块是 7 秒 vs 14 秒的事，不值当省。
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if not MODEL_DIR.exists():
        sys.exit(f"找不到模型：{MODEL_DIR}\n先从 ModelScope 下：python scripts/fetch_model.py")

    print(f"加载模型 {MODEL_DIR.name}（float32）…")
    model = SentenceTransformer(str(MODEL_DIR), model_kwargs={"dtype": torch.float32})
    # 6.0.1 起旧名 get_sentence_embedding_dimension 已弃用，会打 FutureWarning。
    # getattr 兜底是为了换版本时不至于直接崩。
    dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    print(f"  维度 {dim()} / 最大序列 {model.max_seq_length} token")
    return model


def main() -> None:
    DATA.mkdir(exist_ok=True)

    # ---- 1. 切块（复用第 1 步，保证两步用的是同一套切法）----------------
    text = load_text(CORPUS)
    chunks = chunk_text(text, CHUNK_SIZE)
    texts = [c["text"] for c in chunks]
    print(f"语料 {len(text):,} 字符 → {len(chunks)} 块")

    # ---- 2. 编码 --------------------------------------------------------
    model = load_model()

    # normalize_embeddings=True：这个模型管线末尾自带 Normalize，本来就是归一化的，
    # 实测开不开逐位相同（README 有记录）。写上是为了换模型时不至于忘了。
    # batch_size=1 —— 别改成 16，实测 CPU 上批处理反而**更慢**。
    # 交替测 3 轮，每轮 16 块（4,218 token），排除顺序干扰：
    #     batch_size=16   20.06 / 20.92 / 21.49s
    #     batch_size=1    14.65 / 15.13 / 15.04s      ← 纯编码快约 30%
    # 原因：CPU 上批处理没有并行收益，却要按批内最长序列补齐、张量更大，
    # 注意力又是 O(n²)。（2026-09-19 实测。换 GPU 的话结论会反过来，重新量。）
    #
    # 但**端到端没有 30%**：全脚本 188s/202s → 164s，只快约 15%。
    # 因为模型加载那 ~15s 是固定开销，把比例稀释了。
    # 教训：微基准的比例不能直接当成端到端收益。
    #
    # batch_size 不影响输出：两种设置下范数极差都是 2.38e-07，逐位相同。
    vectors = model.encode(
        texts,
        batch_size=1,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # ---- 3. 自检：范数必须精确等于 1 --------------------------------------
    # 这不是走过场 —— 它直接验证 float32 那个决定生效了。
    # bfloat16 下这里的极差会是 1e-3 量级，float32 下是 1e-7。
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1)
    spread = float(norms.max() - norms.min())
    print(f"\n向量 {vectors.shape}  dtype={vectors.dtype}")
    print(f"范数 最小 {norms.min():.8f} / 最大 {norms.max():.8f} / 极差 {spread:.2e}", end="  ")
    if spread < 1e-5:
        print("✓ float32 生效")
    else:
        print("✗ 范数没落在 1 附近 —— 八成又读成 bfloat16 了")

    assert vectors.shape == (len(chunks), 1024), f"形状不对：{vectors.shape}"

    # ---- 4. 落盘 --------------------------------------------------------
    src_sha = sha256_file(CORPUS)

    with (DATA / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "i": c["i"],
                "start": c["start"],
                "end": c["end"],
                "text": c["text"],
                # chunk_hash 现在没用，是给增量更新留的位置：
                # 文档改了之后，重算每块的 hash，只有变了的块需要重新编码。
                # 没有它就只能全量重编码。
                "chunk_hash": sha256_text(c["text"]),
                "source_file": CORPUS.name,
            }, ensure_ascii=False) + "\n")

    np.save(DATA / "vectors.npy", vectors.astype(np.float32))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(CORPUS),
        "source_sha256": src_sha,          # 语料指纹：变了就说明该重建索引
        "source_chars": len(text),
        "chunk_size": CHUNK_SIZE,
        "chunk_count": len(chunks),
        "model": MODEL_DIR.name,
        "embedding_dim": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "normalized": True,
        "query_instruction": None,          # v2 再加，见 README
    }
    (DATA / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n写好了：")
    for name in ("chunks.jsonl", "vectors.npy", "manifest.json"):
        print(f"  data/{name:<16} {(DATA / name).stat().st_size:>12,} 字节")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
