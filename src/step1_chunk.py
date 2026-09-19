#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第 1 步：切块。

把设计文档按固定长度切成小块，打印块数和切口预览，供人工翻看。

设计取舍（写在这儿，免得三个月后忘了当初为什么这么选）：

  * **固定长度、不重叠** —— 先要一个"最笨但完全可复现"的基线。
    语义切块（按标题切、按段落切）是后面的事；没有基线，就没法说
    "语义切块到底好在哪"。先量出来，再改进。

  * **500 字符** —— 这个数不是被模型逼的（Qwen3 上下文 32K token，
    500 字符连零头都不到），是检索粒度问题：块太大，一个向量里混进
    好几个主题，检索时会"沾边就中"；块太小，一个完整的意思被拆散。
    500 是拍脑袋的第一版，第 3 步检索不中时会回来改它。

  * **不复制语料** —— 直接读 Desktop 原路径。你写 yunxi 时会不断改这份
    文档，重跑脚本就能顺便练到增量更新。

跑法：

    .venv/Scripts/python.exe src/step1_chunk.py              # 看统计
    .venv/Scripts/python.exe src/step1_chunk.py --preview    # 翻全部切口
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 语料。改这里，或者用 --corpus 临时指定别的文件。
CORPUS = Path(r"C:\Users\33705\Desktop\项目\2026-07-04-yunxi-redesign.md")

CHUNK_SIZE = 500

# 预览每行两侧各显示多少字符。太长了终端会折行，折了反而看不清切口。
PREVIEW_WIDTH = 34


def load_text(path: Path) -> str:
    """读语料。

    必须显式 encoding="utf-8"：Windows 上 open() 默认走系统 ANSI 代码页
    （简体中文机器上是 GBK），这个文档里有大量非 GBK 能表示的字符，
    不写 encoding 会直接 UnicodeDecodeError —— 或者更糟，静默读成乱码。
    """
    if not path.exists():
        sys.exit(f"找不到语料：{path}\n改 step1_chunk.py 里的 CORPUS，或用 --corpus 指定。")
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[dict]:
    """按固定长度切，不重叠。

    range(0, len(text), size) 的最后一段会自动短一截（不足 size 就切到末尾），
    所以块数 = ceil(len(text) / size)，不是 floor。这一点是第 1 步对账的关键：
    如果你数出来跟这个对不上，就是循环写错了。

    返回的每条都带 start/end 偏移量。现在看着没用，但增量更新时要用它
    报"这次改动影响哪几块"—— 所以从一开始就带上，别等需要了再回头补。

    这个函数是给第 2 步 import 的，所以保持纯粹：不打印、不读写文件。
    """
    return [
        {
            "i": i // size,
            "start": i,
            "end": min(i + size, len(text)),
            "text": text[i : i + size],
        }
        for i in range(0, len(text), size)
    ]


def flatten(s: str) -> str:
    """把换行、制表符换成可见符号，只为了一行内显示。

    markdown 的块里全是换行，不拍扁的话每个块要占十几行，
    148 个块变成两千行，根本没法扫。⏎ 是 U+23CE。
    """
    return s.replace("\r\n", "\n").replace("\n", "⏎").replace("\t", "→")


def preview(chunks: list[dict]) -> None:
    """打印全部切口，一行一块。

    看的方法：**竖着扫**。第 i 行的右半边（尾）和第 i+1 行的左半边（头）
    接起来，就是第 i 个切口 —— 那两截读不读得通，就是它切没切坏。
    """
    print(f"{'#':>4}  {'偏移':>13}   {'← 块头':<{PREVIEW_WIDTH}} ‖ {'块尾 →'}")
    print("-" * (4 + 2 + 13 + 3 + PREVIEW_WIDTH + 3 + PREVIEW_WIDTH))
    for c in chunks:
        head = flatten(c["text"][:PREVIEW_WIDTH])
        tail = flatten(c["text"][-PREVIEW_WIDTH:])
        print(f"{c['i']:>4}  {c['start']:>6}-{c['end']:<6}   {head:<{PREVIEW_WIDTH}} ‖ {tail}")


def main() -> None:
    ap = argparse.ArgumentParser(description="第 1 步：按固定长度切块")
    ap.add_argument("--corpus", type=Path, default=CORPUS, help=f"语料路径（默认 {CORPUS}）")
    ap.add_argument("--size", type=int, default=CHUNK_SIZE, help=f"块长，字符（默认 {CHUNK_SIZE}）")
    ap.add_argument("--preview", action="store_true", help="打印全部切口，供人工翻看")
    ap.add_argument("--show", type=int, default=2, help="不带 --preview 时，抽样打印前几块全文")
    args = ap.parse_args()

    text = load_text(args.corpus)
    chunks = chunk_text(text, args.size)

    # ---- 统计 ----------------------------------------------------------
    # 理论块数用 ceil 算，而不是把"142-150"这种区间写死在代码里。
    # 文档每天都在长，硬编码的区间第二天就过期；算出来的永远不会。
    # （README 里那个 142-150 是 72,819 字符时的余量，这里是它的推广。）
    theory = -(-len(text) // args.size)  # ceil，等价于 math.ceil(len/size)

    print(f"语料   {args.corpus}")
    print(f"        {len(text):,} 字符 / {args.corpus.stat().st_size:,} 字节 / {text.count(chr(10)) + 1:,} 行")
    print(f"        （中英文别混：os.path.getsize 给的是字节，中文一个字占 2-3 字节）")
    print()
    print(f"块长    {args.size} 字符，不重叠")
    print(f"理论块数 ceil({len(text):,} / {args.size}) = {theory}")
    print(f"实际块数 {len(chunks)}" + ("   ✓ 与理论一致，循环是对的" if len(chunks) == theory else "   ✗ 对不上，循环写错了，先查这个再往下走"))
    print()

    lengths = [c["end"] - c["start"] for c in chunks]
    short = [c for c in chunks if c["end"] - c["start"] < args.size]
    print(f"块长分布 最短 {min(lengths)} / 最长 {max(lengths)} / 平均 {sum(lengths) / len(lengths):.1f}")
    print(f"        {len(short)} 块不满 {args.size}（只有最后一块不满是正常的）")
    print()

    # ---- 抽样 ----------------------------------------------------------
    if args.show:
        print(f"---- 前 {args.show} 块全文 " + "-" * 40)
        for c in chunks[: args.show]:
            print(f"\n[#{c['i']}  {c['start']}-{c['end']}]")
            print(c["text"])
        print()

    # ---- 预览 ----------------------------------------------------------
    if args.preview:
        print("---- 全部切口 " + "-" * 48)
        print("竖着扫：第 i 行右半边 + 第 i+1 行左半边 = 第 i 个切口。\n")
        preview(chunks)
    else:
        print(f"要看全部 {len(chunks)} 个切口：加 --preview")


if __name__ == "__main__":
    # Windows 上 Python 默认按系统代码页（GBK）写 stdout，打印中文会
    # UnicodeEncodeError 或者变成乱码。强制 UTF-8 —— 尤其是输出被重定向到
    # 文件或管道的时候（终端里有时看着正常，一重定向就炸）。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
