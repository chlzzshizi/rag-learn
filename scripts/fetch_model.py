#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 ModelScope 下载 embedding 模型到 models/ 目录。

为什么不用 HuggingFace：
    这台机器上 huggingface.co 的 model.safetensors (1.19GB) 会在传输途中被
    反复重置——实测每传 5-10MB 断一次（peer closed connection）。小文件正常，
    只有那个大文件这样。hf_hub_download 会断点续传，爬到 471MB 后重试次数用尽，
    放弃并删掉了半截文件。
    同样的文件在 ModelScope（国内源）上字节数完全一致，不经过那道墙。

用法：
    .venv/Scripts/python.exe scripts/fetch_model.py
    .venv/Scripts/python.exe scripts/fetch_model.py --model BAAI/bge-m3 --dest models/bge-m3

支持断点续传：中断了直接重跑，会从已有字节接着下。
"""
import argparse
import os
import sys
import time

import httpx

CHUNK = 8 * 1024 * 1024      # 每段 8MB。实测这个粒度在这台机器上能稳定跑完。
RETRIES = 8
# 必须带 Recursive=True：默认列表只给顶层，嵌套目录（如 1_Pooling/config.json）
# 会以 type=tree 的条目出现，里面的文件一个都拿不到。
LIST_API = ("https://www.modelscope.cn/api/v1/models/{model}"
            "/repo/files?Revision=master&Recursive=True")
FILE_API = "https://www.modelscope.cn/api/v1/models/{model}/repo?Revision=master&FilePath={path}"

SKIP_SUFFIX = (".gitattributes",)


def mb(n):
    return "%.1f MB" % (n / 1024.0 / 1024.0)


def list_files(client, model):
    r = client.get(LIST_API.format(model=model), timeout=30, follow_redirects=True)
    r.raise_for_status()
    files = r.json().get("Data", {}).get("Files", [])
    out = []
    for f in files:
        if f.get("Type") != "blob":
            continue
        if f["Path"].endswith(SKIP_SUFFIX):
            continue
        out.append((f["Path"], int(f["Size"])))
    return out


def fetch(client, model, path, size, dest):
    """下载单个文件，从 dest 现有大小处续传。"""
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print("  [skip] %-28s 已完整 (%s)" % (path, mb(size)), flush=True)
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    pos = os.path.getsize(dest) if os.path.exists(dest) else 0
    if pos > size:                      # 残留的坏文件，从头来
        os.remove(dest)
        pos = 0

    url = FILE_API.format(model=model, path=path)
    t0 = time.time()
    with open(dest, "ab" if pos else "wb") as f:
        while pos < size:
            end = min(pos + CHUNK - 1, size - 1)
            last = None
            for attempt in range(1, RETRIES + 1):
                try:
                    got = 0
                    with client.stream("GET", url, follow_redirects=True,
                                       timeout=60,
                                       headers={"Range": "bytes=%d-%d" % (pos, end)}) as r:
                        r.raise_for_status()
                        for c in r.iter_bytes(256 * 1024):
                            f.write(c)
                            got += len(c)
                    if got != end - pos + 1:
                        raise IOError("段不完整: 收到 %d / 期望 %d" % (got, end - pos + 1))
                    pos += got
                    break
                except Exception as e:                 # noqa: BLE001
                    last = e
                    f.seek(pos)                            # 丢弃这段写坏的部分
                    f.truncate()
                    if attempt < RETRIES:
                        time.sleep(min(2 ** attempt, 10))
            else:
                print("  [FAIL] %-28s %s" % (path, last), flush=True)
                return False

            pct = 100.0 * pos / size
            done = pos >= size
            sys.stdout.write("\r  [%s] %-28s %6.1f%%  %8s  %5.0fs   "
                             % ("ok" if done else "dl", path, pct, mb(pos), time.time() - t0))
            sys.stdout.flush()
    sys.stdout.write("\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--dest", default=None,
                    help="默认 models/<模型名末段>")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = args.dest or os.path.join(root, "models", args.model.split("/")[-1])

    print("模型:   %s" % args.model)
    print("目标:   %s" % dest)
    print()

    with httpx.Client() as client:
        files = list_files(client, args.model)
        total = sum(s for _, s in files)
        print("共 %d 个文件，合计 %s\n" % (len(files), mb(total)))

        t0 = time.time()
        for path, size in files:
            if not fetch(client, args.model, path, size, os.path.join(dest, path)):
                print("\n失败。重跑本脚本会从断点继续。")
                return 1

    print("\n完成，耗时 %.0fs" % (time.time() - t0))
    print("在 .env 里写:  EMBEDDING_MODEL=%s" % dest.replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
