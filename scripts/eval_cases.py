#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评测题库 —— 检索评测和答案级评测**共用这一份**。

分家的代价是漂移：同一个问题在两个脚本里锚点不一样，跑出来的数就对不上了。

## 锚点是什么

每组锚点是一份**完整答案**的代表句子：

    mode="any"  集中题：任意一组凑齐 → 答得出
    mode="all"  散落题：每一组都要凑齐 → 才答得出

锚点必须是**句子级**的，不能是关键词。第一版用了 'DECR' / '分叉点' 这种词，
结果 'DECR' 落在 8 块里 —— 量出来的是词频不是答案位置，5 道"集中题"全被算成散落。

## 新增题目时注意

**锚点要从语料里"抄"，不要从脑子里"想"。** 我犯过一次：按"唯一性"挑锚点，
系统性选中了埋在代码块和目录树里的句子（散文里的正常表述因为重复出现反被筛掉），
把难度虚增了。抄原句没有这个问题，而且 `audit()` 会替你验。
"""

from __future__ import annotations

import sys

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
         groups=[["**NULL**（线上确实不知道他是谁）"],                     # §4.3 表
                 ["注册即登录，**只要手机号+密码**"]]),                     # §11.6

    dict(type="散落", mode="all",
         q="一个 token 会因为哪些原因失效？",
         groups=[["登出时 Token 加入 Redis 黑名单"],                      # §6.2 按票
                 ["auth:staff:invalidAfter:<staffId> = 当前毫秒"]]),       # §6.2 按人

    dict(type="散落", mode="all",
         q="项目里 Redis 存了哪些东西？各自的键名是什么？",
         groups=[["coupon:stock:{couponId}", "coupon:grabbed:{couponId}"],        # §11.5
                 ["blacklist:token:", "auth:staff:invalidAfter:<staffId> = 当前毫秒"]]),  # §6.2

    dict(type="散落", mode="all",
         q="「管理员不参与业务」具体体现在哪几件事上？",
         groups=[["不参与订单操作，也不能改价"],                            # §4.2
                 ["管理员 → 403「管理员不参与发券」"]]),                     # §11.6
]


# ── bug 文档的题（6 道）──────────────────────────────────────────────────
#
# bug 记录每条结构固定：`### 现象 / 根因 / 修复 / 教训 / 教训`。所以
#   **问题取自「现象」**（症状语言 = 检索时实际会用的词）
#   **锚点取自「根因 / 修复 / 教训」**（原因语言）—— 是引用，不是发明
#
# 这天然是难题：实测 6 条里 4 条「现象」和「根因」**标识符零重合**
# （现象"支付报 500" vs 根因"`save()` 只会 INSERT 不会 UPDATE"）。
# 查询用的是症状词，答案躺在原因词里 —— 这一跳正是要量的东西。
#
# ## 选题规则（谁选的、怎么选的，写清楚）
#
# 原本打算「每 10 条取 1 条」，**实测被否决**：抽出的 5 条里只有 2 条像样的题
# —— Bug 21/31/41 是测试脚本元 bug（"bench-coupon.sh 量的是 JVM 冷启动"），
# 没人会这么问；Bug 41 的现象首句直接是段 bash 代码。
# **文档后半段从领域 bug 漂移成了测试工具 bug。**
#
# 改成规则抽候选、**用户拍板**。用户选了「订单/状态机、并发/数据一致、权限/安全」
# 三组各两条；**具体编号用机械规则定：每组取列出的前两条** → 1、2 / 10、11 / 16、20。
# 这个规则里没有我的判断 —— 换个执行者会抽出同一批编号。
#
# ## ⚠️ 两条带「口径修订」的题：只锚在没被作废的节上
#
# Bug 16 和 Bug 20 的`修复`/`验收`节都被后续口径修订**部分作废**：
#   * Bug 16 的「跨店 403 / total=0」两行已作废（所有店长可管所有门店订单）
#   * Bug 20 的修复代码 `requireStaffStore` 已成**死代码**（管理员订单权限
#     先被提为同级、2026-09-11 又被整条收回），原文本人标注"这就是本条最讽刺的地方"
#
# 所以这两题的锚点**只取`根因`和`教训`** —— 那两节没被作废，且修订说明里
# 明确写了"教训不变"。锚在作废段落上会拿语料自己说已过时的话去判分。
#
# 锚点全部经 `audit()` 同款方式**逐字验证存在**（25/25 通过），不是从脑子里想的。
BUG_CASES: list[dict] = [
    # ── 集中：一份解释就够，`根因`和`教训`各是一份等价答案 ──
    dict(type="集中", mode="any",
         q="创建订单成功，为什么一调用支付接口就报 500？",
         groups=[["`OrderRepositoryImpl.save()` 写死了 `orderMapper.insert()`",
                  "MySQL 主键冲突 → 500"],                        # 根因
                 ["`save()` 方法必须同时处理",                          # 教训
                  "判断依据：主键 id 是否为 null"]]),

    dict(type="集中", mode="any",
         q="一个订单明明只有一条明细，操作几次后查出来却有 32 条，为什么？",
         groups=[["`save()` 里的 `insertBatch` 没有区分新建和更新",   # 根因
                  "每次 save 都插一次"],
                 ["用 `isNew` 变量控制，只在新建时插入明细"]]),      # 修复

    dict(type="集中", mode="any",
         q="抢券的库存键会变成负数脏数据，之后补货时剩余量算不对，为什么？",
         groups=[["`decrement` 和 `increment` 是 Redis 上两条独立命令",   # 根因
                  "加回的数量无法精确匹配"],
                 ["去掉加回逻辑。`remaining < 0` 直接返回",                # 修复
                  "补偿操作本身也有竞态"]]),                              # 教训

    # ── 散落：两个不同的子答案，**缺一个就答不完整**，所以 mode=all ──
    dict(type="散落", mode="all",
         q="同一顾客快速点两次抢券，第二次返回 500，而且他以后再也抢不了券，为什么？",
         groups=[["撞 `uk_coupon_customer` 唯一键",                    # 为什么 500
                  "全局异常兜底返回"],
                 ["写操作顺序错误：先写缓存（SADD）后写数据库（insert）",  # 为什么永远抢不了
                  "数据库失败时缓存留下脏标记"]]),

    # ── 集中 ──
    dict(type="集中", mode="any",
         q="为什么顾客拿自己的 token 能推进别人的订单？",
         groups=[["身份信息在 `JwtInterceptor` 里已经解析好并放进了 request attribute",  # 根因
                  "订单控制器**一个都没取**"],
                 ["每个接口都要问两遍",                            # 教训（未作废）
                  "身份只能来自 token"]]),

    dict(type="集中", mode="any",
         q="用 admin 账号登录明明成功了，紧接着调用订单接口却报「登录信息已升级，"
           "请重新登录」，重登一百次也一样，为什么？",
         groups=[["`storeId` 为 `null` 有**两种完全不同的原因**",     # 根因
                  "把管理员也归进了"],
                 ["**`null` 是有歧义的**",                            # 教训（未作废）
                  "合并处理就会给出误导性的错误"]]),
]


def bug_candidates():
    """按规则抽出每条 bug 的（序号、标题、现象首句），供挑选。

    全是机械抽取，没有一处靠判断。
    """
    import re
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    import rag

    text = rag.CORPUSES[1].read_text(encoding="utf-8")
    out = []
    for i, blk in enumerate(re.split(r"(?m)^## Bug ", text)[1:], 1):
        title = blk.split("\n")[0].strip()
        m = re.search(r"### 现象\s*\n+(.{0,90})", blk, re.S)
        symptom = m.group(1).replace("\n", " ").strip() if m else "(无现象节)"
        out.append((i, title, symptom))
    return out


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")
    for i, title, symptom in bug_candidates():
        print(f"\n[Bug {i:>2}] {title}")
        print(f"         现象 → {symptom}")
