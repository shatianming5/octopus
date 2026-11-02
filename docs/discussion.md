# Octopus 设计讨论：可分叉推理状态与并行 Agent Runtime

**整理日期：2026-09-08。来源：用户提供的完整讨论文本。**

> 这是项目开始前的设计讨论，保持原样，不随实现更新。文中"尚未创建""暂不碰 LoRA/RL"等说的是 2026-09-08 的状态；现在实现了什么、核实了什么以 [README.md](../README.md) 和代码为准。

本文按主题重组原讨论，保留架构、接口草案、研究问题、评价方案、工作定位及风险。重复观点合并；原讨论尚未解决的冲突单独列出，不擅自变成已确定规范。[原始讨论全文](discussion-original.txt) 保留全部措辞与示例。

> **状态声明**：这是设计记录，不是实现报告、文献查新报告或实验报告。代码均为接口/伪代码草案；所有性能数字、成功率、评分和“预计最好”的策略都没有实验验证。本文补充的技术边界用于避免误读，不代表已实施的功能。

## 目录

1. [问题、定位与核心主张](#1-问题定位与核心主张)
2. [计算模型与状态拓扑](#2-计算模型与状态拓扑)
3. [核心对象与接口草案](#3-核心对象与接口草案)
4. [KV 共享、partial block 与生命周期](#4-kv-共享partial-block-与生命周期)
5. [Branch 状态机与资源调度](#5-branch-状态机与资源调度)
6. [Semantic Commit 与 WorldState](#6-semantic-commit-与-worldstate)
7. [RQ1：何时 checkpoint 与 fork](#7-rq1何时-checkpoint-与-fork)
8. [RQ2：分支应该继承多少状态](#8-rq2分支应该继承多少状态)
9. [三层 Benchmark](#9-三层-benchmark)
10. [Baseline 与公平比较](#10-baseline-与公平比较)
11. [消融与核心指标](#11-消融与核心指标)
12. [优势如何呈现与示例结果](#12-优势如何呈现与示例结果)
13. [实现边界、模块与推进顺序](#13-实现边界模块与推进顺序)
14. [创新性、风险与工作评价](#14-创新性风险与工作评价)
15. [设计状态、待决问题与来源说明](#15-设计状态待决问题与来源说明)

## 1. 问题、定位与核心主张

### 1.1 要解决的真实问题

多个 Agent 明明依赖大量相同信息，却反复复制、总结、发送和重新处理上下文。常见执行形式是：

```text
Agent A -> message -> Agent B -> summary -> Agent C -> Orchestrator
```

Octopus 的提议是从共享认知主干产生临时执行分支，再返回语义增量：

```text
                  Shared cognitive trunk
                     /      |      \
                    A       B       C
                     \      |      /
                       Semantic deltas
```

| 维度 | 传统独立 Multi-Agent | Octopus 提议 |
|---|---|---|
| 执行单元 | 新建独立 Agent/context | 从现有推理状态 fork 临时 Branch |
| 共享信息 | 消息、完整上下文复制或 summary | 不可变共享 KV prefix + 任务 capsule |
| 分支轨迹 | 各自完整上下文 | 共享主干 + 私有 suffix |
| 结果回流 | 消息传递 | 结构化 Semantic Commit，经 reducer 更新 WorldState |
| 资源生命周期 | 常以 request/agent 为中心 | 显式 checkpoint、fork、abort、release、GC |

这是一组目标语义，不表示所有现有框架都没有上述能力；与具体系统的差异仍需要文献和实现层面的比较。

### 1.2 论文与系统定位

优先定位为 **“An inference runtime for branching agentic reasoning”**，而不是泛化的 “A New Multi-Agent Framework”。

候选标题：**Octopus: Forkable Inference State for Parallel Agentic Reasoning**。

叙事顺序：LLM inference → long-running agent → parallel exploration → context replication problem → Octopus。

原讨论建议的主 claim：

> At equal task quality, Octopus substantially reduces redundant context processing and coordination overhead.

这是**待验证的目标**，不是已经成立的结论。更准确的研究 thesis 是：

> Octopus reduces the cost of parallel agentic reasoning by turning context replication into shared inference state + semantic deltas.

首轮比较原则：**Same model. Same weights. Same task. Different runtime.** 不把模型训练带来的收益混入运行时收益。

## 2. 计算模型与状态拓扑

### 2.1 三个核心定义

$$
\text{Branch} = \text{Immutable Shared KV Prefix} + \text{Mutable Private KV Suffix}
$$

$$
\text{Checkpoint} \ne \text{KV Copy},\qquad
\text{Checkpoint} = \text{Immutable KV View}
$$

$$
\text{Commit} \ne \text{KV Merge},\qquad
\text{Commit}: \text{Private Neural State} \rightarrow \text{Semantic Delta}
$$

原讨论用下式表达主干演进：

$$
C_{t+1} = C_t + \operatorname{Reduce}(\Delta_1,\Delta_2,\ldots,\Delta_n)
$$

这里的 `+` 是逻辑上的状态更新：Brain 将 reducer 产生的增量作为后续输入继续推理，形成新的 token/KV 状态；**不是把两个 KV tensor 相加或拼接**。

### 2.2 一轮执行如何收敛

```text
                         C0
                  Brain checkpoint
                  /      |       \
                 A       B        C
              4k tok   2k tok    6k tok
                 |       |        |
              CommitA CommitB  CommitC
                  \      |      /
                    State Reducer
                         |
                  ~300-token delta
                         |
               WorldState / Brain append
                         |
                         C1
                    /         \
                   D           E
```

以上 token 数仅为示意。核心描述是：**逻辑推理轨迹按树分叉，语义历史不断收敛到 Brain 主干。** 物理 KV 存储共享页面，不等于一棵不共享节点的物理内存树。

例如 A 在 C0 后探索 4,000 tokens，B 探索 7,000 tokens；返回的 Commit 可能只有 120 和 160 tokens。Brain 消费短增量，而不是吸收两条完整私有轨迹。压缩是否保留足够信息需要任务质量评价。

## 3. 核心对象与接口草案

### 3.1 四种一级对象

```text
OctopusSession
  |-- Checkpoint
  |     |-- Branch A
  |     |-- Branch B
  |     `-- Branch C
  `-- WorldState
```

| 对象 | 职责 |
|---|---|
| `OctopusSession` | 持有 Brain、当前 checkpoint、共享语义状态和活跃分支 |
| `Checkpoint` | 某一合法推理位置的不可变 KV 视图与对应逻辑状态 |
| `Branch` | 从 checkpoint 继承共享前缀，拥有任务与私有执行轨迹 |
| `WorldState` | reducer 管理的结构化事实、产物和未决问题；具体 schema 尚未定义 |

原讨论的数据结构整理如下。类型定义展示设计字段，不是可直接运行的完整实现：

```python
@dataclass
class OctopusSession:
    session_id: str
    brain_branch_id: str
    current_checkpoint_id: str
    world_state: "WorldState"
    active_branches: dict[str, "Branch"]


@dataclass
class Checkpoint:
    checkpoint_id: str
    token_ids: list[int]
    num_tokens: int
    block_ids: list[int]
    next_position: int
    model_id: str
    ref_count: int
    sealed: bool


@dataclass
class Branch:
    branch_id: str
    parent_checkpoint_id: str
    shared_blocks: list[int]
    private_blocks: list[int]
    task: "Task"
    generated_tokens: list[int]
    status: "BranchStatus"
    created_at: float
    last_active_at: float


@dataclass
class KVBlock:
    block_id: int
    gpu_slot: int
    ref_count: int
    num_filled_tokens: int
    immutable: bool
    content_hash: bytes | None
    last_access_time: float
```

`block_ids` 只是引用，例如 `[17, 31, 42, 99]`；复制这个列表不是复制 GPU 上的 KV tensor。`shared_blocks` 与 `private_blocks` 的区分服务于隔离和释放；但块被后续 checkpoint/branch 共享后，其所有权必须反映真实状态，不能仅凭旧的 “private” 标签释放。

`content_hash` 是原草案字段，不代表另建一套 hash 机制的决定。若复用 vLLM，应先使用后端已有的 block/cache 标识与管理能力。逻辑对象也不等于必须重复实现一份物理 block pool。

### 3.2 KV Manager API

```python
class OctopusKVManager:
    def checkpoint(self, branch_id: str) -> Checkpoint: ...

    def fork(self, checkpoint_id: str, n: int = 1) -> list[str]: ...

    def allocate(self, branch_id: str, num_tokens: int) -> list[int]: ...

    def seal(self, branch_id: str) -> Checkpoint: ...

    def abort(self, branch_id: str) -> None: ...

    def release_checkpoint(self, checkpoint_id: str) -> None: ...

    def gc(self) -> None: ...
```

`checkpoint()` 与 `seal()` 的区别、checkpoint 本身持有什么所有权，以及 `release_checkpoint()` 释放哪一层引用，都还是实现前需要收敛的 API 语义。

`fork()` 的关键动作：

```text
cp = checkpoints[checkpoint_id]
for each requested branch:
    create branch metadata
    shared_blocks = copy(cp.block_ids)
    private_blocks = []
    generated_tokens = []
    status = RUNNABLE
    acquire references to the shared blocks
return branch identifiers
```

这描述了“共享 full-block KV，不复制其 tensor”的目标。复制引用、更新元数据和调度依然有成本；不能把 zero-copy 写成零时延或严格 O(1) fork。

### 3.3 高层用户 API

```python
octopus = Octopus(model="...")
brain = await octopus.create_session()

await brain.run("Investigate why authentication is failing.")
cp = brain.checkpoint()

branches = await cp.fork([
    InspectCodeTask(),
    InspectLogsTask(),
    SearchDocsTask(),
])

results = await octopus.gather(branches)
await brain.commit(results)
answer = await brain.continue_()
```

这是易用性草案。高层 `cp.fork(tasks)` 与底层 `fork(checkpoint_id, n)` 是不同层次，需要在实现中接通。`gather()` 展示等待一组结果的最简单用法，不意味着系统必须总等全部分支结束；原讨论也希望支持异步提交与提前取消，具体消费 API 尚未确定。

底层流程：`create_session → allocate KV → prefill → checkpoint → freeze/acquire shared blocks → fork requests → allocate private pages → asynchronous decode → semantic commit → release branch references → append delta to Brain`。

## 4. KV 共享、partial block 与生命周期

### 4.1 为什么 checkpoint 必须 immutable

Brain 已处理 `A B C D E F`，映射为 `[P0][P1][P2]`。C0 建立后，其表示的前缀不能被任何后续分支改写：

```text
                   C0: [P0][P1][P2]
                       /    |    \
                      A     B     C

A: [P0][P1][P2][A0]
B: [P0][P1][P2][B0][B1]
C: [P0][P1][P2][C0-private]
```

共享的是前缀页面；新 token 写入分支自己的后缀。原文以 Brain + 三个 Branch 演示 `P0.ref = P1.ref = P2.ref = 4`。**如果 Checkpoint 也独立持有页面引用，则还需要计入它；“4”只是所选所有权模型下的示例，不是可直接照搬的规范。**

原提议的重要约束是 `ref_count > 1` 时不可原地修改；另外，只要一个块仍属于不可变 checkpoint，即便某时刻引用数为 1，也不能破坏该 checkpoint 的视图。

### 4.2 v0.1：先限定完整 block 的 fork

设 block size = 16，尾页只有 9 tokens。若 A、B 共享这张 partial page，并同时写第 10 个位置，就会冲突。

原讨论的工程简化是：**v0.1 checkpoint 仅建立在 block boundary，例如 16、32、48、64 tokens。** full blocks 共享，新增 token 从分支私有页面开始。

这不是说所有真实任务天然对齐。语义边界与物理 block 边界通常不重合，因此该限制会约束首版支持的 fork 点。未对齐时如何处理仍待设计；不能为凑边界擅自截掉有效 token 或添加会改变推理语义的 padding。

### 4.3 v0.2：partial-block copy-on-write

后续候选方案：

```text
Shared: [P0][partial P1]

A:      [P0][copy(P1) -> A1]
B:      [P0][copy(P1) -> B1]
```

完整页面 zero-copy，尾部 partial page 每个分支复制一份。原讨论的额外成本估计为最多 `1 KV block × branch count`，这里的一个 block 指相应逻辑页对应的 KV 存储；真实字节数由后端、模型层数、KV dtype 和布局决定。它只界定 fork 时尾页复制，不包括后续生成的私有 KV。

### 4.4 abort 与 GC

原 abort 示意：

```text
Branch A shared:  P0 P1 P2 P3
Branch A private: A4 A5 A6

abort:
    release this branch's shared references
    release its private blocks when no live owner remains
    remove branch metadata
```

遍历式释放复杂度约为 `O(number of referenced blocks)`。真实释放还必须遵守后端对在途 decode 的生命周期，不能在 GPU 仍使用页面时立即复用。原文“直接 free 所有 private blocks”的伪代码，仅适用于这些块确实没有其他持有者的情况。

| 状态 | 含义 | 处理 |
|---|---|---|
| HOT / ACTIVE | 活跃 request/branch 或被有效 checkpoint 保留的页面 | 不可当作空闲页复用 |
| WARM | 没有必须保留的活跃引用，但还留有可复用的 prefix 内容 | 可由后端缓存保留；内存压力下驱逐 |
| FREE | 可立即分配的物理容量 | 供后续请求使用 |

原讨论希望通过 LRU 式缓存保留已不活跃的前缀，方便回到旧计划；这与“checkpoint 保证可 fork”不能混为一谈。**若一个 checkpoint 仍承诺指向有效 KV，其页面不能像普通无引用 cache 一样随意驱逐；若允许驱逐，则恢复/失效语义必须明确。** 现阶段未选择持久化或恢复方案。

## 5. Branch 状态机与资源调度

### 5.1 原始状态草案

原文枚举：

```python
class BranchStatus(Enum):
    CREATED = "created"
    RUNNABLE = "runnable"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    COMMIT_READY = "commit_ready"
    COMMITTED = "committed"
    ABORTED = "aborted"
    GC = "gc"
```

概念流程为创建 → 可运行 → 运行 → 等待工具/完成计算 → 可提交 → 提交或放弃 → 资源回收。

**原文存在两个尚未解决的差异**：状态图出现 `FINISHED`，枚举却没有；状态图把 `WAITING_TOOL` 直接接到 `COMMIT_READY`，但工具返回后也可能需要恢复模型推理。本文不将其伪装成已经严格定义的状态机。还需明确运行中的取消、pause/evict/resume 和终止后的资源处理语义。

### 5.2 Scheduler 只负责资源，不直接理解任务语义

```python
class BranchScheduler:
    def submit(self, branch): ...
    def choose_next_batch(self): ...
    def pause(self, branch_id): ...
    def resume(self, branch_id): ...
    def cancel(self, branch_id): ...


@dataclass
class ResourceBudget:
    max_new_tokens: int
    max_kv_blocks: int
    priority: float
    deadline: float | None
```

调度输入包括 GPU KV budget、token budget、priority、deadline、tool waiting 和 dependency。Brain/策略层负责判断任务价值，Scheduler 使用这些已提供的信号分配资源。

示例：A/B/C 的优先级分别为 0.9/0.4/0.7；GPU KV 容量紧张时，优先保留 A、C，对 B 暂停或考虑驱逐。这是策略示意，不是已经指定的调度算法。暂停并不自动释放 KV；驱逐涉及后续恢复成本，不能等同于廉价 pause。

### 5.3 Branch value 与提前取消

原讨论提出未来可使用：

$$
V(b) = P(\text{useful result}) \times \text{Impact} - \text{ComputeCost}
$$

开出 10 个触手，不意味着必须等 10 个全部返回。若某分支方向错误、被其他结果覆盖或继续计算的价值很低，可以取消，释放不再需要的私有 KV。

这一点意在区别于固定 `spawn N → wait all`。概率、Impact、成本的估计方法及量纲尚未定义；v0.1 不因此引入额外价值模型或训练任务。

## 6. Semantic Commit 与 WorldState

### 6.1 Commit 协议

自由文本不作为唯一主格式；可以保留 `summary`，但核心结果应可由 reducer 处理：

```python
@dataclass
class Evidence:
    source: str
    locator: str | None
    excerpt: str | None


@dataclass
class Fact:
    id: str
    predicate: str
    value: Any
    confidence: float
    evidence: list[Evidence]


@dataclass
class Commit:
    commit_id: str
    branch_id: str
    summary: str
    facts: list[Fact]
    artifacts: list[str]
    actions_taken: list[str]
    unresolved: list[str]
    success: bool
```

一个补齐标识字段的结构示例：

```json
{
  "commit_id": "commit-a",
  "branch_id": "branch-code",
  "summary": "Found a likely authentication issue.",
  "facts": [
    {
      "id": "fact-cookie-samesite",
      "predicate": "auth.cookie.samesite",
      "value": "Lax",
      "confidence": 0.98,
      "evidence": [
        {
          "source": "repo",
          "locator": "auth/cookie.py:42",
          "excerpt": null
        }
      ]
    }
  ],
  "artifacts": ["auth/cookie.py:42"],
  "actions_taken": [],
  "unresolved": ["Confirm whether this setting causes the observed failure."],
  "success": true
}
```

示例中的路径、认证问题与 confidence 都是虚构演示；找到一个配置值不等于证明故障原因，模型给出的 confidence 也不等于经校准的正确率。`success` 应说明任务完成语义，不应被 reducer 当成事实真伪证明。

### 6.2 提交后的真实路径

```text
Branch private trajectory
       |
semantic compression
       |
     Commit
       |
  State Reducer
       |
   WorldState
       |
Brain receives delta
       |
Brain appends tokens and computes their KV
       |
next checkpoint
```

Branch 私有轨迹不会自动成为 Brain 的 KV。commit 生成、reducer 处理和 Brain 消费增量都有成本，实验必须计入，不能只统计 fork 节省的 prefill。

reducer 如何处理重复事实、互相矛盾的结果、过时的分支发现、artifact 引用及提交顺序，原讨论没有给出实现。本文保留为待决问题，不额外引入通用事务框架。

## 7. RQ1：何时 checkpoint 与 fork

$$
RQ1:\quad \text{When should the Brain checkpoint/fork?}
$$

### 7.1 Checkpoint：语义边界 + 周期边界

原建议不是每个 token 做 checkpoint，也不是只按固定时间，而是混合：

$$
\text{CheckpointCandidates} = \text{SemanticBoundary} \cup \text{PeriodicBoundary}
$$

| 候选点 | 共享内容或作用 |
|---|---|
| User turn end / prompt boundary | 一轮共同输入已确定 |
| Brain 完成 reasoning/planning | 问题理解、全局约束、当前判断和计划已形成 |
| Task graph stabilized | 分解已稳定、外部探索尚未开始，适合产生多个分支 |
| 每 8k / 16k / 32k tokens | 长轨迹中保留较近的候选点，减少只能从早期状态重新开始的代价 |

典型例子：用户要求修复登录问题，Brain 决定看代码、查日志、查文档；在分解完成、三个探索尚未开始时 checkpoint。

候选消融：A 固定 token 间隔；B turn boundary；C planning boundary；D semantic + periodic。原讨论预测 D 最好，**尚无证据**。这些是策略候选点，还必须满足第 4 节中后端支持的物理 checkpoint 条件。

### 7.2 Fork：识别 Task DAG 的 independent frontier

不能只因为“可以并行”就无条件 fork。原讨论提出：

$$
U_{\mathrm{fork}} = P_{\mathrm{parallel}} \times V_{\mathrm{information}}
- C_{\mathrm{compute}} - C_{\mathrm{merge}} - C_{\mathrm{contention}}
$$

含义是综合任务独立性、信息互补性、计算成本、合并成本和资源竞争。它是决策框架，不是已经可执行的数值算法；各项估计与标定方法待定。

```text
                 diagnose
                /    |    \
             code   logs   docs
                \    |    /
                 hypothesis
                     |
                    patch
                     |
                    test
```

查代码、查日志、查官方文档可以位于同一 independent frontier；修改代码后运行修改后的 tests 存在依赖，不能当成等价的独立 fork。

首轮最需要回答的是：哪些已有边界足以带来收益，而不是立即训练一个复杂决策器。

## 8. RQ2：分支应该继承多少状态

$$
RQ2:\quad \text{How much of the Brain state should each branch inherit?}
$$

更深层的问题是：**What is the minimum cognitive state required to spawn a capable branch?**

### 8.1 Zero-copy KV 不等于 zero-cost context

Brain 可能有 80k tokens，而 CodeArm 实际只需要 15k 相关内容。共享避免了重复存储或 prefill 的一部分工作，但 decode 的 attention 仍可能访问很长的 KV；旧前缀也会占用保留容量，并可能引入无关上下文干扰。

$$
\text{KV sharing} \ne \text{free context}
$$

### 8.2 三种继承模式

| 模式 | 方式 | 优点 | 限制 |
|---|---|---|---|
| Full Fork | 继承当前 Brain 的完整合法前缀 | 不因截短丢失已有上下文；可复用其已算 KV | 保留长上下文，attention 成本和无关信息仍在 |
| Prefix Fork | 从较早的合法 checkpoint 分叉，例如 A→B→C→D，而不继承 E | 容易与 prefix 共享相容，避开后续无关轨迹 | 可能缺少 D 之后的重要信息 |
| Selective Context Fork | 理想上只取相关信息 | 有机会减少保留上下文与干扰 | 不能直接把任意旧 KV 块摘出并拼成等价的新序列 |

对 `[A][B][C][D][E][F][G][H]`，不能简单取 `[A][C][F][H]` 就声称与该短序列重新推理的 KV 等价。因果 Transformer 的后层状态依赖此前历史和位置，`KV(F)` 原本是在前面的上下文条件下得到的。任意删块或重排若要作为近似算法研究，需要另行证明质量，不能混入首版的精确共享语义。

### 8.3 现实候选：Shared Trunk + Context Capsule

保留一段**连续、合法的共享前缀**，再把任务与缺失的相关信息压成短 capsule，作为分支新的输入计算：

```text
Shared checkpoint + small task capsule + arm-local trajectory
```

```json
{
  "task": "inspect authentication code",
  "relevant_facts": [
    "refresh fails only in Safari",
    "backend token generation succeeds"
  ],
  "relevant_files": ["auth/cookie.py"]
}
```

这里的 “projection/selective” 指选择 checkpoint 并重编码语义信息，**不是任意 KV gather**。capsule 的生成、prefill 和潜在信息损失都要计入比较。

### 8.4 继承比例实验

$$
r = \frac{\text{shared inherited tokens}}{\text{Brain tokens at the reference point}}
$$

候选 `r = 100%, 75%, 50%, 25%, 0%`，比较 `Quality(r)`、`Latency(r)`、`Memory(r)`，寻找 Pareto frontier。实际 checkpoint 受合法位置与 block 对齐约束，报告实现得到的比例，不把理想比例当成精确可达。

`0%` 表示不继承所研究的 Brain KV 主干，不表示分支没有 system prompt、任务指令或 summary。固定提示开销和比例分母需要在实验协议中统一。

原讨论提出的可能发现是 full-context inheritance often unnecessary，但不应预设答案：如果 100% 最好，就如实报告；只有测得 25–50% + capsule 保持质量且降低成本，才能提出对应结论。

## 9. 三层 Benchmark

只跑一个真实任务 benchmark 并报告“更快”，不足以说明收益来自运行时、并行结构还是策略变化。原讨论提出 micro → synthetic → real-agent 三层。

### 9.1 第一层：Runtime Microbenchmark

| 变量 | 原讨论候选范围 |
|---|---|
| Shared prefix length | 4k、8k、16k、32k、64k、128k tokens |
| Branch count | 1、2、4、8、16、32 |
| 比较对象 | Normal independent requests、vLLM APC、Octopus Fork |
| 指标 | TTFT、TPOT、GPU KV memory、Prefill FLOPs、Throughput、Peak VRAM、Fork latency |

这些是设计空间，不是要求把所有组合立即跑完。超出所选模型 context limit 或设备 KV 容量的点不能作为正常可比较输入。

原讨论建议参考 vLLM 的 fixed-prefix、ShareGPT、prefix repetition 等 prefix-caching benchmark 方法；具体版本和脚本未核验。

目标是拆开测量 fork 开销、实际 prefill 工作、共享页面驻留与峰值内存。必须区分冷缓存、暖缓存，以及 fork 前 Brain 已经计算好的共享状态：不能把 Octopus 的共同 prefill 排除，却把 baseline 的对应工作计入。

尤其应允许 APC 使用完全相同、可命中的前缀。Octopus 未必在每个纯缓存 microbenchmark 上胜过 APC；可能的额外价值需要在状态生命周期、取消和端到端策略中体现。

### 9.2 第二层：Synthetic Branching Benchmark / OctoBench

目的：控制真实 benchmark 无法独立控制的并行结构、共享程度和信息相关性。

原例子：提供 100 个文件，File 17 包含 database clue、File 43 包含 API clue、File 81 包含 config clue；任务需要综合三个来源，而不是仅找出其中一个。

| 变量 | 意义 | 原讨论候选值 |
|---|---|---|
| `N` | Branch count | 1、2、4、8、16 |
| `L` | Shared context length | 4k、16k、64k |
| `R` | Relevant context ratio | 0.1、0.25、0.5、0.75、1 |
| `D` | Dependency depth | 1、2、4 |

还希望控制 branch independence、context overlap 和 branch depth。`R` 是任务上下文相关比例，`r` 是实际继承比例，两者不是同一变量。

这些变量用于设计可解释的对照，不意味着直接运行整个笛卡尔积。需要有可判定的综合答案和任务依赖，避免 benchmark 只奖励复制几个显眼线索而无法代表分支推理。

共享上下文的原始成本直觉：

$$
C_{\mathrm{MA}} \approx N L_s + \sum_i L_i
$$

$$
C_{\mathrm{Octopus}} \approx L_s + \sum_i L_i
$$

$$
\mathrm{Saving} \approx (N-1)L_s
$$

这里首先是**未缓存独立请求下的重复 prefix-token 处理量模型**，不是完整 GPU 时间或 FLOPs 定理；prefill attention 的成本也不一定随长度线性变化。Octopus 还要支付 capsule、commit、调度和 decode 等成本，而能命中的 APC 也可能把共享前缀计算接近降到一次。不能用这组公式直接证明 `Octopus > APC`。

### 9.3 第三层：Real Agent Benchmarks

| Benchmark | 适合的任务/触手 | 主要观察 |
|---|---|---|
| SWE-bench Verified | CodeArm、TestArm、RepoSearchArm、DocsArm；真实 repo issue 修复 | Resolved %、wall-clock time、GPU-seconds/solved issue、input tokens/solved issue、peak KV |
| GAIA | WebArm、ResearchArm、FileArm；推理、浏览、文件/多模态工具任务 | 任务成功率、时延、工具开销与 GPU 成本，检验是否不限于 coding |

原讨论将 SWE-bench Verified 描述为 500 个经人工筛选的问题，并提到 Docker 化 harness；GAIA 被描述为覆盖 reasoning、web browsing、多模态和 tool use 的 general assistant benchmark。这些是原讨论的资料线索，本次没有重新核验数据版本、许可、split 或可运行条件。

真实任务比较单 Agent、Multi-Agent 与 Octopus；工具可用性、数据 split、模型能力、超时和预算必须一致。GAIA 若采用不具多模态能力的模型，不能悄悄将多模态子集与其他系统混比。首版并不因此要求实现所有 Arm 或立刻运行两个完整 benchmark。

## 10. Baseline 与公平比较

原文说“至少五个”，实际列出了 B0–B5，合计六个配置；其中 B0–B3 是外部/传统对照，B4–B5 是 Octopus 变体。

| ID | 配置 | 目的 |
|---|---|---|
| B0 | Single Agent | 单模型串行执行，给出串行质量/成本参考 |
| B1 | Independent Multi-Agent | 每个 Agent 独立拿完整 context，测无共享状态的复制代价 |
| B2 | Multi-Agent + summary | 主 Agent 给子 Agent 压缩上下文，比较传统语义压缩方案 |
| B3 | Shared Prefix Cache / vLLM APC | 相同 prefix 的后端缓存复用，无 Octopus 显式 runtime；关键强基线 |
| B4 | Octopus Full-Fork | Octopus 生命周期与提交机制，但全量继承 Brain KV |
| B5 | Octopus Selective | checkpoint + capsule + adaptive fork 的完整候选系统；尚未实现 |

**B3 必须存在。** 否则容易把 paged KV / prefix caching 已经提供的收益归到新系统上。

| 公平性要求 | 为什么影响结论 |
|---|---|
| 同模型、同权重、同 tokenizer/数值设置 | 排除额外训练或模型变化造成的提升 |
| 同任务、工具权限、数据与质量口径 | 不能靠不同任务难度制造优势 |
| 缓存状态、共享前缀字节/token 一致性和预热成本透明 | APC 需要真实可命中的相同前缀，不能人为破坏 cache hit |
| 核心 runtime 对照尽量固定任务分解与分支工作量 | 区分运行时收益与少做工作/换策略的收益 |
| 策略比较另报告预算、取消量和完成任务质量 | adaptive fork/cancellation 会改变实际执行量 |
| 计入 Brain、capsule、commit、失败分支和最终答案成本 | 避免只计便宜的支路或成功样本 |
| 同硬件、并发、终止条件与明确的时延测量范围 | 避免把 GPU 资源扩容或工具等待差异当作架构提升 |

APC 主要按相同 prefix 提供缓存复用；Octopus 希望暴露显式 checkpoint、分支生命周期、共享状态语义、semantic commit、context selection 和取消。这些是待比较的抽象/机制，单有 API 名称并不能证明 novelty 或性能优势。

## 11. 消融与核心指标

### 11.1 消融矩阵

| 变体 | 希望回答的问题 |
|---|---|
| Full Octopus | 完整候选机制的质量/成本参考 |
| No checkpoint policy | 固定或简单边界是否已经足够；与语义+周期策略比较 |
| No semantic commit | 结构化语义增量相对普通结果回传是否有收益 |
| No context capsule / projection | 截短继承和补充语义信息各自贡献多少 |
| No adaptive fork | 固定分叉与效用驱动分叉有何区别 |
| No branch cancellation | 提前终止是否节省成本，以及是否损伤质量 |
| No shared KV | 生命周期/策略不变时，KV 共享本身贡献多少 |

原讨论没有定义每个 “no X” 的精确替代机制。实现实验时应一次改变要研究的因素，并明确替代行为；不能将禁用 capsule、改变继承长度和改变任务分解同时发生的结果解释为单一模块贡献。

### 11.2 指标定义与口径

| 指标 | 应记录的含义 |
|---|---|
| Task success / Resolved % | 在相同任务集上的完成质量；“质量不掉”需明确容忍范围及不确定性 |
| Wall-clock completion time | 用户任务开始到终止结果的端到端时间，包含工具等待和汇总 |
| TTFT | 明确从请求提交还是 fork 调用开始，到首个输出 token；不能混用 |
| TPOT | decode 每输出 token 的耗时，关注继承长度造成的 attention 成本 |
| Fork latency | checkpoint 到可调度分支的建立开销；是否含 capsule prefill 应说明 |
| Throughput | 固定硬件与并发下的 token 或任务吞吐，注明单位 |
| GPU-seconds | 所有占用 GPU 的时间之和，注明使用占用时间还是有效计算时间 |
| GPU-seconds / solved task | 整个评价集合的总 GPU-seconds / solved 数，计入失败尝试消耗 |
| Input tokens / solved task | 所有输入 token 的累计/solved 数，同时拆开实际 prefill 与 cache reuse |
| Prefill / Decode FLOPs | 若可测则分别报告；估算值应注明模型和口径，不用 token 数直接冒充 FLOPs |
| GPU KV memory | 唯一物理 KV 页面占用，避免将共享引用重复计为物理内存 |
| Peak VRAM | 包含模型、KV、工作区等的峰值显存，不等于 peak KV |
| Context redundancy | 共享上下文被重复实际处理的比例，明确计数对象与分母 |

原讨论提出：

$$
\mathrm{Efficiency} = \frac{\mathrm{TaskSuccess}}{\mathrm{GPUSeconds}}
$$

可操作化时，在同一任务集合上用 `solved tasks / total GPU-seconds`；若使用 success rate，需要保持任务数一致。没有解出的任务时，`GPU-seconds / solved task` 无有限值，应显式标注而不是丢掉这些任务。

另一个候选指标：

$$
\mathrm{Agentic\ Efficiency} =
\frac{\mathrm{SuccessRate}}
{\mathrm{Prefill\ FLOPs} + \lambda \cdot \mathrm{Decode\ FLOPs}}
$$

这是原讨论中的候选表达，`lambda` 的含义与取值尚未确定，不能把它当成已有标准指标或事后调权美化结果。

Context Redundancy Ratio 的原表达：

$$
\mathrm{CRR} =
\frac{\text{repeated context tokens processed across branches}}
{\text{total context tokens processed}}
$$

需要区分三件事：逻辑请求中重复出现的 tokens、实际重新计算 prefill 的 tokens、decode 继续读取缓存的工作量。APC 和 Octopus 都可能不重算某段 prefix，但逻辑输入 token 仍包含它。没有统一这一口径，CRR 下降不具可比性。

原讨论猜测传统系统可能有 70–90% 冗余、Octopus 理想上 10–30%；**这两个范围不是测量结果或可引用事实**。

## 12. 优势如何呈现与示例结果

### 12.1 最值得呈现的三张图

| 图 | 坐标/内容 | 需要支撑的结论 |
|---|---|---|
| Branch scaling | 横轴 branch count；纵轴 GPU cost/实际 context 处理量，包含 APC | 共享部分的重复成本如何增长；不能预画成 Octopus 恒定 |
| Inheritance Pareto frontier | 继承比例与 Success、Latency、Memory 的关系 | full inheritance 是否必要，存在什么质量/效率折中 |
| End-to-end completion | Single Agent、Multi-Agent、Octopus 的完成时间 | 在可比质量下是否真正减少用户等待 |

随 branch 增加，即使共享 prefix 不重复计算，私有 decode 和工具工作仍会增长；Octopus 总成本不能仅凭架构示意就宣称为平线。

### 12.2 原讨论的消融表样式——全部为假设数字

**下表仅保留原讨论期待的结果形态，禁止当成实验数据引用。**

| Variant | Success | Latency | GPU-sec | Tokens | Peak KV |
|---|---:|---:|---:|---:|---:|
| Full Octopus | 62% | 81 s | 110 | 42k | 19 GB |
| − KV sharing | 62% | 129 s | 181 | 42k | 31 GB |
| − projection | 59% | 113 s | 157 | 81k | 27 GB |
| − adaptive fork | 60% | 101 s | 149 | 61k | 24 GB |
| Multi-agent | 60% | 144 s | 210 | 105k | 35 GB |

这些 “GPU-sec / Tokens” 在原文未明确是每任务平均还是其他聚合，不能补成既定测量协议。

### 12.3 原讨论的继承比例示意——不是发现

| 配置 | 假设 Accuracy | 假设 Latency |
|---|---:|---:|
| 100% Brain KV | 64% | 1.00× |
| 50% KV + capsule | 64% | 0.73× |
| 25% KV + capsule | 62% | 0.58× |
| 0% KV + summary | 54% | 0.51× |

该表用于说明 Pareto frontier 的展示方式，不说明 50% 或 25% 已被证明更优。

### 12.4 其他原始效果示例——不是目标承诺

| 示例场景 | Baseline / Multi-Agent | Octopus | 原讨论希望强调的内容 |
|---|---|---|---|
| 成功率、费用与时延 | 60%；$4.80/task；180 s | 61%；$2.20/task；95 s | 成功率接近时减少费用和等待 |
| 归一化系统效率 | 58.3% success；100% GPU-sec；100% context processed | 58.1% success；54% GPU-sec；37% context processed | intelligence 相近，重复计算更少 |
| 单纯精度提升的对照叙事 | SWE-bench 48% | 53% | 原讨论认为仅强调 +5 个百分点，不如解释成本曲线干净 |

以上全为说明性数字。特别是 58.3% 与 58.1% 接近，不自动构成统计意义的“质量等价”。

## 13. 实现边界、模块与推进顺序

### 13.1 v0.1 优先借用后端能力

原建议基于 vLLM 已有 paged KV / prefix caching 的设计方向，不先写 attention kernel。需要考察 `KVCacheManager` 的 request→block 映射、引用计数、prefix reuse、allocate/free 与 block pool 行为是否能承接显式 checkpoint/fork。

这些是接入调查目标，不是对某个当前版本公开 API 的保证。能做缓存命中，不自动等于能安全保留任意活跃 request 的状态并继续 fork。

### 13.2 逻辑模块草图

```text
octopus/
  kv/
    block.py
    pool.py
    checkpoint.py
    manager.py
  runtime/
    branch.py
    scheduler.py
    session.py
  state/
    commit.py
    reducer.py
  tests/
    test_fork.py
    test_abort.py
    test_gc.py
    test_branch_isolation.py
```

这是原讨论的候选分层，**本仓库尚未创建这些实现文件**。`block.py`/`pool.py` 可以是后端适配或逻辑视图，不应因为草图里有文件名就重复造 vLLM 已有的内存管理器。第一阶段不先建设 WebArm/CodeArm 等完整应用。

### 13.3 首先需要成立的四个 invariant

| Invariant | 应证明的实际行为 |
|---|---|
| Fork 不复制 shared KV | 在支持的 full-block 场景下共享同一物理 KV，而不只是 Python 引用列表看起来相同 |
| Branch 写入不改变 parent | 分支新增 token 不污染 checkpoint 表示的前缀状态 |
| Abort 后引用正确 | 释放被取消分支持有的资源，保留其他 owner 仍需要的页面 |
| Kill 不影响 sibling | 取消任意分支不会破坏其他分支的合法继续执行 |

这些目标需要在实际后端执行边界成立；纯元数据模拟只能证明对应模拟逻辑，不能代替 GPU KV 隔离证据。

### 13.4 建议推进顺序（不是已启动任务）

| 阶段 | 内容 | 边界 |
|---|---|---|
| 接口收敛 | 选定后端版本，明确 checkpoint 所有权、block 对齐与 Branch 状态机 | 不把草案矛盾带入实现 |
| Runtime v0.1 | full-block checkpoint/fork、私有 suffix、abort/release/GC，最小可用提交链路 | 先四个 invariant，不做完整 Arm 平台 |
| Runtime microbenchmark | 与独立请求和 APC 比较 | 拆开共同 prefill、fork 与 decode 成本 |
| OctoBench | 控制独立性、共享长度、相关比例与依赖深度 | 回答核心机制与 scaling 问题 |
| Inheritance / policy | Prefix + capsule、checkpoint/fork/cancel 策略及必要消融 | 不预设最优继承比例 |
| Real tasks | SWE-bench Verified / GAIA 中与模型、工具能力匹配的任务 | 检验端到端迁移与质量/成本 |
| v0.2 或后续 | partial-block copy-on-write；是否扩展由实际需求决定 | 不混入首版 zero-copy 的支持范围 |

本次工作仅整理并发布文档，没有启动实现、训练、GPU 作业或 benchmark。

## 14. 创新性、风险与工作评价

### 14.1 最值得保留的三个方向

| 候选贡献 | 核心内容 | 尚需证明 |
|---|---|---|
| Systems | Forkable KV-backed inference runtime：checkpoint/fork/branch/abort/commit/GC | 显式状态抽象的正确性、可用性及相对强缓存基线的增量价值 |
| Agent architecture | Shared cognition + asynchronous semantic commits | 降低复制/协调成本而不损伤结果质量 |
| Adaptive inheritance | 选择分支需要继承的最小有效认知状态 | Quality–Latency–Memory trade-off 是否稳定存在、策略是否优于简单选择 |

“KV cache 从优化机制提升为 runtime primitive”是概念重心：不是只谈 prefix caching，而是把合法推理状态纳入可操作的执行生命周期。

### 14.2 最大风险：只是 prefix caching + orchestrator

潜在评审意见：**“Isn't this just vLLM prefix caching + an agent orchestrator?”**

如果最终只有 `vLLM prefix caching + parallel requests + JSON result`，可能是一项有用的工程整合，却不足以支撑更强的研究贡献。

希望体现的差别是：

```python
cp = brain.checkpoint()
a, b, c = cp.fork(3)
brain.commit(a.result)
```

而不只是：

```python
llm.generate(same_prefix + prompt_a)
llm.generate(same_prefix + prompt_b)
```

**接口不同不等于功能或性能必然不同。** 必须用可观察的生命周期语义、上下文继承策略、取消行为和端到端成本说明新抽象带来的实际价值，也要接受纯共享 prefix 场景下与 APC 接近的结果。

### 14.3 OS 类比及其边界

| OS | Octopus 类比 |
|---|---|
| Process | Branch |
| Memory pages | KV blocks |
| `fork()` | 从 checkpoint 派生推理分支 |
| Copy-on-write | 不可变共享前缀、私有后缀；partial page 需单独处理 |
| IPC | Semantic events |
| Process exit | Branch abort/终止 |
| Return value | Semantic Commit |
| Scheduler | Branch Scheduler |
| Page eviction | KV 缓存驱逐/GC |

这个类比可以帮助设计接口，但不能隐含声称已有通用进程内存快照、事务合并或任意暂停恢复能力。Octopus 是 “fork cognition” 的探索，不是把完整 OS 机制搬进 Agent 框架。

### 14.4 第一篇暂不碰 LoRA/RL

原讨论倾向把变量收紧到相同模型和权重：KV runtime + agent architecture + inheritance 已足够构成一组问题，再加入 LoRA 训练会让归因不清。

后续独立方向可称为 **Learning to Fork: Training LLMs for Branching Cognition**，研究何时 fork、fork 几个、继承多少、何时 abort/commit。LoRA/RL 仅记录为未来可能性，不是本次计划或授权训练任务。

### 14.5 原讨论的主观评价

原讨论给出的印象分是 Idea 8.5/10、Research question 8/10、技术成熟度 3/10，认为实验充分时有较好的论文潜力；也用“若仅为缓存加编排，可能从 8.5 降到约 5”表达风险。

这些分数只是当时的主观评价，不是外部评审、查新结果或经过量化的研究质量。值得继续研究的理由是状态抽象与继承问题，而不是“章鱼”隐喻本身。

## 15. 设计状态、待决问题与来源说明

### 15.1 当前设计状态

本节保留讨论形成时的状态，不回写成实验结论。已经收敛并实现的 v0.1
接口、full-block 尾部重算、原生 KV lease owner 与验证入口曾记录在
`docs/v0.1-runtime.md`（已随 #219 删除，见该提交之前的历史）；真实首轮数据见
[`results/v0.1`](../results/v0.1)。

| 项目 | 讨论状态 |
|---|---|
| Immutable shared prefix + mutable private suffix | 核心提议；未实现 |
| Commit 是语义增量，不是 KV merge | 明确的设计方向 |
| 复用 vLLM，不先写 attention kernel | 首版工程建议；具体接入点未验证 |
| v0.1 full-block checkpoint | 工程简化提议；真实语义边界不对齐的处理待决 |
| partial-block copy-on-write | v0.2 候选 |
| Scheduler 资源层 / Brain 价值层分工 | 职责方向；具体算法未定 |
| Semantic + periodic checkpoint | 策略假设，不是已证最优策略 |
| Shared trunk + context capsule | 可实现的继承研究方向；不是任意 KV pruning |
| APC baseline | 评价中不可省略的关键对照 |
| LoRA/RL | 首轮范围外，保留后续研究可能 |
| 性能提升与质量等价 | 均未验证 |

### 15.2 实现前必须明确的开放问题

| 问题 | 原讨论留下的缺口 |
|---|---|
| Checkpoint 所有权 | `ref_count` 统计 Brain、Branch、Checkpoint 的哪一层；何时 acquire/release |
| 活跃 checkpoint 与缓存驱逐 | 保证可 fork 的页面如何保留；普通 warm cache 与有效 checkpoint 不能混用 |
| `checkpoint()` vs `seal()` | 是否同义、各自何时使视图不可变，以及 parent 后续如何继续 |
| 对齐与真实输入 | semantic boundary 不在完整 block 末端时，首版到底支持什么调用 |
| Branch 状态机 | 是否保留 `FINISHED`；工具返回、取消、暂停、驱逐分别如何转移 |
| 私有块变为共享块 | branch 被 checkpoint/fork 后，abort 不能按旧标签无条件 free |
| Backend 接入 | 后端版本、稳定 API、request 的状态驻留和物理页面生命周期 |
| Semantic reducer | 冲突、重复、过时结果、顺序和 artifact 引用如何处理 |
| 异步 commit | 如何在不等待所有 Branch 的情况下让 Brain 接收有效结果 |
| Fork/value policy | 效用项如何估计、预算如何影响决策，简单策略是否足够 |
| Inheritance policy | 哪个合法 prefix、capsule 内容与生成成本，以及最优比例是否存在 |
| 实验口径 | 质量等价、冷暖缓存、GPU 成本、逻辑 token/真实 prefill/缓存读取的区分 |

### 15.3 来源与完整性说明

本记录仅整理用户给出的讨论；原文出现 `Sources` 标记，但没有附可核对的文献或链接。因此不能据此声称完成了相关工作查新，也不能把“vLLM 最近支持某种 cache retention/checkpoint”等描述直接当作当前版本的已验证功能。

需要后续核验的资料线索包括：vLLM 的 paged KV、`KVCacheManager`/block pool/prefix caching、完整 block 复用与 eviction、agentic cache retention 和 prefix-caching benchmark；SWE-bench Verified 的数据版本与 harness；GAIA 的任务划分与工具要求。

原文最后停在“进一步收紧成：”，没有附后续文字。本整理不补造不存在的原文结论；全文保存在 [docs/discussion-original.txt](discussion-original.txt)。本文的定位与结论均来自此前已经表达的内容，技术澄清则明确以边界或待决问题呈现。
