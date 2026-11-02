# Octopus：可下载真实任务与 prefix/contention 发现

核验日期：**2026-09-11**。结论：**先用真实同文档多问题证明长前缀机会，
再用真实到达/prefix-hash 轨迹检查竞争结构，最后用完整 Agent 任务验证
收益。不能把 SWE-bench、公司模拟或 ML 训练任务的名字当作 KV 压力保证。**

本次范围是联网一手来源核验、实际小文件下载、输入结构与 CPU tokenizer
审计、可复现工具和后续协议（工具与协议已于 #219 移出主线，恢复点 `4f82ad5`）；未启动 GPU/模型推理。下文的“适合”是基于
输入/源码的推荐，不是 Octopus 性能排名。

## 阅读与复现入口

| 文件 | 内容 |
|---|---|
| [工程/企业来源报告](benchmark-discovery-engineering.md) | SWE-bench Verified/Multimodal、TheAgentCompany、WorkArena++ 及替代任务的具体 ID、访问/环境/评分约束 |
| [研究/数据来源报告](benchmark-discovery-research-data.md) | MLE-bench、DeepResearch Bench、BrowseComp、MuDABench，以及 QuALITY、QASPER、LOFT、MultiDoc2Dial、Mooncake/BurstGPT；逐项一手引用和下载命令 |
| CPU probe 使用说明、下一轮验证协议 | 已在 #219 删除（`docs/workload-prefix-probe.md`、`docs/benchmark-validation-protocol.md`），恢复点 `4f82ad5` |
| [小型来源证据](../results/benchmark-discovery/) | GET/HEAD、revision、数据结构计数和派生审计；不是模型执行结果 |

## 先修正 P1 的问题定义

P1 **已经有 prefix reuse**：两方法各 40 actions 的 native adopted-prefix
总量都为 **39,584 tokens**。问题是只有一个后续 consumer，且没有目标
eviction/rejection/adoption 差异；233 次 policy draw 没有辨识力。
两方法 prefill 为 7,759 / 7,761 tokens，历史 9/40 paired prompts 存在
序列化差异，不能把两 token 差异归因于 retention。0/4 resolved 也说明
完整任务质量没有通过。来源：[P1 报告](../results/p1-real-agent/report.md)、
[机器可读统计](../results/p1-real-agent/report.json)。

因此新的候选必须分别回答：**有真实独立问题吗？最终 token 前缀相同吗？
实际请求何时共存/返回？目标 prefix 会失驻留吗？普通 APC 是否已经足够？**

## 最有把握的机制候选

| 用途 | 首选与具体输入 | 已证实 | 尚未证实 |
|---|---|---|---|
| 原文长共享前缀、无需答案设计 arms | **QuALITY v1.0.1 test**：116 篇文章、2,128 原生问题；`article_id=63860` 的两组共 17 问 | 同文章两份记录正文一致；7.1 MB 可取，test 无答案标签 | document-first 是 serving 模板适配；没有真实 arrival/tool DAG；test 无本地 gold 评分 |
| 科学问答、公开可本地评分 | **QASPER test v0.3**：416 论文、1,451 问；`1911.10742` 中的数据规模、标注方法、baseline 等原生问题 | 3.87 MB 包可取，含官方 evaluator；384 论文至少两问 | 原 LED reader question-first，需声明模板适配；gold 必须隔离；不是完整研究 Agent |
| 官方原生 corpus-first | **LOFT QUEST / SciFact 128k**：两个不同 corpus，各 100 test queries | 官方共享 corpus/few-shot 在 query 前；确定性 retrieval evaluator | 128k 是数据标签不是 Qwen token 实测；不直接适配 P1 的 16k；单 corpus 常驻可能让 APC 已足够 |
| 真实到达与缓存关系 | **Mooncake FAST'25 Tool&Agent**：23,608 请求 | 真实采样 arrival、input/output lengths、512-token cumulative prefix hashes | 无原文/token IDs、工具时序、parent IDs、任务答案；不能验证有用 arms 或原文 tokenizer |

来源和许可逐项见[研究/数据报告](benchmark-discovery-research-data.md)。
QuALITY 和 QASPER 的多个问题是原生任务，但把它们同时提交是人为选择的
服务调度；不能把这个选择包装成原 benchmark 原生并行 Agent。

### 已下载并完成的精确 token 审计

固定 Qwen2.5-7B-Instruct tokenizer/chat template、16-token block，按输入
ID 选取而非按答案或 LCP 筛选。以下是 **83 个原生问题 + 12 个构造对照**，
不是全数据集的 tokenizer 统计；未生成答案。question-first 只是布局反例，
**不是拿来宣称胜过 APC 的弱基线**。

| 输入 | 本次实际审计 | Context-first 共同前缀 | Question-first | 最大 prompt + 256 reserve |
|---|---|---:|---:|---:|
| QuALITY | 4 文章、71 原生问题 | 2,693–4,876 tokens | 19 tokens | 5,315 |
| QASPER | 4 论文、12 原生问题 | 4,996–13,065 tokens | 19 tokens | 13,346 |
| P1 新构造对照，非历史 prompts | 4 issues、12 个角色问题 | 106–879 tokens | 19 tokens | 1,151 |

全部请求加 reserve 均适配现有 **16,384-token** 窗口，无截断。QASPER
实际论文为 `1601.03313`、`1602.03483`、`1602.07563`、`1603.00957`；
只用 title/abstract/full_text 与原问题，不含 answers、gold evidence 或图表。
在**假定**两份不同文档 × 各三问题同时驻留、每请求预留 256 output tokens
时，共享块去重后的 context-first 峰值为 QuALITY **543.375 MiB**、
QASPER **1,188.25 MiB**。QASPER 的两个窗口仍超过假定 1,024 MiB
预算 65 / 188 blocks，说明它是比本次 QuALITY 样本更强的容量压力候选。
**这是 CPU 容量算式，不是 native eviction；1 GiB 也不是 P1 正常 profiling
的实测容量。** 普通 APC 应得到相同 context-first 输入。

三份[派生 audit](../results/benchmark-discovery/probe/)和
完整复现命令与 probe 工具已在 #219 删除（恢复点 `4f82ad5`），派生 audit 仍在仓库里。Probe 在内存中处理并丢弃
原始语料；仅少量 answer-free 样例留在忽略目录，正文不提交到仓库。

### Mooncake 为什么有用，又为什么不能单独完成验证

Tool&Agent 请求输入均值 **8,596.24 tokens**、输出均值 **182.13 tokens**，
到达覆盖 0–3,536,999 ms，同一毫秒最多 47 请求。它直接补上静态 QA
没有的真实请求到达和块关系，但到达重合不等于当时请求实际同时执行。

研究审计按原文件次序、无限容量、先前请求瞬间完成的乐观假设，得到
**57.03% input-token 可复用上界**；这不是 APC hit rate、HBM 实测或节省。
同刻请求按行序可见尤其乐观。最后不足 512 tokens 的块不计。不得把这些
hash IDs 当 tokenizer IDs，也不能跨 trace 文件混用 hash namespace。
原始 `output_length` 可用于离线 demand replay，不给在线 policy 偷看未来。

将 Mooncake arrival 套在 QuALITY 内容上可以设计实验，但必须叫
**hybrid synthetic replay**，不能再称原真实 trace。详细下载、算法假设
和计数在[来源报告 §4.5](benchmark-discovery-research-data.md#45-mooncake-fast25真正可下载的-prefix-hash--arrival)。

## 对用户原清单的关键修正

完整 Agent 轨道的推荐不是机械沿用“SWE-bench → TAC → MLE-bench”：

| 推荐位置 | 具体任务/版本 | 为什么保留或降级 |
|---|---|---|
| 原始真实 issue 首选 | **SWE-bench Multimodal v2** `GoogleChrome__lighthouse-10176` | issue 明确两个 bug，检测/去重与展示/信号可互补，回归设计可并行；480 test 已公开、本地可评分，旧 hidden-test 说法过时 |
| 长规格机制补充 | **FeatureBench fast** `mlflow__mlflow.93dab383.test_numpy_dataset.1beaad57.lv1` | 6 个目标文件、23,749 字符规格支持多个接口分支；是代码遮蔽/恢复生成任务，不冒充原始生产 issue，约 14.07 GB 镜像未拉取 |
| 原生文档分支后续候选 | **MuDABench** 年报子集 | per-document extraction 确有并发代码，但需本地可测的文档链路与严格金标隔离 |
| 企业完整性补充，不作第一机制样本 | **TheAgentCompany** `finance-budget-variance` / `admin-employee-info-reconciliation` | 前者数据可取但一条 dataframe 流程即可完成；后者访谈可并行但公共背景短，NPC 另需环境 LLM |
| 顺序/真实性对照 | **SWE-bench Verified**、**CORE-Bench**、**SWE-smith trajectories** | 能下载真实任务/代码/历史，但不证明长共享多分支；CORE 的独立脚本工作也可能不需要多个 LLM |
| 暂不阻塞主线 | **WorkArena++** | 代码开放，当前官方实例 manual-gated、匿名取实例 401；682 任务与每 level 235 个抽样 `(class,seed)` 不是同一计数单位 |

具体 task/source、许可、命令、官方材料冲突与环境成本见
[工程报告](benchmark-discovery-engineering.md)。**没有核验到同时原生提供
长共同背景、有用多臂、真实到达/工具 DAG 和可评分答案的全套 benchmark。**

**“长任务能拆”只是第一层筛选，不是 prefix/contention 结论。**
SWE-bench 的 issue 与初始仓库约束可能很短；共享 repo 不意味着所有 arms
应预读整个 repo。企业系统任务可以自然跨来源，但浏览器状态和后续写操作
往往依赖前序。需要先确认只读并行分支及确实共同需要的证据快照。

MLE-bench 的训练数据大不等于 LLM prompt 长；75 个比赛的全量数据约
3.3 TB，Lite 约 158 GB，训练成本会掩盖 prefill 信号。DeepResearch Bench
原版 100 queries、BrowseComp 1,266 queries 都没有统一长资料正文/工具
轨迹；适合信息收集或低共享对照，不能仅靠共同研究主题证明 exact prefix。

MuDABench 更值得保留：源码确有 per-document `asyncio.gather`，166 simple
与 166 complex 题共用 589 PDFs。但 GitHub clone 不含 PDFs，实际 HF
文件总计约 4.06 GB；默认文档抽取在 ChatDOC 服务，不代表本地 Brain
持有全文 KV。metadata 的 `value_*`/`detail_*` 是评分事实，删掉最终答案
字段仍会泄漏。两套题逐对共享同一文档集，不能当 332 个独立工作集。

## 交付与结论边界

所有比较都必须给 Multi-Agent APC 相同 prefix 布局、同一支持缓存的服务
端与相同工作，不能用禁用 APC/分散实例/不同前置角色作唯一基线。
共享活跃 trunk 不能按 fanout 重复计显存；更有识别力的 retention 情境是
真实工具等待后的返回与其他任务竞争，而不只是一次大 fanout。

第三方完整正文不重新分发进源码仓库；probe 丢弃下载的原始语料，样例仅存
忽略目录。公开的
派生报告与 task/source IDs 可复现。许可不明确、需要账号、评分器未执行、
内容超出模型窗口等限制均保留。**实际 native KV 差异、端到端质量与
Octopus 相对 APC 的收益仍未验证**；下一步 GPU 工作需另行明确授权。
