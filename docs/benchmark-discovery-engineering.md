# Octopus 工程与企业任务源发现：共享长背景、并行有用分支与缓存竞争

> 历史调研（2026-09-11），不随实现更新。文中的"Octopus""P1""eviction/rejection/adoption""APC 对照"指当时的 KV 保留 / admission 研究线；其代码与配套文档（`benchmarks/p1*.py`、`admission*.py`、`contention*.py`、`workload_probe.py`、`docs/workload-prefix-probe.md`、`docs/benchmark-validation-protocol.md`）已在 #219 删除，恢复点 `4f82ad5`。当前系统见 [README.md](../README.md)。

检索日期：**2026-09-11（UTC）**。范围：公开的一手代码、作者数据集、官方托管制品与发布说明。本文没有安装环境、调用付费 LLM、运行 GPU、执行基准任务或测得缓存收益。字符数是读取真实输入后计算的 Unicode 字符数，**不是 tokenizer token 数**。来源版本及小型派生证据见文末和 `results/benchmark-discovery/engineering/source-metadata.json`。

## 1. 结论：有可下载的候选，没有已证实的“天然长前缀并发基准”

**机制首选是 FeatureBench `fast` 的多接口任务；真实 GitHub issue 首选是 SWE-bench Multimodal v2。两者均需要明确标注为研究者增加的并行工作流，而非基准原生多 agent 并发。** 如果“真实任务”严格限定为生产用户原始需求，则 FeatureBench 的代码遮蔽生成任务不符合这一来源条件，应把 Multimodal 放在首位；不能因为 FeatureBench 的输入更长就掩盖这一差别。[FeatureBench 生成管线][f-pipeline]、[Multimodal 数据说明][mm-card]。

没有发现可以同时直接证明以下全部条件的现成发布：一份**确有必要的长背景**、同一任务内多个**互补且有工作价值**的 LLM 分支、完全相同的 token 前缀，以及会导致策略分歧的 KV 竞争。下表的排序是**预期机制匹配度，不是榜单成绩**；“可拆分”是基于后文具体输入的推断，不是观测到的 GPU 效果。

| 候选 | 已核验的任务来源与可用量 | 对 Octopus 真正有用的机制 | 关键反证与定位 |
|---|---|---|---|
| **FeatureBench fast** | 真实开源项目上的测试驱动遮蔽/恢复任务；`full=200`、`fast=100`、`lite=30`，后两者不能与 full 相加 | MLflow 样例有 6 个目标文件、23,749 字符接口说明，可分成类型推断、对象序列化、schema 构造等互补 arms | 不是原始 GitHub issue，也非原生并发；两条 MLflow 任务即使同 base/image，遮蔽后的代码背景仍可能不同；镜像约 14.07 GB。[数据][f-card]、[样例][f-row9]、[初始化][f-runtime] |
| **SWE-bench Multimodal v2** | GitHub issue/PR；当前实际 `test=480`、`dev=100`，公开本地评测 | Lighthouse 样例原文明确两个不同 bug；收集器/stack-pack 与展示层可协作，另有独立回归分支 | issue 本身仅 1,636 字符，需要真实相关代码补成长背景；仍然是研究者 fanout，且 gold evaluator 不覆盖全部设想的工作价值。[数据][mm-card]、[输入][mm-input]、[评测][mm-eval] |
| SWE-bench Verified | 500 个经人工确认的 Python issue/PR | 可作真实修复与顺序负对照，或多任务共享服务的竞争源 | 样例是单个 separability bug，不天然产生数个长时间互补 arms；500 个实例不是一个共享长前缀。[数据][sv-card]、[样例][sv-row] |
| TheAgentCompany | 175 个任务目录/镜像条目，**含 example**；人写任务、模拟公司环境 | 财务表有真实输入支持 5 个部门分片；HR 核对可并行联系不同 NPC | 财务任务更适合一次确定性 dataframe 计算；HR 背景短且需环境 LLM。角色很多不等于多个被测 agent 共享请求前缀。[目录][t-images]、[预算输入][t-budget-task]、[HR 输入][t-hr-task] |
| WorkArena++ | README 称 682 个组合任务；当前 agent curriculum 每 level 抽样 235 个 `(task, seed)` | 程序化企业流程、共享公司协议；可作顺序/访问受限对照 | 单浏览器操作及状态依赖明显；当前官方实例需人工批准，匿名下载实测 401。[README][w-readme]、[curriculum][w-curriculum]、[实例入口][w-access] |
| CORE-Bench（替代项 2） | 90 个论文 capsule，45 train + 45 test；3 个难度共 270 任务视图 | CULP 一个任务原生要求跑 3 个独立脚本；这是最清楚的实际独立工作分支之一 | 官方 `run.sh` 仍顺序执行；全部代码/说明仅 8,787 字节，执行极简单，弱长背景/弱 LLM 价值；不是企业工作流。[数据][c-train]、[制品][c-archive] |
| SWE-smith trajectories（替代项 3，官方轨迹） | `tool=24,100`、`xml=26,076`、`ticks=25,826` 个格式化记录 | 可下载真实 agent 消息，做 CPU tokenizer、序列化与顺序重入对照 | 三种格式不是三倍独立任务；没有原生分支图或并发时间线，转换过的消息也不是逐请求 wire trace。[数据][s-traj-card]、[转换代码][s-convert] |

用户提供的 P1 结果是：单 consumer、233 次 policy draws，但 eviction/rejection/adoption 没有差异，各方法 0/4 solved。本文**直接采用这个上下文，没有重新调查 P1**。它支持的下一步是先检查可测机制是否存在，不是立即换一个更知名的 benchmark 再跑同样的串行 agent。

## 2. 必须分开的四个概念

| 概念 | 可以从哪里验证 | 不能据此推出什么 |
|---|---|---|
| 语义上的共享背景 | 同一公司文档、同一仓库、同一任务规格 | 不能推出两次请求有相同 token IDs；文档排序、模板、角色消息、检索结果都可能不同 |
| exact token-prefix reuse | 对**最终请求模板化后的 token 序列**计算 LCP；还要核验模型、adapter、图片处理和 cache namespace | 不能只比较字符串中是否包含同一段文档，更不能用词相似度替代 |
| 任务间竞争 | 多个 task 的请求在同一个缓存域重叠、有不同可重用块组 | evaluator 的 `workers=4` 只可能意味着并行跑容器/测试，不代表 4 个并发 LLM 请求 |
| 任务内 fanout | 一个输入确实包含互补子问题，分支产物进入同一个最终交付 | 多个 agent 重复解同一题、best-of-N、自洽投票，不等于互补 arms；串行工具动作也不是并发 |

实现层证据可参考 `vllm-project/vllm:vllm/v1/core/kv_cache_utils.py:584–648`：块标识由 parent hash、当前 token IDs 及额外键构成；额外键包含 LoRA、multimodal、cache salt、prompt embeddings。故**同一段代码在不同前文之后不是可共享的前缀块**，相同图像 URL 也不足以证明相同视觉缓存身份。这里只借它说明机制，并未假设 Octopus 已使用这版 vLLM。[源码][kv-source]

建议的请求形状属于**研究协议**：固定 system/tools → 固定任务背景包（完整规格 + 必需的只读代码/数据快照）→ 最后追加 arm-specific 指令。不要把不同角色的 system prompt 放在长背景前面，也不要把分支 ID、随机时间戳插到共享包中。后续工具返回应进入各分支的后缀；背景包若真的变化，应视为新版本，而不是假装仍是同一前缀。

即使首轮所有 arms 共享很长前缀，也可能**更容易全部命中而没有竞争**。要产生有意义的策略选择，必须存在多个有未来价值的可重用块组、真实请求重叠和受限容量；只有一个永不退出的共享 root，或一直被 active requests 引用的块，不足以验证淘汰/adoption 策略。

## 3. SWE-bench Verified：高真实性，不是自然 fanout

### 3.1 数据、许可、获取

当前 HF revision `78f471bf655a3137b2e8a75af1501690ec009ec3` 的 test 为 **500 行**；单 parquet 的 dataset card 下载大小为 **6,304,616 bytes**。来源是 Python 仓库 issue/PR，单位测试验证修复。代码仓库为 MIT；当前 HF card **没有单独声明 license 字段**，不应将 harness 的 MIT 自动解释为全部 issue、代码片段及第三方仓库的统一许可。[SWE-bench/SWE-bench:LICENSE][swe-license]、[数据 card:1–90][sv-card]

只下载数据，不启动环境：

```bash
curl -fL 'https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/resolve/78f471bf655a3137b2e8a75af1501690ec009ec3/data/test-00000-of-00001.parquet' -o verified-test.parquet
# 无 datasets/Arrow 环境时，可先取得一行 JSON；该 viewer API 跟随当前数据，不是 revision-pinned。
curl -fL 'https://datasets-server.huggingface.co/rows?dataset=SWE-bench/SWE-bench_Verified&config=default&split=test&offset=0&length=1' -o verified-row.json
```

未来从本报告的 harness revision 安装（需先有 Docker，**本次未执行**）：`git clone https://github.com/SWE-bench/SWE-bench.git` → 在新 checkout 中 `git switch --detach 02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` → `pip install -e .`。task repository 与 HF dataset 是另外两份制品，不能只安装 Python 包就认为已拿到环境。[官方安装入口][swe-readme]

当前行结构已经包含 `image`、`eval_script`、`eval_type`、`log_parser` 等 harness 字段，不能照搬旧文档中“只有 issue + base_commit”的描述。agent 可见输入应白名单提取 `problem_statement`，以及控制器使用的 `instance_id/repo/base_commit`；**不要把整行 JSON 送给模型**。[实际 rows][sv-row]

### 3.2 实例和环境入口

`astropy__astropy-12907`：`base_commit=d16bfe05a744909de4b27f5875fe0d4ed41ce607`，问题是 nested `CompoundModel` 的 `separability_matrix` 错误，原始问题说明 **1,246 字符**。任务目录有 `problem_statement.md`、`task.yaml`、`Dockerfile`、`environment.yml`、`eval.sh`、`gold.patch`、`test.patch`、`tests.json`。Dockerfile 配置 Linux amd64、conda，具体 Python 为 **3.9.20**；不是要求所有 Verified 任务都使用同一 Python。[任务元数据][sv-task]、[具体 Dockerfile:23–123][sv-docker]

已安装对应 harness 后的 evaluator 命令（本次未执行）：

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified --split test \
  --instance_ids astropy__astropy-12907 \
  --predictions_path predictions.jsonl --max_workers 1 --run_id octopus-verified-pilot
```

prediction 单行格式为 `{"instance_id":"astropy__astropy-12907","model_name_or_path":"octopus-pilot","model_patch":"<unified diff>"}`。新版 CLI 等价入口为 `swebench eval verified -p predictions.jsonl -i astropy__astropy-12907 -j 1 --run-id octopus-verified-pilot`。本地重建镜像时还要下载 `SWE-bench/swe-bench-tasks`，通过 CLI 的 `--task-repo` 指定；不要将这个参数直接附到上面的 legacy argparse 命令，当前 legacy parser 未暴露它。[SWE-bench/SWE-bench:README.md:60–114][swe-readme]、[run_evaluation.py:797–853][swe-args]

官方整体机器建议是 x86_64、**120 GB 空闲磁盘、16 GB RAM、8 CPU 核**，并非已测得的单任务最低配置；单实例测试超时默认 1,800 秒。CPU 测试环境不要求推理 GPU，agent inference 的成本另算。[README:121–126][swe-machine]、[parser:823–839][swe-args]

### 3.3 并行性判断与泄漏边界

**推断**：这个输入可以安排根因定位、独立构造反例、修复后回归，但前两者很容易在找到同一个小函数后结束，第三者依赖修复；不构成“天然数个长时间有用 arms”的强证据。多个同仓库 issue 也通常有不同 base commit。只有同一个 repo 名称并不产生 exact prefix。

`patch/gold.patch`、`test_patch/test.patch`、`tests.json`、`FAIL_TO_PASS/PASS_TO_PASS` 和内嵌测试的 `eval_script` 留在 evaluator 控制面。`hints_text` 含 PR 首次提交前评论，但是否提供 hints 必须固定，不可从事后 PR/solution 取提示。[SWE-bench/SWE-bench_Verified 数据字段说明:78–90][sv-card]。推荐将 Verified 保留为真实性/顺序负对照，不作为本次机制主样本。

## 4. SWE-bench Multimodal v2：当前已开放，不能继续引用旧访问限制

### 4.1 版本纠偏、数据与访问

**截至本次检索，当前 test 可以公开下载并本地评测，不需要申请原先的隐藏 test 服务。** SWE-bench README 在 2026-09-01 宣布 v2 fully open source、480 个本地评测任务；harness `run_evaluation.py:715–719` 直接允许 test 本地运行，hosted submission 只是另一选项。[发布记录][swe-news]、[当前实现][mm-local]

HF revision `4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5` 的实际 metadata/rows 是 **100 dev + 480 test**，两份 parquet 合计下载大小 **11,303,640 bytes**。**相同 revision 的 card 正文仍写 102 dev + 510 test = 612**；这是官方材料内部不一致，本文以实际 split metadata 和 rows 的 `num_rows_total` 为准，不能将 612、旧论文集合和 480 混为一个可评测版本。[card metadata:45–61 与正文:68–72][mm-card]、[实测 rows][mm-row]

代码 harness MIT；该 HF card 也没有独立 license 字段。示例 Lighthouse 源码是 Apache-2.0；其他仓库及 issue 截图的权利边界需逐源处理，不作“全部图片都 MIT”的推断。[harness license][swe-license]、[GoogleChrome/lighthouse:package.json:170–173][lh-package]

```bash
curl -fL 'https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal/resolve/4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5/data/test-00000-of-00001.parquet' -o multimodal-test.parquet
git clone https://github.com/SWE-bench/swe-bench-multimodal-tasks.git
# 在这个新 checkout 内选取本报告核验的任务版本
cd swe-bench-multimodal-tasks
git switch --detach 8d3ed32f37af9ceb8b5778b92a0088384ed057d4
```

不需要先 clone 全任务库就能拿到 CPU probe 输入：直接下载下述 `problem_statement.md`，并从同一任务的 `problem_assets/` 下载两张 PNG。本次两条图片 URL 都得到 HTTP 200、PNG 文件头，Git tree 的文件大小分别 **62,742 / 92,884 bytes**。[任务目录][mm-dir]

### 4.2 真实任务支撑的三个 arms

`GoogleChrome__lighthouse-10176`，base `a5dbdd2eff4238213dcaad404d0e06d8d5d2489b`，问题标题 “Duplicated stack packs”。原文明确 “There's actually two bugs here”：重复 stack packs；实验性的 React/Fast Path 项不该直接显示给用户，但可能仍应保留 JSON 信号。[SWE-bench/swe-bench-multimodal-tasks:tasks/GoogleChrome__lighthouse-10176/problem_statement.md:1–20][mm-input]

下面的分解**来自 issue 和 base 代码，不来自 gold patch**：

| 分支 | 必需的共享输入 | 独立有用产物 | 依赖/冲突 |
|---|---|---|---|
| A：检测标识与 stack-pack 去重 | issue、两张截图、`stack-collector.js`、`stack-packs.js` 的现有 artifact 合约 | 收集/匹配端的修复与针对重复项的说明 | 与 B 必须先约定保留/展示的字段；不能各自改变 schema |
| B：用户展示与调试信号 | 同一 artifact 合约、`audits/dobetterweb/js-libraries.js` | 区分用户可见项目与应保留的底层信号 | 依赖共同契约，但不必等待 A 的完整实现 |
| C：独立回归设计 | 同一输入、允许看到的 base tests | 来自 issue 的去重/可见性回归与兼容性检查 | 测试设计可并发；最终执行与集成仍依赖 A/B |

base 中 `js-libraries.js` 消费 `Stacks` 并映射为可见表格；`stack-packs.js` 根据 `detector:id` 匹配 pack；`stack-collector.js` 负责采集。它们分别 **2,550 / 2,326 / 3,108 字符**，`package.json` 为 **9,793 字符**，issue 为 **1,636 字符**。把这些相关输入全部拼接也只是约 **19,413 字符（不含分隔符与图片）**；不能宣称已得到 32K/64K token 前缀。[GoogleChrome/lighthouse:lighthouse-core/audits/dobetterweb/js-libraries.js:27–63][lh-audit]、[stack-packs.js:40–75][lh-packs]、[stack-collector.js:39–91][lh-collector]、[package.json][lh-package]

有一条实际的序列化反例：HF row 与 GitHub raw 文件按原始 bytes 解码后**逐字相等，均为 1,636 字符**；但 Python 文本模式读取会规范化 CRLF，使本地字符串变成 **1,618 字符**，与原始 HF 字符串不同。这不是两个官方来源内容不一致，而是读取/格式化路径改变了表示。**语义相同不能代替 token-identical 检查。** probe 应生成一次公共块并原样复用，而不是让每个 arm 各自读取/格式化。

### 4.3 执行、评分与不利证据

任务镜像 `swebench/sweb.eval.x86_64.googlechrome_1776_lighthouse-10176:latest` 的 Docker Hub 标签实测 active，报告 `full_size=917,636,021 bytes`（约 0.918 GB，不能当作解压后磁盘占用）。源码 Dockerfile 为 Linux amd64、Node **16.20.2**，包含 Chrome、NVM、字体等浏览器依赖。[任务配置][mm-task]、[Dockerfile][mm-docker]、[镜像 API][mm-image]

```bash
# 对应版本的 SWE-bench 已安装后；仅评测最终合并 patch
swebench eval multimodal -p predictions.jsonl \
  -i GoogleChrome__lighthouse-10176 -j 1 --run-id octopus-mm-pilot \
  --task-repo ./swe-bench-multimodal-tasks
# predictions.jsonl 与 Verified 使用同一三个核心字段
```

实际 `eval.sh:130–134` 在应用测试改动后运行 `yarn jest --no-colors lighthouse-core/test/audits/dobetterweb/js-libraries-test.js`，不是逐图视觉判分，也不是对三个 arms 各给一个独立总分。**A 的工作有 issue 依据，但不能据此声称官方评分充分覆盖 A 的全部价值。**保留官方 resolved 指标，同时将分支产物是否进入最终补丁作为机制诊断，不擅自改 benchmark 分数。[具体 evaluator][mm-eval]

另有明确工程噪声：任务库 README 警告高并发下 stdout/stderr 标记顺序会使通过的测试被判失败，点名三个实例并建议低并发复跑；HF card 提醒部分 Carbon accessibility 测试在线拉规则引擎，判定可能随网络端变化。这些是**评测并行/外部依赖噪声，不是模型 KV eviction**。[SWE-bench/swe-bench-multimodal-tasks:README.md:28–41][mm-repo-readme]、[数据 card:143–145][mm-card]

图片必须区分 `image_assets.problem_statement` 与 `image_assets.test_patch/patch`。后两者可能包含 expected rendering，控制器只能在评测阶段注入；harness 明确为避免 agent 在容器中看到测试图，选择评测时 staging。[SWE-bench/SWE-bench:swebench/harness/run_evaluation.py:172–185][mm-assets]。纯 CPU 文本 tokenizer probe 可以检查文本前缀，但不验证视觉 embeddings/cache key。

## 5. TheAgentCompany：企业共享语境强，真实可并行 LLM 工作未必强

### 5.1 获取和依赖

当前 Git commit `98b68ef82a47690c316f42fddb05baafaab56851`。README 称有 175 个 task images；Git tree 实数 175 个 `workspaces/tasks/*/task.md`，列表明确包括 `example-image`，所以不能把它当作 175 个互异正式生产任务。六种角色是软件工程、产品、数据、HR、财务、行政；它们是任务分类，不是一次求解里的六个被测协作 agent。仓库 MIT，内置第三方服务/仓库/文档仍各有许可。[README:139、174–193、218–219][t-readme]、[workspaces/README.md:38–80][t-images]

可以直接下载任务文本和源文件；完整环境则是 Docker/Compose 下的 GitLab、Plane、ownCloud、Rocket.Chat 及备份数据。官方要求至少 **30 GB 空闲磁盘**，首次启动说明给出数分钟到 30 分钟；初始化会 reset 依赖服务并等待健康检查，文档称可至 10 分钟。官方参考机器是 EC2 t3.2xlarge，本文未实测资源最低值。[TheAgentCompany/TheAgentCompany:docs/SETUP.md:14–50][t-setup]、[docs/EVALUATION.md:45–53][t-eval-doc]

```bash
git clone https://github.com/TheAgentCompany/TheAgentCompany.git
# 只下载官方 setup 脚本作检查；本次未运行它，也未改变 Docker socket 权限。
curl -fL 'https://github.com/TheAgentCompany/the-agent-company-backup-data/releases/download/setup-script-20241208/setup.sh' -o setup.sh
# 未来完整评测所需的单任务镜像
docker pull ghcr.io/theagentcompany/finance-budget-variance-image:1.0.0
```

setup release 的 `setup.sh` 元数据为 **2,674 bytes**，不代表服务备份只有这么大。本次核验了 release 与源文件，没有拉取/启动 30+ GB 服务，也没有验证所有旧服务镜像在当前平台可完整恢复。[官方 release][t-release]

### 5.2 两个实读任务，而不是营销式角色拆分

**预算任务 `finance-budget-variance`。** task.md 指定 ownCloud 的 `/Documents/Financials/budget.xlsx`、`actual_spending.xlsx`，比较每部门/类别/月的预算和实际值，标出同时超过 10% 与 5,000 美元的差异，并产出 ownCloud 的 `flagged_variances.xlsx`，列名为 `Flagged_Variances`。依赖只有 ownCloud。[task.md:1–9][t-budget-task]、[dependencies.yml:1][t-budget-deps]

源目录的两份 xlsx 实际可下载，大小 **12,945 / 14,466 bytes**，各 **420 条数据 + 1 表头**。Engineering、Sales、Marketing、HR、Operations 各 84 行。用标准库解析 OOXML 并按原始单元格值串行为 CSV，分别得到 **14,073 / 14,089 字符**；这仅是长度探查序列化，Month 是 Excel serial date，**不是可直接冒充正式日期转换的 agent 输入**。[budget.xlsx][t-budget-data]、[actual_spending.xlsx][t-actual-data]、[派生证据](../results/benchmark-discovery/engineering/source-metadata.json)

**推断**：同一阈值/字段定义/完整数据快照可作为共有背景，5 个部门各一个读取分支，最后单写者合并。可是相同工作一条 dataframe group/filter 流程就能完成，给每个 LLM arm 复制全表可能只是人为重复上下文。它适合作为“有长输入但无需多个 LLM”的反例或确定性工作流探针，**不推荐用它证明多智能体并行推理的必要性**。

这里还有评测契约偏差：task 要求考虑季节性，evaluator 直接 merge、算差额/百分比、取绝对值阈值，并用 pandas `Series.equals` 比较输出顺序；没有显式季节调整。模型即使给出合理季节性解释、或只改变正确 ID 的排序，也可能与 grader 不一致。这个风险必须记录，不应擅自改 evaluator 或把其答案/排序送进公共前缀。[TheAgentCompany/TheAgentCompany:workspaces/tasks/finance-budget-variance/evaluator.py:41–76][t-budget-eval]

**HR 任务 `admin-employee-info-reconciliation`。** agent-visible `/workspace/employees.csv` 为过时员工表，需从 Rocket.Chat 找出当前员工并 DM 补全；CSV 源文件 17 行。不同员工的访谈可以并行，最终合并到同一个 CSV，互补性比重复修 bug 更明确。**但共有任务说明和表都短**；NPC 各自的隐藏 scenario 不是可塞给求解 agent 的长背景。[task.md:1–12][t-hr-task]、[employees.csv][t-hr-csv]、[dependencies.yml][t-hr-deps]

`pm-projects-analytics` 是读取六项项目统计后发一条 Rocket.Chat 汇总；`sde-report-unit-test-coverage-to-plane` 是 clone → 跑 coverage → 根据数值建 issue。前者可分项读取但信息量短，后者明确串行。不能把“同时使用 Plane/GitLab/Rocket.Chat”包装成天然扇出。[项目统计输入][t-pm-task]、[coverage 输入][t-sde-task]

### 5.3 精确 evaluator、NPC 与状态隔离

容器布局：`/instruction/task.md` 是 agent 指令，`/workspace` 是交付，`/utils/init.sh` 初始化，`/utils/eval.py` 评分。**当前 source 的参数为 `--result_path`，README 仍写 `--output_path`**；这里以 `workspaces/base_image/eval.py:77–105` 的 argparse 为准，但旧 `:1.0.0` 已发布镜像中的 CLI 尚未启动验证，使用镜像时应先在容器中看 `python_default /utils/eval.py --help`。[README:84–167][t-readme]、[实际 argparse][t-eval-source]

```bash
# 在未来已启动且隔离的 task 容器中：
SERVER_HOSTNAME=localhost bash /utils/init.sh
# 完成任务后，由 evaluator 进程设置官方文档给出的 DECRYPTION_KEY；不要注入 agent 进程。
python_default /utils/eval.py --trajectory_path /workspace/trajectory.json --result_path /workspace/result.json
```

上面的最简初始化仅说明无 NPC 任务的入口；需要 NPC 或 LLM grader 的任务还必须配置 `LITELLM_MODEL/LITELLM_BASE_URL/LITELLM_API_KEY`。官方环境 LLM 建议接近 Claude 3.5 Sonnet/GPT-4o，成本与 agent LLM 分开；本次没有调用。预算 grader 是 pandas/ownCloud 的确定性检查，HR 需要 NPC 交互。[docs/EVALUATION.md:6–15][t-eval-doc]、[init.sh:24–26][t-init]、[预算 evaluator][t-budget-eval]

官方 `evaluation/run_eval.sh:91–119` 是逐 task 调 `run_eval.py` 的循环，**不是原生多任务并发调度**。`init.sh` 会 reset 服务；多个 task 若共用一套可变服务并同时 reset，环境结果可互相污染。跨任务竞争试验需不同状态实例或明确分时初始化；同一任务的只读分析 arms 可共享快照，写入由最终合并者执行。这是依据 reset 实现的隔离推断，而不是已运行出的失败。[循环][t-run]、[init.sh:10–12][t-init]

不得向 agent 提供 `evaluator.py`、`checkpoints.md`、NPC `scenarios.json` 的秘密事实或解密配置。官方明确允许 benchmark 使用者检查 rubric，但禁止 examinee 读取。源码研究者看到了它们，不意味着后续 LLM 输入可以含它们。[docs/EVALUATION.md:121–126][t-eval-doc]

## 6. WorkArena++：组合流程不等于并发；当前官方实例确实 gated

### 6.1 任务数、许可、访问

当前代码 revision `a772230a94cf1caf4166b8ead3983f3b3786455b`。README 将 L1 写为 **33 类 / 19,912 实例**，WorkArena++ 写为 **682 个组合任务**，每类可采样大量配置。不要将 682 当作静态 JSON 中独立可下载的 682 条业务工单。[ServiceNow/WorkArena:README.md:58–60][w-readme]

实际 `get_all_tasks_agents(filter="l2", meta_seed=42)` 从 curriculum buckets 抽 `(class, seed)`。对当前源码中 `num_seeds × sum(weights)` 做静态计算，五类分别是 44、56、55、48、32，总计 **235/level**；L2 与 L3 使用同一权重结构、不同任务包装。**682、235、L1 的 19,912 是不同统计单位**；本文未安装环境去实例化 682 个注册类，因此不把 README 数字伪装成完整运行清点。[ServiceNow/WorkArena:src/browsergym/workarena/__init__.py:124–169][w-sampler]、[utils/curriculum.py:55–128][w-curriculum]、[compositional/__init__.py:28–54][w-levels]

代码 LICENSE 正文是 **Apache-2.0**，尽管 GitHub API 给出了 `NOASSERTION`。当前官方运行方式要求先在 `ServiceNow/WorkArena-Instances` 填姓名、机构、工作邮箱、研究用途，接受**仅 benchmark evaluation、不可训练或生产使用**的条件，等待人工批准。HF metadata `gated="manual"`；匿名请求 `instances.json` **实测 HTTP 401**。这不是本次某个 Python 包缺失导致的错误。[LICENSE:1–13][w-license]、[README:34–41][w-readme]、[官方实例入口/条款][w-access]

不要把历史的“自己申请免费 ServiceNow PDI”描述成已核验可替代当前官方实例池：本次没有证明任意 PDI 的版本、预置数据、API 权限兼容。代码可公开 clone，企业数据/实例访问不等于同样开放。

### 6.2 实际流程与参数

例：`OnBoardUserTask`，L2 专门化后由命名实现导出 ID **`workarena.servicenow.on-board-user-l2`**（L3 类似）。这个 ID 是从公开类名及 `get_task_id()` 推导，未在远端环境 reset 验证；运行时应把 `(task class, seed, level, meta_seed, instance version)` 都记录下来，而非仅保留人名。[task base:98–105][w-task-id]、[level wrapper:12–25][w-levels]

`onboard_user.py` 给出的真实构造顺序是：查 “Onboarding a new user” 公司协议 → 建 user → 订 MacBook Pro 15 → 建 hardware asset；hardware 的 `assigned_to` 取这个新 user 的姓名。L2 在 goal 给信息，L3 需要从 private task/公司协议获取。账户/硬件归属有真实依赖，不能假设 4 步互相独立。[ServiceNow/WorkArena:src/browsergym/workarena/tasks/compositional/onboard_user.py:112–221][w-onboard]

`CompositionalTask.setup_goal()` 维护 `subtasks`；`validate()` 通过 `valid_index` 检查当前 subtask；官方演示是同一个 `env.page` 上逐项 `cheat`。这说明默认接口不是原生多 worker，同时只证明验证状态机逐项推进，**不意味着每种组合任务的所有业务动作都数学上不能重排**。[compositional/base.py:83–95、215–236][w-base]、[README:170–193][w-demo]

访问批准后的官方安装入口（仅供复现，本次未执行）：

```bash
pip install browsergym-workarena
playwright install
```

运行接口是 `browsergym.workarena.get_all_tasks_agents(...)`、`BrowserEnv(task_entrypoint=task, headless=...)`、`env.reset()`、agent 的 `env.step(action)`。preferred runner 为 AgentLab；它的 parallel experiments 是不同 benchmark episodes，不是同一 task 的 LLM fanout。官方 README 的 demo 使用 oracle `env.task.cheat(...)`，**不能拿 demo 的成功率当 agent 成绩**。[README:28、43–53、110、155–196][w-readme]

本次由于 gating，没有拿到一条真实远端 reset 后的 observation、DOM、private-task 文本或完整公司协议。因此其**真实公共输入 token 长度、独立分支数和单任务墙钟成本均未验证**。不能把生成器里随机出的固定配置或隐藏 validation 状态当作 agent 的共享长背景。

## 7. 三个额外来源：只在有实际证据的维度上称“更强”

### 7.1 FeatureBench：长规格最强，但真实性定义和遮蔽边界很重要

#### 发布、下载和可见信息

代码 revision `8d4e347ec57546685c5a87e8676bf575db022ea6`；本次实际检查 HF **main revision `76b4a4566e04f4bcc13c35125d4f301791efa736`**。官方 `v1.1` tag 当时指向 `6ed8be1152d3a159317122141dcbdfcfa911ba34`，**不等于 main**；`fb eval` 默认可能回落到 v1.1，必须显式指定本次数据 revision。8 月 24 日发布说明说 V1.1 主要修正 problem statement 的不一致/缺漏。[README:20–24][f-readme]、[HF refs API][f-refs]、[harness CLI:54–61][f-cli]

`full=200`、`lite=30`、`fast=100`；三份 parquet 合计下载 **9,324,309 bytes**。代码和 HF card 均标 MIT，但被恢复的项目代码仍按项目许可。`fast` 发布说明明确不要求 GPU；官方 57.2 秒/题的 gold evaluation 均值来自 **2026-03-27 的发布说明与 Intel Xeon Platinum 8457C / 944 GB RAM 机器**，不是本次 revision 的实测、也不包含 LLM 推理，更不能当普通笔记本保证。[LiberCoders/FeatureBench:README.md:20–24][f-readme]、[HF card][f-card]

```bash
curl -fL 'https://huggingface.co/datasets/LiberCoders/FeatureBench/resolve/76b4a4566e04f4bcc13c35125d4f301791efa736/data/fast-00000-of-00001.parquet' -o featurebench-fast.parquet
# 已有 datasets 的 CPU 环境中可用：
# load_dataset("LiberCoders/FeatureBench", split="fast", revision="76b4a4566e04f4bcc13c35125d4f301791efa736")
```

重要：**`patch` 不是普通 SWE-bench 意义下的修复 patch，而是 corruption/masking patch，反向可恢复答案。** `RuntimeHandler` 从 `/root/my_repo` 恢复项目、应用这个 mask、默认删除 F2P 测试，然后给 agent 已损坏的 `/testbed`；gold evaluator 用 `preprocess_hf_patch()` 反向 patch 并剔除 F2P 文件。直接 clone `base_commit` 把原实现送给 agent 会泄漏答案；把 corruption diff 放进 prompt 同样会泄漏被删除的实现。[LiberCoders/FeatureBench:featurebench/infer/runtime.py:128–200][f-runtime]、[featurebench/harness/utils.py:276–287][f-mask]

#### 三条真实样例及可执行分解

本次用 `/rows?...&length=1` 读取，检查 `truncated_cells=[]` 后才计算长度；**`first-rows` 返回的某些字段被截成 100 字符，不能用它们做长度统计**。

| 实例 ID | 输入规模与真实目标 | 并行工作分解（推断） | 限制 |
|---|---|---|---|
| `mlflow__mlflow.93dab383.test_numpy_dataset.1beaad57.lv1` | **23,749 字符，6 文件**：`types/utils.py`、`data/dataset_source.py`、`data/evaluation_dataset.py`、`data/dataset.py`、`types/schema.py`、`data/schema.py` | A：tensor shape/dtype/schema inference；B：Dataset/Source JSON、属性、evaluation feature access；C：Schema/TensorDatasetSchema 构造与反序列化；最终集成者连接接口 | 三块通过 Schema/JSON 契约相连，需要先共享接口，而非三套独立重写。最合适的主样本。[实际输入][f-row9] |
| `pypa__packaging.013f3b03.test_metadata.e00b5801.lv1` | **43,308 字符，2 文件**：`src/packaging/metadata.py`、`requirements.py` | A：邮件解析/RFC822 序列化；B：Metadata/validator、异常聚合与版本约束；C：PEP 508 Requirement 解析 | A/B 在同一文件不同符号，合并冲突更大；validator 的 `requires_dist` 依赖 C，最终测试有 join。[实际输入][f-row0] |
| `mlflow__mlflow.93dab383.test_validation.2e1c5076.lv1` | **22,317 字符，3 文件**：`genai/scorers/base.py`、`validation.py`、`builtin_scorers.py` | A：scorer decorator；B：scorer list/data validation；C：builtin scorer 集合选择 | 有共享类型和 Databricks 条件分支，名字中 GenAI 不代表必须在线调用服务；本次未运行测试验证所有环境行为。[实际输入][f-row10] |

对照：`fastapi__fastapi.02e108d1.test_compat.71e8518f.lv1` 是恢复 4 个小 helper；输入虽有 9,073 字符，但部分函数就是 no-op shim。**长模板不自动代表复杂并行需求**。不把这一小 helper 题当强机制样例。[FastAPI 实际输入][f-row1]

MLflow 两个样例 base 都是 `93dab383a1a3fc9882ebc32283ad2a05d79ff70f`，image 都是 `libercoders/featurebench-specs_mlflow-instance_efffa87e`；这有利于下载复用，但**不同 mask 导致任务代码不同，因此不保证跨任务公共代码文本一致**。选择“同一任务的完整规格 + 相同损坏后只读快照”做 task 内共享，比默认假设同 repo 的 task 间 prefix 更可靠。[样例][f-row9]、[样例][f-row10]、[mask 实现][f-runtime]

#### 环境、交付、evaluator

harness 项目要求 Python **>=3.12**；这不同于样例容器的 Python 3.10 base。Docker Hub API 实测 MLflow 镜像 active、Linux amd64、`full_size=14,073,256,440 bytes`；packaging 镜像为 **10,178,975,763 bytes**。`fast` 是免 GPU/较快**评测**，不是轻量镜像。MLflow `repo_settings` 的 shm 是 **4g**、测试 timeout 1,200 秒；packaging timeout 600 秒，样例 GPU 配置为空。[pyproject.toml:1–9][f-pyproject]、[镜像 MLflow][f-mlflow-image]、[镜像 packaging][f-packaging-image]、[真实 rows][f-row9]

官方源码安装入口是 `git clone https://github.com/LiberCoders/FeatureBench.git` → 进入新 checkout 后 `git switch --detach 8d4e347ec57546685c5a87e8676bf575db022ea6` → `uv sync` → `source .venv/bin/activate`；随后 `cp config_example.toml config.toml`。可只预拉选定镜像 `docker pull libercoders/featurebench-specs_mlflow-instance_efffa87e:latest`，不必使用会获取整个 fast 镜像集合的 `fb pull --mode fast`。本次只读取这些入口，没有执行安装或拉镜像。[LiberCoders/FeatureBench:README.md:62–95][f-install]

在未来独立环境安装此 revision 后，最终输出合并 patch；不要把原仓库代码当任务初始状态：

```bash
fb eval -p predictions.jsonl --dataset LiberCoders/FeatureBench \
  --data-version 76b4a4566e04f4bcc13c35125d4f301791efa736 \
  --split fast --task-id mlflow__mlflow.93dab383.test_numpy_dataset.1beaad57.lv1 \
  --n-concurrent 1
```

prediction 接受 `instance_id`、`model_patch`、可选 `model_name_or_path/n_attempt`；也支持从 `test_result.git_patch` 归一化。评测产物为 `eval_outputs/{instance_id}/attempt-{n}/test_output.txt`、`patch.diff`、`report.json`。`fb eval` 对应 `featurebench.harness.run_evaluation`，里面的 ThreadPoolExecutor 是**并行评测实例**，不是已实现本文三个 arms。[harness/utils.py:23–67][f-pred]、[docs/harness_cli_arg.md:74–84][f-cli]、[run_evaluation.py:1215–1217][f-workers]

不启用 `--white`，不启用去掉 interface descriptions 的 `--without`，保持本报告依赖的公开接口规格。源码会在 white-box 模式保留测试、在 without 模式删掉长接口部分，都会改变泄漏边界或前缀机制。官方完整 inference runner 默认把单个 `instruction` 交给一个 agent；本文的 fanout 要另标 Octopus adapter，不能称为原生 FeatureBench 协议。[featurebench/infer/run_infer.py:947–957][f-infer]

### 7.2 CORE-Bench：最直接的三个真实执行分支，但长背景不够

当前原始数据 **45 train + 45 test** 已核验：train 是公开 JSON，test 是公开 `.json.gpg`，官方公开口令 `reproducibility`，本次使用现有 gpg 成功解密并计数，未安装环境。90 个论文 capsule × easy/medium/hard = 270 个难度视图，不是 270 个独立仓库。代码/card MIT，具体 capsule 的代码与数据许可仍需逐项看。[siegelz/core-bench:README.md:13–46][c-readme]、[HF 数据入口][c-hf]

实际输入 `capsule-6460826`（CULP，train）要求执行 `iris_sample.py`、`zoo_sample.py`、`wine_sample.py`，报告各数据集 CN/AA accuracy 共六个问题。公开 archive HTTP 200，**13,752 bytes**；里面三个脚本各自载入数据、固定 train/test split、调同一 `culp.classifier.culp`，没有读取另一个脚本的输出。因此 **Iris / Zoo / Wine 三个执行 arms 是真实输入支持的独立分解**。[任务 JSON][c-train]、[capsule/DOI][c-doi]、[原始制品][c-archive]

但 `code/run.sh:4–6` 逐条顺序调用 Python，官方并不原生并发；所有 `code/` 下 `.py/.md/.sh` 总共 **8,787 bytes**，三个分支基本只需各跑一个命令。把它做成长时间、多 LLM arm 的重型任务没有充分依据。可用于低成本独立分支负/正机制对照，不是主工程候选。

实读还发现 Iris/Zoo 脚本循环输出 `CN/AA/RA/CS` 名称，却把调用参数硬编码为 `link_predictor='CS'`；Wine 才传入循环变量。CORE 是复现发布结果，不是修算法题，不能悄悄“纠正”它后再与官方数值比较。环境 Dockerfile 是 Python 3.6.3 / Ubuntu 16.04、networkx 2.1、pandas 0.23.1、scikit-learn 0.19.2；代码 MIT、data/LICENSE 为 CC0。archive 中这些事实可直接复核，本次未运行这个老环境或验证 Code Ocean registry pull。[制品内 `code/iris_sample.py:9–13`、`code/zoo_sample.py:10–13`、`code/wine_sample.py:19–23`、`environment/Dockerfile:1–8`][c-archive]

```bash
curl -fL 'https://raw.githubusercontent.com/siegelz/core-bench/e32a2980e72fe6eb04ee04eb749458f570625663/benchmark/dataset/core_train.json' -o core_train.json
curl -fL 'https://corebench.cs.princeton.edu/capsules/capsule-6460826.tar.gz' -o capsule-6460826.tar.gz
```

任务向 agent 提供 `task.txt` 及 capsule，输出 `environment/report.json`，key 是原问题、value 是回答。旧 evaluator `benchmark/evaluations.py::eval_result_json` 做数值/文本答案检查，`score_results` 汇总；不要为数值任务默认打开可选 LLM summary。[README:94–125][c-readme]、[benchmark/evaluations.py:42–104][c-eval]

当前 README 推荐 **HAL**，称旧 harness 不再维护。推荐的 test 入口为：

```bash
# 未来已安装 HAL 及其 corebench dependencies，已按官方步骤解密 test metadata 后：
hal-eval --benchmark corebench_hard --agent_dir agents/core_agent \
  --agent_function main.run --agent_name CORE-Agent --max_tasks 1 \
  -A model_name='<configured-model>'
```

此命令运行 test，并不自动选中上述 train CULP。若只想按公开 train 样例做适配，可先从 train JSON 选出 CULP **整条记录供 evaluator 独占**，旧可复现入口为 `python main.py --experiment_name culp-pilot --agent_dir <adapter-dir> --agent_script <adapter.sh> --dataset_file <culp-only.json> --no_gpu --task_limit 1 --benchmark_level codeocean_hard`。这属于旧 harness 的自定义 pilot，不冒充当前 HAL leaderboard 运行。[HAL README:257–286][c-hal-readme]、[HAL corebench.py:31–55、223–238][c-hal-code]、[旧 main.py:12–24][c-main]

**泄漏边界**：metadata 的 `results` 同时装着问题 keys 和答案 values，只给 agent keys；easy 本来就提供 results，不能拿它证明复现能力。medium/hard 去掉原结果；hard 还去掉 `REPRODUCING.md`、`environment/`、`code/run.sh` 等执行提示。本文读取这些文件是研究任务源，未来 hard-mode 背景包不可照抄它们。[siegelz/core-bench:benchmark/benchmark.py:208–234][c-boundary]

反例 `capsule-7038571` 虽写 “run main.py three times”，实际任务是 preprocessing → embedding → link prediction，属于依赖链，不是三个独立实验。[同一 train JSON][c-train]

### 7.3 SWE-smith 官方轨迹：CPU probe 有用，不能当原生协作轨迹

数据 revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`，不 gated，MIT。metadata 实际格式 splits 为 tool 24,100、xml 26,076、ticks 25,826，总下载大小 **3,171,137,897 bytes**。card 正文仍只介绍最初用于训练的 5,017 条；仓库 README 说 26K trajectories。要报告当前具体 split 的实际记录数，**不能把三个编码格式相加成独立 agent 经历**。[当前 card/schema][s-traj-card]、[SWE-bench/SWE-smith:README.md:64–66][s-readme]

配套 task 数据当前 main revision `ea6d7173829c7ec8fa16c22055699ff2e9188091` 有 **59,136 train 行**，已经不是 README 的约 52K；其任务为 bug generation 等方式生成，不能一概称自然发生的真实 GitHub issues。[任务数据 API][s-task-api]、[SWE-smith README][s-readme]

可下载的一行 tool 轨迹：

```bash
curl -fL 'https://datasets-server.huggingface.co/rows?dataset=SWE-bench/SWE-smith-trajectories&config=default&split=tool&offset=0&length=1' -o trajectory-row.json
```

实读 `instance_id=django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u`，`traj_id=django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u.8qa84d2e`。row 有 `messages`（**JSON 字符串，需再 json.loads**）、`instance_id`、`resolved`、`model`、`traj_id`、`patch`；本例 resolved=true，49 messages：1 system、1 user、24 assistant、23 tool；文本 content 合计 **36,456 字符**，最初 user 仅 **2,741 字符**。content 可能是 string 或 text-part list，不能把 `len(list)==1` 当作一个字符。[实际 row][s-traj-row]

因此长轨迹主要是**逐步累积的历史**，不是一开始就有 36K 字符公共背景。提供现有历史供 tokenizer replay 可行，但拿后续工具/assistant/patch 作为原题输入会泄漏解题过程。新求解仅保留允许的初始消息；做离线 replay 要明确是已知轨迹回放，不能报告新 solve rate。

更重要的是转换损失：`swesmith/train/traj_mgr/utils.py:18–36` 从最后一步的 history/query 提取消息，并注明只是一种近似；xml/ticks 转换会改 role/模板，xml 甚至替换 system prompt。dataset schema 没有每个请求的起止时间或 branch DAG。它适合 tokenizer/template/LCP 检查，不足以重建真实调度重叠或并发 cache 压力。[SWE-bench/SWE-smith:swesmith/train/traj_mgr/utils.py:18–36、45–60、81–101][s-convert]

配套代码的求解评分入口是 `python -m swesmith.harness.eval --dataset_path SWE-bench/SWE-smith --predictions_path <predictions.jsonl> --run_id <run-id> --workers 1`（还需按该版本 task 配置准备环境）；trajectory conversion 是 `python -m swesmith.train.traj_mgr.collect_trajs --traj_dir ... --eval_dir ...`，后者只转换训练记录，**不是任务 evaluator**。[swesmith/harness/eval.py:1–10][s-eval]、[collect_trajs.py:1–27][s-collect]

## 8. 最小 pilot：先证明输入与并发机制，再考虑 GPU

以下是**建议协议/成本上限，不是本次实验结果**。

| 阶段 | 最小范围 | 要得到的决策量 | 成本/停止条件 |
|---|---|---|---|
| CPU 输入探查 | FeatureBench MLflow row 9、packaging row 0；Multimodal Lighthouse；各一个真正不同的任务组作竞争候选 | 最终模板后的 tokens、pairwise LCP、完整公共块数、合法输入占比；每个 arm 的互补产物是否具体 | 只取小型 rows/源文件；0 GPU、0 LLM 调用。长输入不足就停，不复制/填充无关文档 |
| 调度/缓存可观测性检查 | 每个 task 2–3 arms；允许自然工具等待后重入；至少两个有价值的不同前缀/后缀组 | 同时 active consumers、请求时间线、共享块与独特块 footprint、可淘汰集合、缓存命中/拒绝/淘汰/adoption 的实际机会 | 如果仍只有 1 consumer、没有竞争或不同策略决策仍同形，不能用 policy draws 数代替机制证据 |
| 有授权后的极小闭环 | 优先 1 个 MLflow schema 任务 + 1 个 Lighthouse 任务；每策略最多 2 tasks × 3 arms，最后合并交付 | 成功/失败、有效分支贡献、官方最终评分，以及服务侧缓存指标 | 单 task 建议预设 60–100 次总模型调用或总输出预算；这是待选择的工程上限，不是已核验需求。先不扩展到 full benchmark |

容量应按被测模型/缓存实现的**实际块 footprint**选择，不凭“上下文很长”设一个任意极小 KV 容量。首个 probe 可报告 `P=LCP(tokens_i,tokens_j)` 与 `floor(P/block_size)`；真正策略实验还需有多个未来会重用的块组，以及共享域、cache salt、adapter、视觉输入的一致性。试图把一个任务所有 source 都塞入每个 arm、或把 budget 压到连单个正常请求都放不下，只是在制造压力，不是发现自然机制。[块 identity 源码][kv-source]

不要混淆三层并发：LLM arms 的请求重叠、arm 的 shell/test 并发、官方 evaluator 同时跑多个 task。为了降低 Multimodal 的 stdout/stderr 噪声，完全可以保持最终 evaluator `-j 1`，而单独研究此前 agent 请求的并发。需报告这种区别。[Multimodal 并行评测警告][mm-repo-readme]

| 候选 pilot 的可见成本 | 已核验事实 | 仍不能保证 |
|---|---|---|
| FeatureBench MLflow | JSON 一行即可 probe；完整运行预制镜像约 14.07 GB；shm 4g；测试 timeout 1,200s | 下载/解压用时、实际 RSS、模型调用成本；57.2s 的旧发布均值不能套到这台机器/这版数据 |
| Multimodal Lighthouse | 两图约 156 KB；镜像标签报告约 0.918 GB；harness 默认测试 timeout 1,800s | Chromium/依赖重建、在线目标网页可用性、代理 solve rate |
| TAC budget / HR | 两表合计约 27 KB；完整公司基础设施需 30+ GB；init 文档可至 10min | 非 NPC task 是否能在全部旧镜像上无 key 完整启动；HR 环境 LLM 成本不为零 |
| WorkArena++ | 代码与 Playwright 可获取，但 instance gate 未批准 | 实际任务 observation、长度、实例数量/额度、等待时间与并发成本 |
| CORE CULP | archive 13.8 KB，CPU 科学计算脚本，无训练 GPU 指令 | 老 Python/registry 当前可运行性；长背景不足、无需复杂 LLM 的反证仍存在 |
| SWE-smith | 可只取一行；全格式下载约 3.17 GB | 格式化 history 不含真实并发时间线，不能据此测在线服务收益 |

## 9. 可以直接接 CPU tokenizer 的样本接口

**最有用的免环境入口是 HF `/rows` 的 FeatureBench `fast` offset 9/0，以及 SWE-smith tool offset 0；最稳妥的 revision-pinned 文本入口是 Multimodal 的 GitHub `problem_statement.md`。** 不要求先装 Docker/ServiceNow，也不需要模型服务。

固定版本、无 gold 字段的 Multimodal 文本下载命令：

```bash
curl -fL 'https://raw.githubusercontent.com/SWE-bench/swe-bench-multimodal-tasks/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/tasks/GoogleChrome__lighthouse-10176/problem_statement.md' -o lighthouse-task.md
```

下面是一个只用 Python 标准库的**下载/白名单投影示例**（本报告展示代码，不创建 repo 中的额外工具，也未运行 tokenizer）。它只输出 task 输入，不输出 masking patch、gold tests 或参考答案：

```python
import json
import urllib.request

url = ("https://datasets-server.huggingface.co/rows"
       "?dataset=LiberCoders/FeatureBench&config=default&split=fast&offset=9&length=1")
with urllib.request.urlopen(url, timeout=60) as response:
    payload = json.load(response)
item = payload["rows"][0]
if item["truncated_cells"]:
    raise RuntimeError(f"Truncated input: {item['truncated_cells']}")
row = item["row"]
if row["instance_id"] != "mlflow__mlflow.93dab383.test_numpy_dataset.1beaad57.lv1":
    raise RuntimeError("Dataset order/version changed; do not silently reuse this offset")
public = {k: row[k] for k in ("instance_id", "repo", "base_commit", "problem_statement")}
print(json.dumps(public, ensure_ascii=False))
```

**版本边界**：viewer API 没有在上述 URL 中固定 revision，offset 随数据变化；上述 ID 检查只防行错位，**不证明内容等于指定 revision**。正式复现使用本报告给出的 pinned parquet + `load_dataset(..., revision=...)`，并记录实际 revision。不要将本次读到的 main 样例配上默认 v1.1 的 evaluator。

对 trajectory 使用 `messages = json.loads(row["messages"])`；保留原始角色/工具调用字段及 content parts，让目标模型自己的 chat template 处理。用于初始任务 probe 只取第一条 system/user，不包含未来 assistant/tool；用于离线完整 replay 则明确标注泄漏了已有解题历史、不能衡量新求解。

探针必须使用未来实际服务采用的 tokenizer 与 chat template，比较最终完整请求 token IDs，而不是 `len(problem_statement)`、UTF-8 bytes 或“英文约四字符一个 token”。对于 Multimodal，纯文本探针的结论只覆盖文本部分。**本次没有选择一个未知模型的 tokenizer，故没有虚构任何 exact token-prefix 长度。**

## 10. 当前明确的阻塞、负面结论与交付范围

| 项 | 状态与影响 |
|---|---|
| WorkArena 官方实例 | **实际阻塞**：manual-gated，匿名 `instances.json` HTTP 401；只核验了生成器/工作流代码，未取得真实运行 observation |
| SWE-bench Multimodal | **不是当前访问阻塞**：v2 的 480 test 公开、本地 evaluator 已支持；不能沿用旧 hidden-test 说法。但 card 正文计数过时 |
| FeatureBench | 无 metadata gate；大镜像与 task generation/masking 是实际成本和真实性边界；API main 与 v1.1 tag 不同 |
| TAC | task.md/xlsx/csv 公开；完整服务恢复与旧镜像 CLI 未运行验证；NPC/部分 grader 需要另行配置 LLM；文档 `--output_path` 与源代码 `--result_path` 不一致 |
| CORE | test GPG 是公开口令包装，本次成功访问，不应称为需申请数据；Code Ocean 运行镜像未 pull，老环境可运行性未验证 |
| 轨迹证据 | 下载得到的是顺序 agent history 格式，不是原生协作调度记录；不支持“发现了 native multi-arm benchmark”的结论 |
| 最终机制 | 尚未观测到 GPU 加速、cache-policy 优势、非零 solved 或有意义 contention；这些必须由后续明确授权的执行给出 |

本次只交付本文与一个小型**派生 metadata JSON**，不向 repo 保存原始 gold patch、evaluator 答案、完整轨迹或服务备份；未修改其他 repo 文件、未安装环境、未提交 commit。推荐优先级始终是：**先检验 FeatureBench 的互补接口 arms 是否真的值得并发；再用 Multimodal 的原始 issue 验证真实性。若严格要求 benchmark 原生并发，当前结果是没有已核验满足者，而不是勉强推荐一个串行基准。**

## 11. 版本索引与一手来源

以下 SHA 为检索时正常 revision identity；没有将可变 `latest` 镜像 tag 当作永久不变制品。镜像 API 回传的 digest/更新时间保存在 metadata JSON。

| 来源 | 核验 revision |
|---|---|
| SWE-bench/SWE-bench | `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` |
| SWE-bench/swe-bench-tasks | `3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d` |
| SWE-bench/swe-bench-multimodal-tasks | `8d3ed32f37af9ceb8b5778b92a0088384ed057d4` |
| HF Verified / Multimodal | `78f471bf655a3137b2e8a75af1501690ec009ec3` / `4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5` |
| TheAgentCompany/TheAgentCompany | `98b68ef82a47690c316f42fddb05baafaab56851` |
| ServiceNow/WorkArena / HF WorkArena-Instances | `a772230a94cf1caf4166b8ead3983f3b3786455b` / `2a7d0811bf037b7cb05de266912a08c973360764` |
| LiberCoders/FeatureBench code / HF main | `8d4e347ec57546685c5a87e8676bf575db022ea6` / `76b4a4566e04f4bcc13c35125d4f301791efa736` |
| siegelz/core-bench code / HF | `e32a2980e72fe6eb04ee04eb749458f570625663` / `18ac8edf2532d9edb9d13ae71f715410de6ee5a0` |
| princeton-pli/hal-harness | `16bb03ebc11577fb5ea6dc8bb6c968387085e6aa` |
| SWE-bench/SWE-smith code / HF trajectories | `9b74ac08118a85c39c356802f7961893af73e07f` / `08e109b4a59eaeebf80e4675cd125d42e7ac99a4` |
| vllm-project/vllm | `fe4d4109c24063fb5e60ede9d48a1c0b3e9e4cea` |

[sv-card]: https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/blob/78f471bf655a3137b2e8a75af1501690ec009ec3/README.md
[sv-row]: https://datasets-server.huggingface.co/rows?dataset=SWE-bench/SWE-bench_Verified&config=default&split=test&offset=0&length=1
[swe-license]: https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/LICENSE
[swe-readme]: https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/README.md#L60-L114
[swe-news]: https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/README.md#L34-L38
[swe-machine]: https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/README.md#L121-L126
[swe-args]: https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/swebench/harness/run_evaluation.py#L797-L853
[sv-task]: https://github.com/SWE-bench/swe-bench-tasks/blob/3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d/tasks/astropy__astropy-12907/task.yaml#L1-L15
[sv-docker]: https://github.com/SWE-bench/swe-bench-tasks/blob/3d07b464b7b311a0cbfb5ed5b2d8a3b96f84a33d/tasks/astropy__astropy-12907/Dockerfile#L23-L123
[mm-card]: https://huggingface.co/datasets/SWE-bench/SWE-bench_Multimodal/blob/4e6662d51c48e475f7f346e4fa09a6f8b31fcaa5/README.md
[mm-row]: https://datasets-server.huggingface.co/rows?dataset=SWE-bench/SWE-bench_Multimodal&config=default&split=test&offset=0&length=1
[mm-local]: https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/swebench/harness/run_evaluation.py#L715-L719
[mm-assets]: https://github.com/SWE-bench/SWE-bench/blob/02e7a74ffd0b707aab73d203fe87bdc7c76afc8e/swebench/harness/run_evaluation.py#L172-L185
[mm-dir]: https://github.com/SWE-bench/swe-bench-multimodal-tasks/tree/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/tasks/GoogleChrome__lighthouse-10176
[mm-input]: https://github.com/SWE-bench/swe-bench-multimodal-tasks/blob/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/tasks/GoogleChrome__lighthouse-10176/problem_statement.md#L1-L20
[mm-task]: https://github.com/SWE-bench/swe-bench-multimodal-tasks/blob/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/tasks/GoogleChrome__lighthouse-10176/task.yaml#L1-L17
[mm-docker]: https://github.com/SWE-bench/swe-bench-multimodal-tasks/blob/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/tasks/GoogleChrome__lighthouse-10176/Dockerfile#L1-L80
[mm-eval]: https://github.com/SWE-bench/swe-bench-multimodal-tasks/blob/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/tasks/GoogleChrome__lighthouse-10176/eval.sh#L129-L134
[mm-repo-readme]: https://github.com/SWE-bench/swe-bench-multimodal-tasks/blob/8d3ed32f37af9ceb8b5778b92a0088384ed057d4/README.md#L28-L41
[mm-image]: https://hub.docker.com/v2/repositories/swebench/sweb.eval.x86_64.googlechrome_1776_lighthouse-10176/tags/latest
[lh-audit]: https://github.com/GoogleChrome/lighthouse/blob/a5dbdd2eff4238213dcaad404d0e06d8d5d2489b/lighthouse-core/audits/dobetterweb/js-libraries.js#L27-L63
[lh-collector]: https://github.com/GoogleChrome/lighthouse/blob/a5dbdd2eff4238213dcaad404d0e06d8d5d2489b/lighthouse-core/lib/stack-collector.js#L39-L91
[lh-packs]: https://github.com/GoogleChrome/lighthouse/blob/a5dbdd2eff4238213dcaad404d0e06d8d5d2489b/lighthouse-core/lib/stack-packs.js#L40-L75
[lh-package]: https://github.com/GoogleChrome/lighthouse/blob/a5dbdd2eff4238213dcaad404d0e06d8d5d2489b/package.json
[t-readme]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/README.md
[t-images]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/README.md#L38-L80
[t-setup]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/docs/SETUP.md#L14-L50
[t-release]: https://github.com/TheAgentCompany/the-agent-company-backup-data/releases/tag/setup-script-20241208
[t-eval-doc]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/docs/EVALUATION.md
[t-eval-source]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/base_image/eval.py#L77-L105
[t-init]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/base_image/init.sh#L10-L26
[t-run]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/evaluation/run_eval.sh#L91-L119
[t-budget-task]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/finance-budget-variance/task.md#L1-L9
[t-budget-deps]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/finance-budget-variance/dependencies.yml#L1
[t-budget-data]: https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/finance-budget-variance/budget.xlsx
[t-actual-data]: https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/finance-budget-variance/actual_spending.xlsx
[t-budget-eval]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/finance-budget-variance/evaluator.py#L41-L76
[t-hr-task]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/admin-employee-info-reconciliation/task.md#L1-L12
[t-hr-csv]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/admin-employee-info-reconciliation/employees.csv
[t-hr-deps]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/admin-employee-info-reconciliation/dependencies.yml#L1
[t-pm-task]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-projects-analytics/task.md#L1-L12
[t-sde-task]: https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/sde-report-unit-test-coverage-to-plane/task.md#L1-L3
[w-readme]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/README.md
[w-license]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/LICENSE#L1-L13
[w-access]: https://huggingface.co/datasets/ServiceNow/WorkArena-Instances
[w-sampler]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/__init__.py#L124-L169
[w-curriculum]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/tasks/compositional/utils/curriculum.py#L55-L128
[w-levels]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/tasks/compositional/__init__.py#L12-L54
[w-task-id]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/tasks/base.py#L98-L105
[w-onboard]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/tasks/compositional/onboard_user.py#L112-L221
[w-base]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/src/browsergym/workarena/tasks/compositional/base.py#L215-L236
[w-demo]: https://github.com/ServiceNow/WorkArena/blob/a772230a94cf1caf4166b8ead3983f3b3786455b/README.md#L170-L193
[f-readme]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/README.md#L15-L26
[f-card]: https://huggingface.co/datasets/LiberCoders/FeatureBench/blob/76b4a4566e04f4bcc13c35125d4f301791efa736/README.md
[f-refs]: https://huggingface.co/api/datasets/LiberCoders/FeatureBench/refs
[f-row0]: https://datasets-server.huggingface.co/rows?dataset=LiberCoders/FeatureBench&config=default&split=fast&offset=0&length=1
[f-row1]: https://datasets-server.huggingface.co/rows?dataset=LiberCoders/FeatureBench&config=default&split=fast&offset=1&length=1
[f-row9]: https://datasets-server.huggingface.co/rows?dataset=LiberCoders/FeatureBench&config=default&split=fast&offset=9&length=1
[f-row10]: https://datasets-server.huggingface.co/rows?dataset=LiberCoders/FeatureBench&config=default&split=fast&offset=10&length=1
[f-pipeline]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/docs/pipeline.md#L6-L18
[f-runtime]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/featurebench/infer/runtime.py#L128-L200
[f-mask]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/featurebench/harness/utils.py#L276-L287
[f-pyproject]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/pyproject.toml#L1-L9
[f-mlflow-image]: https://hub.docker.com/v2/repositories/libercoders/featurebench-specs_mlflow-instance_efffa87e/tags/latest
[f-packaging-image]: https://hub.docker.com/v2/repositories/libercoders/featurebench-specs_packaging-instance_c393a6a8/tags/latest
[f-cli]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/docs/harness_cli_arg.md#L16-L84
[f-pred]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/featurebench/harness/utils.py#L23-L67
[f-workers]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/featurebench/harness/run_evaluation.py#L1215-L1217
[f-infer]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/featurebench/infer/run_infer.py#L947-L957
[c-readme]: https://github.com/siegelz/core-bench/blob/e32a2980e72fe6eb04ee04eb749458f570625663/README.md
[c-hf]: https://huggingface.co/datasets/siegelz/core-bench/tree/18ac8edf2532d9edb9d13ae71f715410de6ee5a0
[c-train]: https://raw.githubusercontent.com/siegelz/core-bench/e32a2980e72fe6eb04ee04eb749458f570625663/benchmark/dataset/core_train.json
[c-doi]: https://doi.org/10.24433/CO.0609cc4f-8b95-4d94-8fd0-9456d262b3a5
[c-archive]: https://corebench.cs.princeton.edu/capsules/capsule-6460826.tar.gz
[c-eval]: https://github.com/siegelz/core-bench/blob/e32a2980e72fe6eb04ee04eb749458f570625663/benchmark/evaluations.py#L42-L104
[c-main]: https://github.com/siegelz/core-bench/blob/e32a2980e72fe6eb04ee04eb749458f570625663/main.py#L12-L24
[c-boundary]: https://github.com/siegelz/core-bench/blob/e32a2980e72fe6eb04ee04eb749458f570625663/benchmark/benchmark.py#L208-L234
[c-hal-readme]: https://github.com/princeton-pli/hal-harness/blob/16bb03ebc11577fb5ea6dc8bb6c968387085e6aa/README.md#L257-L286
[c-hal-code]: https://github.com/princeton-pli/hal-harness/blob/16bb03ebc11577fb5ea6dc8bb6c968387085e6aa/hal/benchmarks/corebench.py#L31-L55
[s-traj-card]: https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/blob/08e109b4a59eaeebf80e4675cd125d42e7ac99a4/README.md
[s-traj-row]: https://datasets-server.huggingface.co/rows?dataset=SWE-bench/SWE-smith-trajectories&config=default&split=tool&offset=0&length=1
[s-readme]: https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/README.md#L46-L79
[s-task-api]: https://huggingface.co/api/datasets/SWE-bench/SWE-smith
[s-convert]: https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/swesmith/train/traj_mgr/utils.py#L18-L101
[s-eval]: https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/swesmith/harness/eval.py#L1-L10
[s-collect]: https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/swesmith/train/traj_mgr/collect_trajs.py#L1-L27
[kv-source]: https://github.com/vllm-project/vllm/blob/fe4d4109c24063fb5e60ede9d48a1c0b3e9e4cea/vllm/v1/core/kv_cache_utils.py#L584-L648

[f-install]: https://github.com/LiberCoders/FeatureBench/blob/8d4e347ec57546685c5a87e8676bf575db022ea6/README.md#L62-L95
