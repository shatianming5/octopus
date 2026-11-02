# Octopus benchmark discovery：长文档多问题、共享语料与真实前缀轨迹

> 历史调研（2026-09-11），不随实现更新。"Octopus""P1""adoption""APC 对照"指当时的 KV 保留 / admission 研究线，其代码与配套文档已在 #219 删除（恢复点 `4f82ad5`）；当前系统见 [README.md](../README.md)。

核验日期：**2026-09-11 UTC**。仅使用公开一手来源；不涉及 SWE-bench、TheAgentCompany、WorkArena++。本次执行的是 HTTP 下载/元数据检查、CPU JSON 解析与结构计数、源码/论文阅读，**没有模型运行、GPU、LLM API、依赖安装或提交**。

## 1. 结论与排序

**原文 token-prefix CPU probe 首选 QuALITY v1.0.1 `htmlstripped.test`；真实到达与缓存关系的互补输入首选 Mooncake FAST'25 `toolagent_trace.jsonl`；官方原生 corpus-first 强正例首选 LOFT-QUEST 128k。** QuALITY 有同文档真实独立问题但没有 arrival；Mooncake 有生产采样 arrival 和累计 prefix block 身份，但没有原文/token IDs/工具 DAG/业务答案；LOFT 有官方统一 corpus 提示，但不是线上 agent trace。[Q1][L1][L2][T1][T2]

用户给出的 P1 现象——一次仅一个 continuation，policy draws 存在但 eviction/rejection/adoption delta 均为零——是本报告研究前提，**本次未重新测量**。新 workload 必须使有效并发、可复用块与有限容量同时出现；任务更长、更“agentic”或共同话题更多并不自动满足它。

| 优先级 / 候选 | 精确前缀证据 | 自然并行、争用潜力 | 立即可取输入 / 核心限制 |
|---|---|---|---|
| A1 QuALITY test | 116 篇文章的两个 question-set 记录全部逐字符一致；2,128 问 | 每篇 14–20 个不同问题；多文章可构成不同工作集 | 7,104,581 B JSONL，无答案标签；document-first 是输入保持的 serving 适配，不是生产 prompt/arrival |
| A2 LOFT-QUEST 128k | 官方固定 corpus+few-shot 在当前 query 前 | 328 passages / 100 test queries；单 corpus 热驻留也可能没有 eviction | 1,805,826 B ZIP，corpus 498,151 B；禁止按 qrels 给每题裁出 gold context |
| A3 Mooncake Tool&Agent | 官方定义 512-token cumulative prefix hashes | 23,608 线上采样请求，带真实 arrival；最适合容量/调度结构审计 | 4,415,857 B JSONL；不是原文 tokenizer probe，也没有“有用 arms”评分 |
| B1 QASPER | test 416 论文 / 1,451 问，384 论文有至少两问 | 同论文独立科学问答，多种文档工作集 | 3,865,061 B test+evaluator 包；原 LED reader 是 question-first，长前缀需显式适配 |
| B2 MultiDoc2Dial | 488 篇知识库文档跨对话复用、历史逐轮增长 | 对话间可并行，同一对话后续轮有因果依赖 | 6,868,509 B ZIP；知识库共享不等于 prompt 前缀相同 |
| C MuDABench | 589 个 PDF ID 大量跨题重复；PDF 引用重合已测，文本/token 前缀未测 | 官方 plan 下 per-doc extraction 确有并行分支 | JSON 小，但 PDF 总计 4.06 GB；metadata 内有金标数值，默认文档服务在远端 |
| D DeepResearch Bench / BrowseComp | 问题各异，未提供统一长资料正文 | 不同任务能并行，子任务划分由 agent 产生 | 查询很小；共同主题/网站不是长 exact prefix |
| D MLE-bench | 不同比赛主要共享通用 agent 指令 | 同比赛多试验有用，但无原生可评分 arms DAG | 只取 description/config 做低共享对照；Kaggle 数据/环境重 |
| 对照 BurstGPT | 只有长度，部分有 session ID，**无 prefix hash/文本** | 线上 arrival；第三个分片有完整响应时长 | 适合 timing/length 对照；用 ShareGPT 补内容后是合成 replay |

排序是针对 Octopus 的判断，不是 benchmark 官方排名。表内实际计数保存在 `results/benchmark-discovery/research-data/`。以下明确区分**已读取源码/数据事实、研究推断、未验证和模拟**。

## 2. 必须分开的三件事

| 口径 | 合格证据 | 不合格替代 |
|---|---|---|
| identical prefix-token reuse | 完整 chat template 渲染后的 token IDs 从位置 0 相同，模型/tokenizer/adapter/position/cache salt 相容；或官方定义的 cumulative prefix hash | 同 topic、同文档 ID、共同检索库、相同后缀或正文包含关系 |
| 自然有用并行 | 同文档不同已发布问题互不依赖；不同对话独立；官方 per-document 分支 | 同请求重复 K 次、把单续写强拆、将最终答案或未来工具结果提前注入 |
| 任务 vs arrival/tool trace | question ID 证明问题存在；timestamp 证明请求到达；tool start/end+parent ID 才能还原工具依赖/等待 | 文件顺序就是 arrival、turn_id 就是时间戳、最终报告就是执行过程、elapsed time 就是 TTFT |
| CPU 结构 vs引擎测量 | 字符/块身份、LCP、明确假设的无驱逐上界和容量仿真 | 将其称为实测 APC 命中率、TTFT/GPU savings/吞吐或 Octopus gain |

`固定说明 + 完整文档 + 当前问题` 不改问题或资料，适合独立问答 serving，但属于**公开数据上的模板适配**。它不会自动验证动态规划、arm rejection/adoption 或跨 arm 综合答案。所有系统必须得到相同输入、模板和顺序，不 padding，不依据 gold、预测正确性或有利于 Octopus 的 overlap 选择样本。

## 3. 用户指定候选核验

### 3.1 MLE-bench：真实 ML 工程，当前发现阶段不匹配

**规模/划分。** 官方 75 个 Kaggle competitions；实际读取 complexity split 文件为 low 22 / medium 38 / high 15，另有 7 个 dev competition IDs。它们不是每场比赛的训练/验证/测试数据划分；各比赛自己的 preparation 还重划 public train/test 和 private answers。官方称全量约 3.3 TB、Lite 158 GB；推荐单 run 24 小时、36 vCPU、440 GB RAM、24 GB A10。[M1]

**访问/实例。** 具体真实 task IDs：`leaf-classification`、`spooky-author-identification`。固定 GitHub revision 的 `mlebench/competitions/<id>/description.md`、`config.yaml`、`prepare.py`、`grade.py` 可匿名取；实际数据需 Kaggle API 身份和比赛条款。本次没有下载比赛数据，具体账户授权**未验证**。`leaf-classification/prepare.py` 从公开训练数据重划 holdout、将答案写到 private 目录；`grade.py` 读取私有答案并计算提交指标。[M2]

**评价/金标/许可。** 官方入口 `mlebench prepare -c leaf-classification`、`mlebench grade-sample submission.csv leaf-classification`；批量评分 JSONL 字段为 `competition_id,submission_path`。agent 不得看到 private answers、旧公开 test labels、已发布 solution/submission。README 列有若干已知泄漏/制备问题，未来选择真实比赛必须单独确认。MIT 明确**只覆盖代码，不覆盖下载的外部数据**。[M1][M3]

**Octopus 判断。** 比赛内可做并行模型试验，但这是被测 agent 的设计，不是 benchmark 自带的独立任务 DAG。同一数据目录/CSV 不等于同一 LLM prefix；训练体量大不等于模型输入长。仅保留 descriptions 作低共享负对照，不运行 `prepare --all` 或 `git lfs fetch --all` 做发现。

### 3.2 DeepResearch Bench：100 个输入立刻可取，无共享长资料包

此处指原版 `Ayanami0730/deep_research_bench`、论文 `2506.11763`，**非 DeepResearch Bench II**。实际 `data/prompt_data/query.jsonl` **100 行、100 个不同 prompt，50 zh / 50 en**，38,080 B，每行为 `{id,topic,language,prompt}`。README 的 22 domains 和 96,147 anonymized user queries 是任务构造的需求分布来源，**不是公开 96,147 条 benchmark 请求或 timing trace**。已核验接口没有原版 train/dev/test 分割。[D1][D2]

| 官方 ID | 实际研究任务 | 共享边界 |
|---|---|---|
| `1` | 中国九阶层实际收入/财务状况与中产分析 | 只附 query，未附共同文档正文 |
| `2` | 全球综合实力前十保险公司比较，评估在中国未来发展潜力 | 同类来源可能交叉，但网页正文、顺序和 tool timing 未给定 |

**评价/访问。** Apache-2.0 仓库；输入 query 可匿名 raw 下载。RACE 源码加载 `criteria.jsonl`、reference reports，对 target/reference reports 用 LLM 评判；FACT 做 claim–URL 提取、去重、抓取和支持性判断。当前文档需 Python 3.9+、评审模型 API，FACT 另需 Jina；本次未运行。原始 report 输出不是中间执行 trace。[D1][D3]

**泄漏。** 输入只用 query；criteria、reference/cleaned reports、已有模型答案隔离在 evaluator。agent 输出 schema 是 `{id,prompt,article}`，评价入口 `bash run_benchmark.sh`。不要把 100 个任务塞入人为统一 dossier 来“获得共享 Brain”，那会改变原任务并放大缓存收益。

### 3.3 BrowseComp：1,266 个加扰问题，未附网页快照或轨迹

实际官方 CSV **1,266 行、1,196,283 B**，字段 `problem,answer,problem_topic,canary`；文件名 `browse_comp_test_set.csv`，该公开接口未提供 train/dev。官方 loader 用每行 `canary` 派生 SHA-256 重复密钥，经 base64/XOR 解扰 problem 和 answer。这是避免直接摄入的加扰，不是权限边界或真正隐藏金标。simple-evals 仓库 MIT；CSV 独立数据许可在本次读取的 loader/许可证中**未明确核实**，不把所有网页或 dataset 自动宣称 MIT。[B1]

CSV 无官方 task ID，可用固定 artifact + 1-based data-row 作为**派生 audit key**。第 1 个问题根据一位非洲作家的生平、采访、必读书等线索询问其担任 probation officer 的年份；第 2 个问题依据 1990–1994 年间足球比赛裁判、黄牌和换人线索确定球队。已解扰问题，未将答案值保存到 evidence。问题没有共同长网页正文。[B1；本地 `artifact-structure-audit.json`]

**输入/评价。** 只解扰 problem，不序列化整行，不向 agent 暴露 answer/canary/grader prompt。官方 query template **先 Question**、后 answer 格式，通常开头即分叉。网页检索/浏览工具由 agent 提供，开放网页变化不受固定数据文件约束；评分使用 LLM judge 比对答案。[B1]

**评分器源码警告。** 本次 pinned `grade_sample` 返回 `match.group(0)`（例如 `correct: yes`），调用处却判断 `grade_result == "yes"`，所以按这条路径 yes 也不计 correct。此为**静态源码发现，未执行复现、未修改**；未来不能直接把该 revision 的 stock score 当可靠结果，需先验证 evaluator 并声明修正。CPU input-only 下载不受影响。[B1:89–123]

**判断。** 合适的低共享 browsing 对照；同题多次采样制造的 prefix 不应冒充跨任务自然共享。

### 3.4 MuDABench：原生多文档分支成立，metadata 金标与远端文档必须隔离

**实际规模。** `simple.json` 166 条 / 1,315,567 B，`complex.json` 166 条 / 1,361,284 B；它们是答案粒度/分析复杂度两套，**非 train/test**。每套引用 589 个不同文档、2,457 个“任务×不同文档”关系；531 个文档被多题引用，单文档最多关联 12 题；每题 5–38 个不同文档、154 种不同文档集合。两套逐行对应的 **166 对题文档集合全部相同**，不能把 332 条视作互相独立的共享证据。[U1][U2；本地结构计数]

**下载位置纠正。** README 列 `data/pdf/`，但本次固定 GitHub tree **没有 PDF**。589 个 PDF 真正位于作者 HF dataset `Zhanli-Li/MuDABench`，公开非 gated；所有问题引用的文档 ID 在 HF 都存在。HF tree API 的文件总大小为 **4,057,308,684 B**，最大 58,789,324 B。对一个 816,536 B PDF 做了 HEAD 200；**没有全量 PDF 下载、OCR 或逐页完整性检查**。clone GitHub repo 不等于已拿到文档。[U3；`muda-pdf-storage.json`]

| 官方 ID / 文档 ID | 真实任务例子 |
|---|---|
| `q_5f548113107a56e1` | A 股 2021 年资本充足率最高三家公司的股票代码；7 份文档 |
| `q_32d98d5d6f2f9340` | 同范围还要求资本充足率，是上题 complex 对应题 |
| `7560dc54-81cb-4d28-bbb1-205f060ad678` | 第一个文档；`symbol=002142,year=2021,doctype=A股年报`；实际身份字段叫 `id` |

**最重要的泄漏。** metadata 不只是身份：真实第一项有 `value_capital_adequacy_ratio_pct`、`schema`，其他项还有 `value_*`/`detail_*`。evaluator 源码明确用 atomic metadata 重建 gold facts。因此只删 `source_answer/final_answer` 仍然泄漏。输入应 allowlist `question_id,question` 和文档身份 `id/symbol/year/doctype`，再配原 PDF/独立转换正文；不默认整包序列化 metadata。保留已发布 question_id，不在去金标后重算——官方 ID 生成包含 metadata。[U4][U5]

**自然并行和默认实现。** `agent_runner.py:270–347` 先生成 plan，对每个 plan 的文档调用 `asyncio.gather`；`doc_concurrency` 控制 per-doc 并行。plan 间仍有顺序，之后 normalization/code/final answer 有依赖。这比人为强拆单续写更接近 Brain+arms。但默认 `extract_single_doc(document_id,prompt)` 在 ChatDOC 远端执行，**不能据此断言 PDF 全文进入同一本地 Brain KV**；CLI 默认 question/doc concurrency 仍为 1，要显式配置。[U4]

**评价/成本/许可。** `eval/evaluate.py --eval-mode rag|agent`，主指标 `final_accuracy`；agent 模式还需 `run_log.json/doc_interactions`。默认 agent/评审需模型 API，ChatDOC 需单独服务；本地替代有 PDF parsing/OCR 和确定性文本转换负担，实测成本未知。仓库/HF card Apache-2.0，不代表所有财报发行方版权已被统一重新授权。作者 2026-07 警告部分题因 distant supervision/文档覆盖不可答，优先 annual reports。可按公开 doctype 预注册年报子集，不可按 gold、答案难易或缓存优势筛题。[U1][U5]

**判断。** 后续 agent-native 候选；本次只能先做去金标 JSON manifest，不能用 metadata 金标摘要代替原文。

## 4. 五个更适合的可下载 workload

### 4.1 QuALITY：最低门槛的同文档不同问题

**完整 split 实读。** v1.0.1 HTML-stripped train/dev/test：**150/115/116 文章，2,523/2,086/2,128 问，共 381 篇、6,737 问**。文件分别 300/230/232 JSONL 行，因为每篇两个 writer question sets。train 流式计数未保留全文；dev/test 实读且所有同 article_id 的文本相同。test 没有 `gold_label,writer_label,validation,speed_validation`，但有金标衍生的 `difficult`，输入也应剔除。[Q1][Q2；`quality-train-counts.json`]

**具体同文档例子。** test 文章 `63860`，*Signal Red*，正文 20,979 个 Unicode characters（**不是 token 数**）。原文件第 1 与第 13 行分别有 9/8 问，共 17 个不同问题：

| question ID | 原始问题 |
|---|---|
| `63860_6KOJTPYF_1` | `What does the phrase, "Here she comes," refer to?` |
| `63860_6KOJTPYF_2` | `Has Shano always lived on Mercury?` |
| `63860_GQIZEKU6_1` | `What passage below BEST describes why Shano continued to board the Stardust after the Red Signal?` |

每题四个原始 options 是合法输入，正确选项本身在 options 中不叫泄漏；正确标签不应给出。例子只展示关系，主 probe 用全部 test 或预注册 input-ID 样本，不能只挑有利文章。[Q2:L1,L13]

**schema。** raw row：`{article_id,set_unique_id,...,title,article,questions:[{question_unique_id,question,options,difficult}],url,license}`。推荐规范化 input-only 两表：context `{article_id,title,article}`，task `{task_id,context_id,question,options}`；task_id 取原 question_unique_id，context_id 取 article_id。请求 ID、当前时间、问题序号不放在文档前破坏 prefix；也不附上同组所有其他问题。

**评价和负担。** CPU 下载/JSON/tokenizer 即可，不需要 browser、GPU 或 API。dev 有 gold，可做 overall/hard accuracy；test 标签官方不公开，正确性要走作者 leaderboard，不能声称有本地 test gold evaluator。官方 deprecated baseline task metric 是预测标签 accuracy，原数据标签 1-indexed，应正确映射输出。[Q1:49–87][Q3]

**许可边界。** README 的 `license` 是逐文章字段；实读有 Project Gutenberg 条款、OANC license、若干 CC-BY-4.0 来源。仓库根未识别统一 license；QA annotations 的独立统一许可及所有法域再分发权**未验证**。应保留逐行 license/来源，不重新标成统一 MIT/CC-BY。[Q1:34–47]

**Octopus 匹配度。** test 每篇 14–20 问，比 QASPER 通常每篇几问有更高自然 fan-out。但这些是静态独立 QA，不能验证动态 adoption；多个短回答可能很快结束，真实争用还需不同文档同时活跃。

### 4.2 LOFT：官方 corpus-first；也是 native APC 的强正例

官方总述 6 task categories、35 datasets、4 modalities，不是 35 个问题，也不是所有任务都有每个长度/split。本次实际下载两个 text retrieval ZIP：[L1]

| Corpus / 官方长度标签 | corpus rows | few-shot / dev / test queries | 核验限制 |
|---|---:|---|---|
| QUEST 32k | 87 | 5 / 10 / **无 test 文件** | 不猜测 `32k/test_queries.jsonl` 存在 |
| QUEST 128k | 328 | 5 / 10 / 100 | 主候选；corpus 498,151 B |
| QUEST 1m | 2,858 | 5 / 10 / 100 | 同 query 的长度变体，不是新增独立问题 |
| SciFact 32k | 86 | 5 / 10 / **无 test 文件** | 小 smoke 输入 |
| SciFact 128k | 357 | 5 / 10 / 100 | 第二真实 corpus；553,622 B |
| SciFact 1m | 2,753 | 5 / 10 / 100 | 更大 context 不代表更自然并发 |

QUEST test `qid=11425373`：`Novels about families set in Boston and New England.`；`11879839`：`2009 Indian, or by Chetan Bhagat, novels`。SciFact test 首题 `qid=1`：`0-dimensional biomaterials show inductive properties.`。统计见 `download-audit.json`、`artifact-structure-audit.json`。官方 128k/1m 是数据构建标签，**不是本次目标 tokenizer 的实测 token 长度**。

**源码为何支持长共享 prefix。** `prompts_retrieval.py:141–176` 注册 corpus instruction、formatting instruction、corpus+few-shot，之后才当前 query；`shuffle_seed=None`。`prompt_registry.py:143–146` 按固定 corpus 顺序生成 pid mapping，`prompts/utils.py:99–145` 的 gold reposition 默认 None。因此默认渲染共享内容不依赖当前 test 答案。`cacheable_corpus=False` 表示 Python prompt 对象是否重新构建，**不等于 token 不可 APC**。[L2][L3]

**金标边界。** query schema `qid,query_text,metadata,answers`，其中 `metadata.qrels` 已含 gold passage IDs，必须同 answers 一起剔除；corpus 输入只取 `pid,title_text,passage_text`。官方五个 labeled few-shot demos 是允许的已给示范，不是当前 test gold；若严格只用无标签内容，应标为 **zero-shot serving 变体**，不能称完整官方 few-shot 复现。禁止按 qrels 为每题裁上下文、移动 gold docs，或跨长度合并重复材料 padding。[L2][L3]

**评价/环境/许可。** QUEST `mrecall@3`、SciFact `recall@1`，官方集合运算 evaluator 是 CPU 确定性代码，无 LLM judge。默认 inference 使用 VertexAI/Gemini，不应为下载 corpus 而执行 inference。软件 Apache-2.0，其他材料 CC-BY-4.0，且保留各原始任务许可约束。[L1][L4]

**争用反例。** 一个 corpus+100 queries 在一次预填充后 native APC 可能已非常好，单 corpus 未必有 eviction。混合官方 QUEST/SciFact 可产生不同工作集，但应称**多 corpus serving replay**，不是作者原单 workload 或生产 arrival；不把 32k/128k/1m 同题当独立请求样本。

### 4.3 QASPER：科学论文多问，CPU 官方评分完整

官方 loader/card 报告 1,585 papers / 5,049 questions，train 888/2,593、dev 281/1,005；本次 test 实读 **416/1,451**，与总数相符。test 每篇 1–10 问，32 篇只有一问。作者 HF 非 gated，数据 CC-BY-4.0、baseline code Apache-2.0。直接 S3 JSON 下载无需旧版 `datasets` 自定义脚本兼容性。[A1][A2]

`qasper-test-and-evaluator-v0.3.tgz` 为 3,865,061 B，包含 `qasper-test-v0.3.json`（18,078,957 B）、`qasper_evaluator.py`、`README-test.md`。train/dev 包 10,835,856 B，本次只 HEAD，未整包下载。**QASPER test 含答案**，不是隐藏标签 test。[A1；`qasper-test-audit.json`]

真实论文 ID `1911.10742`，*End-to-End Trainable Non-Collaborative Dialog System*：`397a1e851aab41c455c2b284f5e4947500d797f0` 问 ANTISCAM 数据集大小；`cc8b4ed3985f9bfbe1b5d7761b31d9bd6a965444` 问 intent 如何标注；`f7662b11e87c1e051e13799413f3db459ac3e19c` 问超越哪些 baselines。它们共享同论文，不需改写。

**输入**取 `title,abstract,full_text[].section_name/paragraphs`、必要的原图表 captions，以及 `qas[].question_id/question`。排除 answers 的 spans、yes/no/free-form answer、evidence/highlighted evidence；不按“可答”“evidence 段落”筛题。预测 `{question_id,predicted_answer,predicted_evidence}`，官方 evaluator 仅标准库，返回 Answer F1、Evidence F1、缺失预测数。忽略图表应标为 text-only，而不是隐去任务变化。[A1][A3]

**核心 caveat：** 官方 LED reader 拼接 `[start]+question+[sep]+context`，相同论文在不同 question 后不共享长 prefix。document-first 是显式 serving 模板适配，不能称原 LED pipeline 原生长共享。[A4:252–289]

### 4.4 MultiDoc2Dial：知识库共享与因果对话，不是并行未来 turns

实际官方 ZIP 包含 train/validation/test **3,474/661/661 dialogues**，48,002/9,195/8,938 turns，总计 **4,796 dialogues、66,135 turns**；488 文档按 SSA 109 / VA 138 / DMV 149 / StudentAid 92 分布。旧 `data_readme.html` 称 test 只有 dummy，但已下载 ZIP 有 661 条完整结构 test dialogues：以当前 artifact 为准，且不混同后续 leaderboard 隐藏 test。[C1][C2；本地结构审计]

额外对全部公开对话的标注引用去重计数：487 个文档被引用，484 个被至少两个不同对话引用，最多一篇对应 81 个对话。这证明跨对话文档复用确实存在；这里只把 gold references 用于描述性统计，**不据此为待测输入选文档**。详见 `multidoc2dial-reuse-counts.json`。

文档 schema：`doc_data[domain][doc_id] -> {title,doc_id,domain,doc_text,spans,doc_html_ts,doc_html_raw}`；dialogue：`dial_data[domain][] -> {dial_id,turns[]}`，turn 有 `role,turn_id,utterance,da,references`。validation `dial_id=1409501a35697e0ce68561e29577b90a` 首轮 `My insurance ended so what should i do`；test `8cb49f32f33505cd9650d60abe05e8ca` 首轮询问忘记更新地址怎么办。turn_id 是顺序，不是时间。[C1]

**泄漏/自然并行。** 原知识库可供工具读取，但当前 turn 的 `references[].doc_id/id_sp` 是 gold grounding，不应凭它挑 context；未来 agent utterance 是目标答案。teacher-forced 历史只能含当前轮之前内容，并与 free-running agent 区分。对话间可并行，单对话未来轮不能全并行；不同对话读到同文档只是正文复用机会，历史在前仍会破坏 prefix。

**环境/评价/许可。** 数据发布与 IBM 实现仓库 Apache-2.0，保留政府资料来源，不据此保证所有权利。IBM 提供 retrieval recall@n、response generation；`scripts/sharedtask_eval.py` 用 ID 对齐 JSON，支持 grounding EM/F1、utterance F1/SacreBLEU/METEOR/ROUGE-L。原 baseline 需旧 Conda/DPR/FAISS/RAG 环境；CPU audit 不需要。ZIP 展开约 96 MB，含重复 HTML/spans，可按具名成员读取，不全展开。[C2][C3]

### 4.5 Mooncake FAST'25：真正可下载的 prefix-hash + arrival

作者说明 conversation/toolagent 分别从不同线上集群采样一小时，优先保留同用户 session 以保存缓存关系。synthetic 来自 ShareGPT、LEval、LooGLE，Poisson arrival。因此前两者是**生产采样 trace**但不是无偏线上全量；后者不能称真实 arrival。[T1][T2:Appendix A]

| 官方文件 | 实际字节 / 请求数 | 实测平均 input/output tokens | 到达时间 |
|---|---|---|---|
| `conversation_trace.jsonl` | 3,029,533 / 12,031 | 12,035.06 / 342.62 | 0–3,536,999 ms，原序非递减 |
| `toolagent_trace.jsonl` | 4,415,857 / 23,608 | 8,596.24 / 182.13 | 0–3,536,999 ms，同 timestamp 最多 47 请求 |
| `synthetic_trace.jsonl` | 1,136,313 / 3,993 | 15,325.48 / 149.12 | 0–1,022,025 ms，**Poisson 生成** |

上述数字为完整小文件实际解析。toolagent 文件最后变更 commit `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb`，2025-02-21；本次下载固定仓库 revision `fa115fd6b76eebbc4e7faf94bff1773ad4640936`。根 `LICENSE-APACHE` 已验，单独 trace license 声明**未验证**；报告仓库 Apache-2.0，不扩张为原始用户内容权利。[T3]

无官方 task/request/session ID；用 pinned 文件+1-based 行号作派生 audit ID。toolagent 实际前两行：

```json
{"timestamp":0,"input_length":6758,"output_length":500,"hash_ids":[0,1,2,3,4,5,6,7,8,9,10,11,12,13]}
{"timestamp":0,"input_length":7322,"output_length":490,"hash_ids":[0,14,15,16,17,18,19,20,21,22,23,24,25,26,27]}
```

`timestamp` 单位 ms；`hash_ids` 是 512-token **包含前序块**的 prefix hash 重映射，**不是 token IDs**。这两请求仅共享第一完整块 512 tokens，非全部 6k。三个文件所有行都满足 `len(hash_ids)==ceil(input_length/512)`；最后块可能不满，APC 估算只取 `input_length//512` 个完整块。[T1][T2]

**本次结构上界，非服务结果。** 按原文件次序构建完整块 prefix trie，与先前路径求 LCP，再插入；假设此前请求瞬间计算完成、无限容量，同 timestamp 仍按文件顺序可见。分母为所有 input tokens，尾部不完整块不计复用：

| workload | 有先前完整块 prefix 的请求数 | 乐观可复用 input-token 比例上界 |
|---|---:|---:|
| conversation | 12,030 / 12,031 | 37.34% |
| toolagent | 23,604 / 23,608 | 57.03% |
| synthetic | 1,761 / 3,993 | 65.04% |

详细值/算法假设在 `mooncake-structural-counts.json`。请求共享率接近 100% 可能仅代表通用短前缀，不能替代 token 加权指标。这些数**不是 native APC hit rate、GPU 节省或 Octopus gain**；忽略计算可用时间/同刻 cold miss，因此不能推导 TTFT/吞吐。不同 trace 的 hash ID 重新映射，**不跨文件混用 hash namespace**。

**缺口。** 没有原文/token、工具 start/end/内容、分支 parent IDs、任务答案/evaluator、终止状态；无法验证有用 arms。可回放真实 request arrival 和块关系，在明确标注的服务时间/容量模型下模拟，不能还原真实 tool waiting。`output_length` 是历史执行结果，可作 decode-demand replay，但在线 admission 不应偷看未来真实 output length，除非所有系统同样 oracle 并另标。

## 5. BurstGPT：真实 timing 对照，不能替代前缀 trace

v2.0 有三个原始分片及对应 without-fails 副本，不是六个独立 workload。README 近似计数：前两片约 5.29M/121 天，第三片约 5.34M/110 天；release API 原始三片大小 **52,283,111 / 144,819,209 / 231,682,327 B**。本次各读取前 4,096 B、HTTP 206，**未全量下载或独立重数请求**。[R1][R2]

实际 `_1`/`_2` header 只有 `Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type`；**只有 `_3`** 的已核验 header 额外有 `Session ID,Elapsed time`，不要把最新 README 的 schema 推广到旧分片。第三片首个事件：timestamp `19440110.0`、session `1722ac82-0a46-4bf0-aa08-89794e7a2b3f`、elapsed 43 秒、input/output 906/446。[本地 `burstgpt-header.json`、`burstgpt-more-headers.json`]

Elapsed 是提交到完整响应，非 TTFT/工具时长；session 相同不保证请求包含相同历史。无文本/token/prefix hash。官方 `PromptMatcher` 从另一份 ShareGPT pool 按 input/output 长度近邻匹配，这是替代内容**合成 replay**，不是还原生产 prefix。根数据许可 CC-BY-4.0、example code 另有 license，没有业务答案 evaluator。[R1][R3]

## 6. 可直接执行的下载和 input-only schema

命令在独立下载目录执行，不将原始语料/gold 提交入 Octopus；不安装环境、不启动模型。GitHub URL 固定 revision，外部 blob 另记录 ETag/Last-Modified。所有路径来自已核验官方接口；例外只到 tree/HEAD 的状态单独注明。

### 6.1 首选：QuALITY test 原文件没有答案标签

```bash
curl -fL --retry 2 \
  'https://raw.githubusercontent.com/nyu-mll/quality/f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test' \
  -o quality-test.raw.jsonl

python3 - <<'PY'
import json
from pathlib import Path

documents = {}
tasks = []
with Path('quality-test.raw.jsonl').open() as source:
    for line in source:
        row = json.loads(line)
        context_id = row['article_id']
        document = {key: row[key] for key in ('article_id', 'title', 'article')}
        if context_id in documents:
            assert documents[context_id] == document
        documents[context_id] = document
        for q in row['questions']:
            tasks.append({
                'task_id': q['question_unique_id'],
                'context_id': context_id,
                'question': q['question'],
                'options': q['options'],
            })
assert len(documents) == 116 and len(tasks) == 2128
assert len({q['task_id'] for q in tasks}) == 2128
for name, rows in [('quality-contexts.input.jsonl', documents.values()),
                   ('quality-tasks.input.jsonl', tasks)]:
    with Path(name).open('w') as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + '\n')
PY
```

context 表 `{article_id,title,article}`；task 表 `{task_id,context_id,question,options}`。按 context_id join，渲染一篇完整文档后放当前问题/选项；不附所有问题、不复写正文。剔除 difficulty/annotation 字段，不依赖任何 gold。tokenizer 对**完整 chat template 一次整体编码**，不要分别 encode prefix/suffix 再拼，假定 BPE 边界不变。

### 6.2 LOFT-QUEST 严格 input-only 提取

```bash
curl -fL --retry 2 'https://storage.googleapis.com/loft-bench/retrieval/quest.zip' -o loft-quest.zip
python3 - <<'PY'
import json
import zipfile

with zipfile.ZipFile('loft-quest.zip') as archive:
    for member, output, keys in [
        ('quest/128k/corpus.jsonl', 'quest-corpus.input.jsonl',
         ('pid', 'title_text', 'passage_text')),
        ('quest/128k/test_queries.jsonl', 'quest-test.input.jsonl',
         ('qid', 'query_text')),
    ]:
        with archive.open(member) as source, open(output, 'w') as target:
            for line in source:
                row = json.loads(line)
                target.write(json.dumps({key: row[key] for key in keys},
                                        ensure_ascii=False) + '\n')
PY
```

已排除 query `metadata.qrels/answers`。只用这些输入推理是无 labeled demos 的 zero-shot 变体；完整官方 few-shot 复现另从 five-shot 文件/固定提示代码生成，不用 test gold。第二 corpus：`https://storage.googleapis.com/loft-bench/retrieval/scifact.zip`。官方 grader：`python run_evaluation.py --answer_file_path <queries.jsonl> --pred_file_path <predictions.jsonl> --task_type retrieval`，输出 ID 格式要对齐官方 pid mapping。[L1][L4]

### 6.3 Mooncake 真实 arrival/hash 输入

```bash
curl -fL --retry 2 \
  'https://raw.githubusercontent.com/kvcache-ai/Mooncake/fa115fd6b76eebbc4e7faf94bff1773ad4640936/FAST25-release/traces/toolagent_trace.jsonl' \
  -o mooncake-toolagent.raw.jsonl
```

raw schema `{timestamp:int(ms),input_length:int,output_length:int,hash_ids:list[int]}`。input-only whitelist `{timestamp,input_length,hash_ids}`；output_length 留在 replay-demand 侧，不给在线 policy。使用真实 arrival，但不把文件同毫秒行先后解释成前一请求已完成；512 粒度不可冒充引擎的更细 block size/LCP。

### 6.4 其他精确接口

| 候选 | 官方直接 URL / 入口 | 输入和 evaluator 边界 |
|---|---|---|
| QASPER | `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz` | `curl -fL URL -o qasper-test.tgz`；成员 `qasper-test-v0.3.json,qasper_evaluator.py`；评分 `python3 qasper_evaluator.py --gold qasper-test-v0.3.json --predictions predictions.jsonl`，gold 隔离 |
| MultiDoc2Dial | `https://raw.githubusercontent.com/doc2dial/multidoc2dial/6b7565989ad14858ee2c7498b605dc6d32ffe9e0/file/multidoc2dial.zip` | 具名成员 `multidoc2dial/multidoc2dial_doc.json` 和 `multidoc2dial_dial_validation.json`；去 refs/未来轮；IBM `scripts/sharedtask_eval.py --task utterance --prediction_json ... --reference_json ...` |
| DeepResearch Bench | `https://raw.githubusercontent.com/Ayanami0730/deep_research_bench/469cce54ea7f6a63c163d3d9fec879cf289ec484/data/prompt_data/query.jsonl` | 原文件 `{id,topic,language,prompt}` 就是 query input；不读 criteria/reference |
| BrowseComp | `https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv` | 按 [B1]:54–68 只解扰 problem；不是整个 CSV input-only；grader 的返回值问题需先确认 |
| MuDABench JSON | `https://raw.githubusercontent.com/Zhanli-Li/MuDABench/471f304b15b8e20809f20a959df091532782acdb/data/simple.json` | `curl -fL URL -o muda-simple.raw.json`；替换 complex.json 取得对应套；allowlist 文档身份，删 atomic facts |
| MuDABench PDF | `https://huggingface.co/datasets/Zhanli-Li/MuDABench/resolve/af2360876c0b8789e2ca1af9d648f9370eb52600/data/pdf/7560dc54-81cb-4d28-bbb1-205f060ad678.pdf` | ID 在 HF tree 已验证；此具体 PDF GET/OCR **未运行**，不声称文本质量已验 |
| MLE-bench description | `https://raw.githubusercontent.com/openai/mle-bench/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/mlebench/competitions/leaf-classification/description.md` | 只做比赛说明 input；真正数据要 Kaggle 条款/API/prepare |
| BurstGPT header | `https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv` | `curl -fL --range 0-4095 URL -o header.fragment`；206 已验，最后行可能不完整；非全量 CSV |

上表 `URL` 是对应单元格的完整 URL，不是另一个发现接口。可用 `curl -fL --retry 2 '<完整URL>' -o <具名文件>` 直接下载。压缩包尽量读取具名成员，不将 gold/整个目录自动导入 agent workspace。

## 7. 可证伪测量和 native APC 强基线

### 7.1 最小测量设计

先对 QuALITY 全 test 或预注册 input-ID 子集做**最终模板 token LCP**：固定 tokenizer revision/chat template/model 配置，保持正文顺序/标点；测不同 task 的 LCP、block-aligned LCP、token 加权共享量、每 context 独立问题数、不同文档工作集。通用短 system prefix 单独报告。file-order 与 document-grouped order 分列，后者只是 locality 上界，不是真实 arrival。

Mooncake 另做原 arrival/hash 的缓存结构 replay，不将其任务内容替换成 QuALITY 后仍叫真实 trace。二者配对必须标 **hybrid synthetic replay**。若上下文放不进目标模型，应报告不适配，不静默截断、按答案选证据或 padding 到目标长度。

| 待检验假设 | 直接否证量 | 失败应如何解释 |
|---|---|---|
| 真正共享大 Brain prefix | 最终 token LCP，扣开通用短指令 | 若只短 system prompt 重复，放弃“大上下文共享”主张 |
| P1 的零事件条件被改变 | active eligible useful arms 时间序列，同时至少两个独立可执行请求 | active≤1 时无法识别并发 scheduling/admission 效果 |
| 容量约束真实起作用 | 可用块 reuse distance、active KV、pending 队列、实际 eviction/rejection/preemption 事件 | 工作集一直放得下、事件为零，policy draws 仍无识别力 |
| Octopus 超越普通 serving | 同输入/资源/容量/quality/SLO 下不同有效任务完成量或延迟差 | 仅降低重复 prefill、仅多缓存容量不证明 Octopus 特有价值 |
| admission/adoption 有用 | 被接纳/拒绝 arms 的依赖、输出与最终质量关系 | 静态 QA 无 adoption 动作，不宣称验证此机制 |
| timing 来自生产 | 保留原 arrival、说明采样偏好和 cache 可用时点 | 无工具完成事件只能结构/仿真，不说真实 tool latency |
| 无标签泄漏 | 字段 allowlist；排除 labels/qrels/atomic values/future turns/reference reports | 一旦泄漏，该实验无效，不是加风险脚注后照用分数 |

容量/负载围绕实际输入工作集与目标硬件定义，区分“装得下”和真实部署容量条件；不要不断缩小 cache、提高 arrival rate 直到某策略获胜。短输出、单 corpus 热驻留、固定短指令也可能使 APC 已足够，这是有效负结果。

### 7.2 native APC 是主基线，不是可省略优化

vLLM 官方 APC 场景正包括**同一长文档的不同查询、多轮对话**；复用 prefix KV 减少 prefill，不减少 decode。块身份包含 parent hash/current token IDs/LoRA/多模态/cache salt，只有完整块可缓存；native block pool 已有引用计数和 LRU 驱逐。因此 Octopus+reuse 不能只对比 prefix caching 关闭、每请求复制全文 KV 的系统。[V1][V2]

主对照用**实际部署版本的 native vLLM APC（或相应引擎原生 prefix/radix cache）开启**，不人为禁用正常 continuous batching/scheduler。同 model/tokenizer、KV 容量、parallelism、routing、admission 限制和时间窗口；无 APC 只消融。cold/warm、并发 cold miss 分清，不能给 Octopus 预热却给 baseline 冷启动。模板排序/缓存容量/salt 改动给基线同等机会。Mooncake 512-token 抽象块仿真不能当 vLLM block pool 精确模拟。

**本次未测 native APC，也未测 Octopus 胜过它。** 57.03% 是 Mooncake 输入结构乐观上界；QuALITY 的 116 个正文组是数据事实，均不是 GPU prefill 节省或最终用户收益。

## 8. 固定版本、证据与未验证项

| 来源 | 固定 revision / commit 日期 | 数据身份 |
|---|---|---|
| openai/mle-bench | `507f92e1138bb6e40dac5c6ee7a6758e6424bf97` / 2026-04-24 | v1 仓库接口；未取 Kaggle |
| openai/simple-evals | `652c89d0ca9df547706735883097e9537d40dc47` / 2026-04-22 | CSV Last-Modified 2025-04-10 |
| Ayanami0730/deep_research_bench | `469cce54ea7f6a63c163d3d9fec879cf289ec484` / 2026-05-11 | 原版 100 queries，非 DRB II |
| Zhanli-Li/MuDABench | `471f304b15b8e20809f20a959df091532782acdb` / 2026-07-14 | HF `af2360876c0b8789e2ca1af9d648f9370eb52600` |
| nyu-mll/quality | `f84977c40dbfef70c9cab48037b7becfc8e45f73` / 2025-01-17 | v1.0.1，发布说明 2022-06 |
| allenai/qasper HF | `fdc9d8214fbab5dd782958601db4d678e6934a54` | S3 v0.3，Last-Modified 2022-03-25 |
| allenai/qasper-led-baseline | `afd0fb96bf78ce8cd8157639c6f6a6995e4f9089` / 2021-10-27 | test 包另附 evaluator |
| google-deepmind/loft | `219f68e470260c22fc53cc8cc2416b78313db774` / 2025-06-13 | 两 ZIP Last-Modified 2024-08-28 |
| doc2dial/multidoc2dial | `6b7565989ad14858ee2c7498b605dc6d32ffe9e0` / 2022-05-01 | 实读 ZIP train/dev/test |
| IBM/multidoc2dial | `2456a5609b447e1a12b716f4d658728f346f3041` / 2022-03-09 | 原模型/evaluator |
| kvcache-ai/Mooncake | `fa115fd6b76eebbc4e7faf94bff1773ad4640936` / 2026-09-11 | toolagent 最近变更 `3cca71da...` / 2025-02-21 |
| HPMLL/BurstGPT | `d895a53bb7b8ec137d0d2fe203b335835a78c10a` / 2026-06-30 | v2.0 release 2026-01-15；只读 header |
| vllm-project/vllm APC docs | `263c4ff95fadb62d80f315171b17d4f623494e56` / 2026-09-08 | 文档身份，不是当前安装版本 |

这些是代码/数据日期，不假装都是论文发表日期。S3/GCS/blob 非 content-addressed，未来下载应重新记 ETag/Last-Modified/bytes，不能因文件名有版本就认为不可变。

证据位于 **`results/benchmark-discovery/research-data/`**，约 252 KB，不含整篇论文/财报/大 dataset 或 QA gold answer 值。

| 小证据文件 | 内容 |
|---|---|
| `repository-revisions.json,extra-revisions.json` | 官方 revisions/dates/路径大小；也保留初次错误候选路径的 404，不把失败当成功 |
| `download-audit.json,http-artifact-metadata.json,additional-access-metadata.json` | GET/HEAD URL、bytes、archive members、ETag/修改日期；大包只 HEAD |
| `artifact-structure-audit.json,quality-train-counts.json,qasper-test-audit.json,multidoc2dial-reuse-counts.json` | schema、实际 split 数、input-only 例子、同文档关系/MuDA 重合、跨对话文档复用 |
| `Zhanli-Li__MuDABench.json,allenai__qasper.json,muda-pdf-storage.json` | HF revision/gated/license card、PDF paths/总尺寸 |
| `mooncake-trace-audit.json,mooncake-structural-counts.json` | 实际 anonymous schema/行例、完整计数、上界假设 |
| `burstgpt-header.json,burstgpt-more-headers.json,BurstGPT-revision.json,BurstGPT-tree.json` | header 差异与 Range 读取证据 |
| `research-summary.json` | 首选 input allowlist、同文档 IDs、Mooncake 来源/能力边界 |

**未验证：** 模型/benchmark 分数，Octopus/APC 实际吞吐/延迟，目标 tokenizer token LCP，PDF 转文本质量，Kaggle 权限，线上搜索/ChatDOC/Jina 运行可用性，全部外部材料的完整权利，Mooncake 工具 DAG/任务答案，以及任何 quality/adoption gain。

## 9. 一手来源索引

引用为拥有者仓库、作者数据或作者论文，代码固定 revision。JSON/压缩数据无适用源码行时给具体成员、item selector/task ID，实际计数另有本地 evidence；不以搜索摘要代替原文。

- [M1] [openai/mle-bench:README.md:71–193](https://github.com/openai/mle-bench/blob/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/README.md#L71-L193)；[已知问题 :260–311](https://github.com/openai/mle-bench/blob/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/README.md#L260-L311)。
- [M2] [openai/mle-bench:mlebench/competitions/leaf-classification/prepare.py:1–59](https://github.com/openai/mle-bench/blob/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/mlebench/competitions/leaf-classification/prepare.py#L1-L59)；[grade.py:1–46](https://github.com/openai/mle-bench/blob/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/mlebench/competitions/leaf-classification/grade.py#L1-L46)。
- [M3] [openai/mle-bench:LICENSE:1–23](https://github.com/openai/mle-bench/blob/507f92e1138bb6e40dac5c6ee7a6758e6424bf97/LICENSE#L1-L23)。
- [D1] [Ayanami0730/deep_research_bench:README.md:52–115](https://github.com/Ayanami0730/deep_research_bench/blob/469cce54ea7f6a63c163d3d9fec879cf289ec484/README.md#L52-L115)；[使用/依赖 :176–302](https://github.com/Ayanami0730/deep_research_bench/blob/469cce54ea7f6a63c163d3d9fec879cf289ec484/README.md#L176-L302)。
- [D2] [Ayanami0730/deep_research_bench:data/prompt_data/query.jsonl:1–2](https://github.com/Ayanami0730/deep_research_bench/blob/469cce54ea7f6a63c163d3d9fec879cf289ec484/data/prompt_data/query.jsonl#L1-L2)。
- [D3] [Ayanami0730/deep_research_bench:deepresearch_bench_race.py:29–30,58–106](https://github.com/Ayanami0730/deep_research_bench/blob/469cce54ea7f6a63c163d3d9fec879cf289ec484/deepresearch_bench_race.py#L29-L106)。
- [B1] [openai/simple-evals:browsecomp_eval.py:16–48,54–123](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/browsecomp_eval.py#L16-L123)；[官方 CSV](https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv)。
- [U1] [Zhanli-Li/MuDABench:README.md:20–39,73–125](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/README.md#L20-L125)。
- [U2] [作者 simple.json，item 0 与 question_id 如正文](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/data/simple.json)；[complex.json 对应 item 0](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/data/complex.json)。
- [U3] [作者 HF API](https://huggingface.co/api/datasets/Zhanli-Li/MuDABench)；[固定 PDF tree API](https://huggingface.co/api/datasets/Zhanli-Li/MuDABench/tree/af2360876c0b8789e2ca1af9d648f9370eb52600/data/pdf?limit=1000)。
- [U4] [Zhanli-Li/MuDABench:agent/agent_runner.py:265–347](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/agent/agent_runner.py#L265-L347)；[CLI :469–504](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/agent/agent_runner.py#L469-L504)。
- [U5] [Zhanli-Li/MuDABench:eval/evaluate.py:145–185](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/eval/evaluate.py#L145-L185)；[最终评分 :1033–1064](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/eval/evaluate.py#L1033-L1064)；[common/question_id.py:1–52](https://github.com/Zhanli-Li/MuDABench/blob/471f304b15b8e20809f20a959df091532782acdb/common/question_id.py#L1-L52)。
- [Q1] [nyu-mll/quality:README.md:14–87](https://github.com/nyu-mll/quality/blob/f84977c40dbfef70c9cab48037b7becfc8e45f73/README.md#L14-L87)。
- [Q2] [nyu-mll/quality:data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test:1,13](https://github.com/nyu-mll/quality/blob/f84977c40dbfef70c9cab48037b7becfc8e45f73/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test#L1-L13)。
- [Q3] [nyu-mll/quality:baselines/lrqa_deprecated/tasks.py:64–81,107–125](https://github.com/nyu-mll/quality/blob/f84977c40dbfef70c9cab48037b7becfc8e45f73/baselines/lrqa_deprecated/tasks.py#L64-L125)。
- [A1] [allenai/qasper:qasper.py:35–46,65–105,118–152](https://huggingface.co/datasets/allenai/qasper/blob/fdc9d8214fbab5dd782958601db4d678e6934a54/qasper.py#L35-L152)。
- [A2] [allenai/qasper:README.md:155–160,219–221](https://huggingface.co/datasets/allenai/qasper/blob/fdc9d8214fbab5dd782958601db4d678e6934a54/README.md#L155-L221)。
- [A3] [allenai/qasper-led-baseline:scripts/evaluator.py:12–95,99–173](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py#L12-L173)。
- [A4] [allenai/qasper-led-baseline:qasper_baselines/dataset_reader.py:252–289](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/qasper_baselines/dataset_reader.py#L252-L289)。
- [L1] [google-deepmind/loft:README.md:1–47,117–175](https://github.com/google-deepmind/loft/blob/219f68e470260c22fc53cc8cc2416b78313db774/README.md#L1-L175)；[许可 :212–237](https://github.com/google-deepmind/loft/blob/219f68e470260c22fc53cc8cc2416b78313db774/README.md#L212-L237)；[QUEST ZIP](https://storage.googleapis.com/loft-bench/retrieval/quest.zip)、[SciFact ZIP](https://storage.googleapis.com/loft-bench/retrieval/scifact.zip)。
- [L2] [google-deepmind/loft:prompts/prompts_retrieval.py:125–180](https://github.com/google-deepmind/loft/blob/219f68e470260c22fc53cc8cc2416b78313db774/prompts/prompts_retrieval.py#L125-L180)。
- [L3] [google-deepmind/loft:prompts/utils.py:99–145,196–245](https://github.com/google-deepmind/loft/blob/219f68e470260c22fc53cc8cc2416b78313db774/prompts/utils.py#L99-L245)；[prompts/prompt_registry.py:143–224](https://github.com/google-deepmind/loft/blob/219f68e470260c22fc53cc8cc2416b78313db774/prompts/prompt_registry.py#L143-L224)。
- [L4] [google-deepmind/loft:evaluation/retrieval.py:26–73,115–140](https://github.com/google-deepmind/loft/blob/219f68e470260c22fc53cc8cc2416b78313db774/evaluation/retrieval.py#L26-L140)。
- [C1] [doc2dial/multidoc2dial:file/multidoc2dial.zip，成员与 selector 见正文](https://github.com/doc2dial/multidoc2dial/blob/6b7565989ad14858ee2c7498b605dc6d32ffe9e0/file/multidoc2dial.zip)；[旧 data_readme.html](https://github.com/doc2dial/multidoc2dial/blob/6b7565989ad14858ee2c7498b605dc6d32ffe9e0/data_readme.html)。
- [C2] [IBM/multidoc2dial:README.md:26–42,90–123](https://github.com/IBM/multidoc2dial/blob/2456a5609b447e1a12b716f4d658728f346f3041/README.md#L26-L123)。
- [C3] [IBM/multidoc2dial:scripts/sharedtask_eval.py:19–99](https://github.com/IBM/multidoc2dial/blob/2456a5609b447e1a12b716f4d658728f346f3041/scripts/sharedtask_eval.py#L19-L99)。
- [T1] [kvcache-ai/Mooncake:FAST25-release/README.md:3–55](https://github.com/kvcache-ai/Mooncake/blob/fa115fd6b76eebbc4e7faf94bff1773ad4640936/FAST25-release/README.md#L3-L55)；[toolagent_trace.jsonl:1–2](https://github.com/kvcache-ai/Mooncake/blob/fa115fd6b76eebbc4e7faf94bff1773ad4640936/FAST25-release/traces/toolagent_trace.jsonl#L1-L2)。
- [T2] [作者 Mooncake FAST'25 paper，Appendix A，p.14](https://github.com/kvcache-ai/Mooncake/blob/fa115fd6b76eebbc4e7faf94bff1773ad4640936/FAST25-release/Mooncake-FAST25.pdf)：本次已读该页，核验 different clusters/session-preserving sampling、synthetic 来源、累计块 hash。
- [T3] [kvcache-ai/Mooncake:LICENSE-APACHE:1–14](https://github.com/kvcache-ai/Mooncake/blob/fa115fd6b76eebbc4e7faf94bff1773ad4640936/LICENSE-APACHE#L1-L14)；[trace 发布 commit](https://github.com/kvcache-ai/Mooncake/commit/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb)。
- [R1] [HPMLL/BurstGPT:README.md:3–55](https://github.com/HPMLL/BurstGPT/blob/d895a53bb7b8ec137d0d2fe203b335835a78c10a/README.md#L3-L55)；[LICENSE:1–61](https://github.com/HPMLL/BurstGPT/blob/d895a53bb7b8ec137d0d2fe203b335835a78c10a/LICENSE#L1-L61)。
- [R2] [HPMLL/BurstGPT v2.0 release](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)；[官方 API](https://api.github.com/repos/HPMLL/BurstGPT/releases/tags/v2.0)。
- [R3] [HPMLL/BurstGPT:example/src/burstgpt/prompts.py:1–13,59–85](https://github.com/HPMLL/BurstGPT/blob/d895a53bb7b8ec137d0d2fe203b335835a78c10a/example/src/burstgpt/prompts.py#L1-L85)。
- [V1] [vllm-project/vllm:docs/features/automatic_prefix_caching.md:3–13,42–51](https://github.com/vllm-project/vllm/blob/263c4ff95fadb62d80f315171b17d4f623494e56/docs/features/automatic_prefix_caching.md#L3-L51)。
- [V2] [vllm-project/vllm:docs/design/prefix_caching.md:3–27,151–177,227–239](https://github.com/vllm-project/vllm/blob/263c4ff95fadb62d80f315171b17d4f623494e56/docs/design/prefix_caching.md#L3-L239)。
