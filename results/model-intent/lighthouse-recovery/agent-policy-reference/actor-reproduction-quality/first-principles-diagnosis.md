# 为什么新增复现流程仍未修复 Lighthouse

**结论：模型先把问题理解错了；新增工具与结束门禁没有让它进入“用执行证据修正判断”的闭环；后续普通消息循环又逐渐挤掉了真实证据。** 这不是已经证实的 KV 故障，也不能直接推出“再训练一个 LoRA 就能解决”。

本轮只分析已完成的同一次实验，并做 CPU/Docker 诊断回放。没有新增实验模型推理、Teacher 调用、训练或官方评测，也没有修改生产控制逻辑。数值、真实输入解码、状态轨迹和公开源码反例见 [机器可读证据](first-principles-diagnosis.json)。

## 1. 放回整个仓库：失败在哪一层

检查覆盖根文档与依赖、81 个顶层 benchmark 模块的入口/导入关系、核心 runtime 及相关测试、现有训练入口、Lighthouse 工具/轨迹/评分链。这是跨模块调用关系核对，不是声称重跑全部历史实验或逐行审计每个文件。

| 层 | 代码与实际职责 | 对决策的约束 |
| --- | --- | --- |
| 当前 action policy | `benchmarks/lighthouse_actor_reference.py:588-650,652-765`：固定 SWE-Gym actor，经 stock `LLMEngine` 生成调用 | 当前路径关闭 prefix caching，不运行 Octopus checkpoint/fork；不能归因于 Octopus 的 KV 选择 |
| 工具与结束门禁 | 同文件 `ActorTools.run/run_reproduction`；`lighthouse_readonly_shell.py`；`openhands_tool_protocol.py` | 控制执行、原版快照、断言身份与能否结束，不替模型发现正确原因/复现 |
| 错误及语义状态 | `src/octopus/tool_memory.py`、`task_memory.py` | 资源版本、错误证书或语义事实不是“正确下一步 action”的证明；expanded 路径未启用旧 semantic capsule |
| Octopus runtime | `src/octopus/runtime.py:91-386`、`state.py:116-174`、`branch_policy.py:44-160` | 提交任务、精确 prefix/capsule 继承、等待分支结束、归并语义 Commit；外部验证与模型自报成功分离 |
| intent 到物理 KV | `model_intent.py`、`future_prefix.py:126-197`、`queue_retention.py:180-268`、`vllm_backend.py` | intent 影响空闲 native blocks 的回收次序，不选择源码位置/工具动作，不产生正确补丁，也不保证延迟收益 |
| 另一条 SWE 接入 | `lighthouse_quality_solver.py:583-633,944-1030`、`p1_solver.py` | 旧路径确实使用 `Octopus/CaseSession` 和单步 continuation；不能与当前 stock reference 混称，也不等于真正多分支探索 |
| 官方评分 | `lighthouse_quality.py:419-491` | 独立容器应用真实 model patch，校验冻结 evaluator 身份；结果不是 actor 输入或 runtime 自报成功 |
| 已有训练 | `tiny_lora.py:34-99` 训练 `checkpoint_id` 选择；`tool_recovery_lora.py:368-450` 训练 synthetic 失败→检查→后续动作 | 两者的目标、格式和预算不同于真实长上下文 SWE action policy，不能换点数据就当作现成训练方案 |

`test_model_intent_native.py`、`test_gpu_fork.py`、`test_branch_policy.py` 分别针对 native intent、fork 生命周期和外部验证边界，不证明 SWE 修复；本轮没有重跑这些测试。`controller_acceptance.evaluate_formal` 还绑定另一套 424-question 协议，不能直接挪作 Lighthouse 的 Formal 验收。后续若调整 action policy，必须先建立真实任务质量基线，再在相同 actor/提示/工具条件下比较实际 fork、native work、前台与后台成本。

## 2. 补丁为什么没有解决问题，还引入了回归

实际顺序：动作 0–3 查看四处；**动作 4 就改了 WordPress ID**；动作 7 的 finish 被拒；动作 10–13 又看了四处。先前报告把“总共八次 view”写成“先看八处再修改”，本轮已纠正。

动作 4 的真实输入包含 `id: 'react'` 和 `id: 'wordpress'`，共有 14,519 tokens，没有裁剪历史。模型却推断需要重命名才能消除 ID 重复。动作 14 甚至声称两个 pack 原来都有 `'wordpress'`，与已读源码矛盾。

它混淆了：**定义表里的 ID 唯一，不代表运行时返回列表不会重复。** 原版 `lighthouse-core/lib/stack-packs.js:44-67` 逐个 detection 查找并追加 pack；两个 React detection 都能选中同一个 React pack。改 WordPress 定义的名字，不改变这条数据流。`audits/dobetterweb/js-libraries.js:44-53` 又是直接把 JavaScript detections 映射为显示项，与该重命名无过滤关系。

在原版与**逐字匹配归档 model.patch**的版本上执行相同的公开源码诊断：

| 产品行为 | 原版 | 模型补丁后 |
| --- | --- | --- |
| 两个 React detection 返回的 pack IDs | `react, react` | `react, react` |
| 显示的库名称 | `React, React (Fast Path)` | `React, React (Fast Path)` |
| 一个 WordPress detection 返回的 pack IDs | `wordpress` | 空列表 |

补丁不但没解决原来的两项行为，还让 WordPress advice 丢失：映射表仍找 `packId='wordpress'`，定义却被改名了。这是公开行为反例，不是用隐藏测试倒推答案；诊断代码从未给 actor 使用，也不算 actor 自己完成复现。

## 3. 为什么新流程没有把它拉回来

### 结束门禁不等于执行顺序或推进策略

`ActorTools.run` 在 `done` 分支检查 red/green；普通 edit/write/shell 仍走原执行路径。真实 Docker 回放确认：

`没有 original_red → 接受动作 4 的修改 → 拒绝 finish → 此时新建 reproduce 要求先恢复原版`

去掉前四次 view、只保留实际 edit，约 10 秒仍能复现，零模型调用。恢复这一个修改后，诊断性断言能真实执行并失败，说明工具可用、恢复路径存在，**不是不可恢复的死锁**。

这项红回放检查的是更强的“未 red 不得先改产品”约束，当前实现没有提供它；不是发现绕过最终验收的漏洞，也不证明增加该约束就能让模型理解问题。此前 246 项检查证明工具/传输/断言门禁可用，其中成功动作顺序来自 scripted CPU fixture；它们没有验证 7B 会自主选出这个顺序。

### 普通消息绕开了推进反馈，而非官方验收

`solve` 识别普通消息后直接回固定 `CONTINUE_MESSAGE`，不调用 `ActorTools.run`，不附当前 `reproduction_state`。反馈还包含“如果认为解决了，先告诉用户，再结束”，而不是持续指出当前仍无原版失败证据。

动作 **15–99 连续 85 次**输出同一段“已经修改，可以在你的环境里测试”的文字；相同文本在动作 9 出现过一次，所以全程最高重复计数为 86。连续后缀只有一种 action、一种 observation、一种失败状态，**零实际工具提交、零产品变化**，却产生 19,805 output tokens，占本轮约 86.25%。

全部 100 次真实 prompt token 序列都不同，不能称为相同缓存输入的重放。问题是环境没有增加新证据，模型却不断把未经验证的结论写回历史。

### 历史裁剪是后期放大器，不是最初错误的原因

新工具定义、先复现再修改的指令、原始 issue 存在于 **100/100** 个实际输入。动作 7 的拒绝进入了下一次输入，89 个可检查的 message→下一次输入均包含准确 continuation。不是提示没接上或反馈丢包。

从输入 67 开始，那次 finish 拒绝的历史对被裁掉；最后一个输入只保留初始指令和动作 14–98 的消息对，没有真实 source/tool observation，也没有动态 missing-red 状态。错误首次出现、连续重复首次开始时都未裁剪，因此不能把初始误判归咎于 context 太小；后期裁剪让错误叙述更难被事实纠正。

## 4. 实验对照支持什么，不支持什么

| 单次实验观察 | 前一 expanded reference | 新 reproduction workflow |
| --- | ---: | ---: |
| 模型请求 | 100 | 100 |
| 总输出 tokens | 2,127 | 22,962 |
| 单次最大输出 tokens | 372 | 443 |
| 实际工具调用 | 8 | 9 |
| 官方 resolved | false | false |

新流程输出量约为前一轮的 10.80 倍，主要增加叙述而非执行证据；两轮都没触及 4096 输出上限。这只是两个具体 episode 的描述性比较，不是 IID 统计或某个单独组件的因果效果估计。

可以确定：本轮没有出现真实的 `证据 → 假设修正 → 相关复现 → 有效修改`。不能确定：换一种训练/提示是否一定成功、强模型失败是否一定说明接口有错、7B 是否在所有策略下都不可能完成。

## 5. 外部建议只取两点

1. **保留独立的 Teacher Ceiling 诊断。** 若获得明确运行授权，固定相同 task/tools/evaluator，把 teacher 作为独立诊断分支，不偷偷替换 7B 基线。`freeze_reference` 明确锁定 actor 身份，这个保护应保留。一次成功仅说明接口存在可行策略；一次失败仍需归因，不能自动判接口有问题。
2. **保留“真实上下文、工具证据与 action 接口对齐”的训练单位。** 但当前已是 `SWE-Gym/OpenHands-7B-Agent`，使用发布的真实 agent-trajectory 全参数 SFT，并非只训过 synthetic recovery。应先识别缺的是任务/语言泛化、长上下文证据使用还是新增工具接口对齐，不能把“再加真实轨迹”本身当答案。

暂不引入整套 post-training pipeline、RL、PRM、ranking loss 或固定数量的数据收集。Lighthouse 可排除在我们新增训练数据之外，但经过反复诊断，不能再称为未接触过的最终泛化验证集。

**最窄的工程决策边界在 action/controller 层的顺序、恢复与状态反馈，不在 KV allocator，也不是 checkpoint-choice LoRA。** 修正这些控制缺口不等于保证正确语义；Teacher/进一步训练仍需独立证据与授权。Octopus 的主问题依然是：真实任务质量成立后，让正常行动中暴露的 intent 带来物理推理工作与延迟收益。当前仍未越过前一个质量门。
