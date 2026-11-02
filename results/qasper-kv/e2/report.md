# E2：高显存 QASPER 原生机制诊断

**16/16 native traces 已完成并验证，但当前实现未通过性能门。KV 机制真实有效，端到端成本反而增加。**
本轮依据用户2026-09-11 21:25的显式指令，在已知E1严格QA失败后继续诊断；没有把原质量门改成通过。

## 固定控制与完整结果

四个arrival seeds（1601/1702/1803/1904）×四种策略，共16条trace、6,752次原始问题请求。
每条422请求，模型、完整输入token IDs、同seed到达时序及实际KV容量33,198,047,232bytes一致。
GPU0承接1601/1904，GPU3承接1702，GPU6承接1803；同seed四方法在同一物理卡上轮换顺序比较。
三个GPU启动后均观察到48,016/49,140MiB显存和100%利用率；这是快照，不是全程busy积分。

FG口径为每篇论文首问到最后一问完成，四篇取均值，再比较四个paired seeds。
它包含相对起点预设在221/237秒的复访等待，不是单次模型生成延迟。BG P95为四seed全部原始BG请求池化分位数。

| 方法 | FG E2E均值(s) | FG相对APC | BG P95(s) | BG相对APC | 每trace目标prefix重算tokens | Cap bind |
|---|---:|---:|---:|---:|---:|---:|
| APC | 247.335 | +0.00% | 49.206 | +0.00% | 33,888 | 0 |
| Queue-aware | 253.658 | +2.56% | 54.251 | +10.25% | 33,888 | 0 |
| B6 semantic fixed | 256.457 | +3.69% | 58.228 | +18.33% | 0 | 0 |
| B6 semantic budgeted | 256.923 | +3.88% | 58.453 | +18.79% | 0 | 0 |

## 预先声明的判据

B6 fixed 为主要候选，budgeted并列报告，不事后挑赢家。
B6 fixed 相对APC的FG平均差为+9.122秒（+3.69%），预设单侧paired-t上界+10.547秒；
相对queue-aware为+2.800秒（+1.10%），上界+3.399秒。四seed均未出现加速。
BG P95相对APC为+18.33%，seed内paired circular block bootstrap的单侧95%上界+20.37%，
均超过+5%门槛。bootstrap为block8、1000次、RNG17，不额外把TTFT或queue-BG设为硬门。

Budgeted同样未通过：FG相对APC+3.88%，BG P95+18.79%、单侧上界+21.01%。
它与fixed都把每trace的33,888个目标prefix重算token降到0，parent adoption从34,016增至67,904；
但所有budgeted trace的cap bind均为0，因此不能声称budget/admission的独立收益。

## 开销、质量与解释边界

Adapter CPU计时均值为APC1.120秒、queue-aware10.667秒、fixed18.529秒、budgeted18.885秒。
fixed的selector约14.464秒、preference provider约2.300秒，属于嵌套计时，不能与adapter相加。
这支持“控制/选择开销抵消KV节省”的解释，但尚未通过消融将全部延迟差因果归于它。
全局prefill仅由2,052,394降到2,018,506，节省约1.65%，不是总体计算减半。

每trace严格满F1为129–130/422，平均Answer F1约46.4%；原100%质量门仍未通过。
同seed与APC相比，输出text/token parity为420–422/422，少数BG回答存在措辞/内容差异；
不能宣称完全相同输出或未经验证就把差异归因于浮点误差。FG输出保持相同。
所有官方评分使用未修改QASPER evaluator、完整422问题子集，gold只在本机隔离目录。

本轮GPU进程墙钟合计 **4656.746秒**，加既有E1/失败启动584秒为 **5240.746秒**，低于10,800秒预算。
执行事件首尾墙钟约 **2324.440秒**，与跨卡求和的GPU-process seconds不同。


## 持久化证据

- [`plan.json`](plan.json)、[`authorization.json`](authorization.json)、[历史执行 helper](../reproduce/run_e2_diagnostic.py)：显式诊断授权、资源/矩阵与进程边界；不是新运行许可。
- [`execution.jsonl`](execution.jsonl)、[`execution-summary.json`](execution-summary.json)：全部16次启动/结束、实际GPU UUID、进程成本，无失败/超时。
- [`e2-export.json.gz`](e2-export.json.gz)：逐trace验证后的请求时序、native计数、生成输出、环境、原始raw路径/大小/SHA256，不含输入tokens或gold。
- [`analysis/report.json`](analysis/report.json)、[`aggregate-summary.json`](aggregate-summary.json)：逐seed配对、原始差值、CI、质量与parity完整结果。
- 外部 `analysis/*.predictions.jsonl` 的模型输出可由 export 在 CPU 上重新导出；外部 `analysis/evaluation_only/` 为隔离gold，不交给模型或提交Git。
- 原始10.32GB raw在`.68`的`/mnt/disk2/user/qasper-kv-e2-diagnostic-20260911/trace-data/`。
  完整归档已保存在外部 `e2-diagnostic/archive/e2-full-raw-artifacts-20260911.tar.zst`，284,672,370bytes；
  压缩SHA与远端一致，16/16原始raw SHA256均与compact export一致，详见
  [archive manifest](archive/e2-full-raw-artifacts-20260911.manifest.json) 和
  [owner validation](archive/e2-full-raw-artifacts-20260911.validation.json)。

实际native源码始终为`616ae10e3f00e250e2744ef0ebd6ce02307baac6`，未在运行中替换。
没有重跑E1、改变问题/模型/到达负载、添加新seed或通过额外GPU寻找获胜点。
本轮不支持端到端Pareto优势、budget收益或Full Octopus claim，也未启动E3/A1/A2。
所有GPU已释放，归档进程也已终止。[精确源码及CPU复现说明](../../../docs/qasper-kv-evidence.md)
区分未修改的测量源码、诊断分析 helper、归档路径和正式质量重评分。
