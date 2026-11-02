# QASPER E1 原生 GPU 有界结果

**E1 压力门通过；严格 QA 质量门失败，E2 及后续阶段未启动。**

以上为原 E1 终态，不是后续诊断的当前状态；下文保留其 no-go 历史。
原始机器记录中的路径与 `e2_executed: false` 也保持当时语义。

在 .68 的 GPU0（RTX4090D）运行 Qwen2.5-7B-Instruct 固定 snapshot，
context-first，16k 窗口、256 输出上限、12 个 FG 与410个原生BG问题。
到达时序是预先控制的复访负载，不是生产/工具时序。

| 容量 | 实际 KV | 完成请求 | 目标 block 逐出事件 | 完整 prefix 重算 tokens | 官方 F1=1 | 平均 Answer F1 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 30.918 GiB | 422/422 | 2,118 | 33,888 | 130/422 | 46.49% |
| stress | 26.917 GiB | 422/422 | 2,118 | 33,888 | 130/422 | 46.49% |

四个后续请求的 native adoption 都降到16 tokens，并分别重算13,040、8,928、
4,976、6,944 tokens，合计33,888；这些是已验证的完整共享块，不是冷启动或
4/9-token 尾巴。两点均没有目标 allocation rejection/preemption，但真实
cache eviction 与重算已满足压力门。正常点按预先规则选中，未按质量选点。

官方未修改 evaluator 对每点全部422个预定问题评分，无 missing prediction。
每点严格满分130/422，其中FG仅1/12，BG129/410；未达到逐题F1=1的门槛。
两容量的422个文本和token输出完全相同，但输出一致不替代任务质量。
因此没有追加E2矩阵，也没有用其他空卡运行不满足门槛的实验。

正常运行中的一次NVML快照为48,150/49,140MiB、100%GPU利用率，约98%显存。
这不是整段GPU-busy积分。有效两条trace的外层墙钟为547秒，另有19秒与18秒
的启动失败，均在任何QA输出前修复并保留。584秒不是包含所有准备、调试、
网络与评分的完整项目墙钟；评分用时单独记录。

实测源码为 `616ae10e3f00e250e2744ef0ebd6ce02307baac6`，对应
[`qasper-native-616ae10.bundle`](../source/qasper-native-616ae10.bundle)（需要仓库已含基础提交`d2a8c65`）。
外部 `e0-native-616ae10.json` 保存完整tokens；
[`scoring/`](scoring/) 发布原始预测导出与官方评分，不含 evaluator-only gold。
详细数据见 [`terminal-summary.json`](terminal-summary.json)；
完整输入与原始数据的位置见[外部 manifest](../large-artifacts.json)。

完整raw与官方评分已保存在外部永久目录 `attempt-003-tokenfix/`，共344,618,357 bytes，
由supervisor核对remote/local payload SHA256及永久副本一致性。传输耗时177秒，
接收线量约36.88 MB。本地完整归档已完成，后续SSH连接异常不再阻塞本轮数据交付。
没有声称Octopus加速、budget收益、完整Agent能力或不同workload泛化。

## 后续显式授权的 E2 诊断

用户随后在21:25明确要求启动E2。16/16诊断trace已完成，详见[独立 E2 报告](../e2/report.md)。
KV重算消除，但FG和BG尾延迟均更差；这不撤销本页原始E1质量门失败的历史结论。
