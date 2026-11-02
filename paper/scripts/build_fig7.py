#!/usr/bin/env python3
"""
Paper-Native High-Typography Figure 7 (1:1 True-Print 6.50in Scale, Inter Typography):
  (a) H200 KV Pool Utilization, Queue Depth & Timeout Rate Under Branch Bursts
  (b) 4x H200 DAG-GRPO Valid Trained Tokens & Train-Rollout KL Alignment
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

ROOT = Path(__file__).resolve().parent.parent
for ttf in (ROOT / "scripts/fonts").glob("*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.rcParams.update({
    "font.family": "Inter",
    "axes.edgecolor": "#3F3F46",
    "axes.linewidth": 0.85,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#3F3F46",
    "ytick.color": "#3F3F46",
    "grid.color": "#E4E4E7",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(6.50, 2.02), dpi=300)
gs = gridspec.GridSpec(
    1, 2, width_ratios=[1.0, 1.0], wspace=0.28,
    left=0.072, right=0.925, top=0.865, bottom=0.22
)

# ==============================================================================
# PANEL (a): H200 SGLang Rollout KV Utilization, Queue Depth & Timeout Rate
# ==============================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(-0.52, 2.58)
ax1.set_ylim(0, 128.0)
ax1.grid(True, axis="y", zorder=0)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

regimes = [
    "Unbounded\nEval (55/GPU)",
    "Unbounded\nRL (No Gate)",
    "Octopus Gate\n(τ = 75% KV)",
]
x_r = np.arange(len(regimes))
w = 0.255

kv_p90     = np.array([99.0, 98.0, 74.5])
queue_max  = np.array([88.0, 72.0,  0.0])
timeout_rt = np.array([46.1, 14.7,  0.0])

ax1.bar(x_r - w, kv_p90, width=w, color="#A6CCF0", edgecolor="#38628C", linewidth=0.7,
        label="KV Pool P90 (%)", zorder=3)
ax1.bar(x_r, queue_max, width=w, color="#F6CE95", edgecolor="#A35D22", linewidth=0.7,
        label="Peak Wait Queue", zorder=3)
ax1.bar(x_r + w, timeout_rt, width=w, color="#C83E3E", edgecolor="#7F1D1D", linewidth=0.7,
        label="Timeout Fail (%)", zorder=4)

ax1.axhline(75.0, color="#1D7A67", linestyle="--", linewidth=1.15, zorder=2)

for i in range(len(regimes)):
    ax1.text(x_r[i] - w, kv_p90[i] + 1.5, f"{kv_p90[i]:.0f}%", ha="center", va="bottom",
             fontsize=6.8, fontweight=600, color="#2D5278")
    ax1.text(x_r[i], queue_max[i] + 1.5, f"{queue_max[i]:.0f}", ha="center", va="bottom",
             fontsize=6.8, fontweight=600, color="#8C4A15")
    col_t = "#0B4F41" if timeout_rt[i] == 0 else "#A62626"
    txt_t = "0%\n(0/1304)" if timeout_rt[i] == 0 else f"{timeout_rt[i]:.1f}%"
    ax1.text(x_r[i] + w + (0.12 if timeout_rt[i] == 0 else 0.0), timeout_rt[i] + 1.5, txt_t,
             ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=col_t)

ax1.set_xticks(x_r)
ax1.set_xticklabels(regimes, fontsize=7.3, fontweight=600, color="#18181B")
ax1.set_yticks([0, 25, 50, 75, 100])
ax1.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=7.3)
ax1.set_ylabel("Usage / Queue / Fail (%)", fontsize=8.0, fontweight=600, color="#18181B")
ax1.set_title("(a) H200 KV Pool & Admission Gate", fontsize=8.7, fontweight="bold", pad=6, color="#18181B")
ax1.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#D4D4D8",
           fontsize=6.7, handlelength=1.2, handletextpad=0.45, borderpad=0.26, labelspacing=0.20)

# ==============================================================================
# PANEL (b): 4x H200 DAG-GRPO Valid Trained Tokens & Train-Infer KL (55 Steps)
# ==============================================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, 56.5)
ax2.set_ylim(0, 1300)
ax2.grid(True, zorder=0)
ax2.spines["top"].set_visible(False)

steps = np.arange(1, 56, dtype=float)
valid_tok_k = np.array([
    101.55, 220.60, 181.66, 144.34, 138.01, 218.33, 252.94, 228.04, 280.26, 262.72,
    446.97, 315.37, 312.46, 409.15, 435.56, 373.07, 499.12, 332.65, 491.86, 446.85,
    325.76, 514.98, 733.95, 546.79, 716.41, 692.89, 397.85, 484.30, 995.74, 528.07,
    464.43, 787.49, 610.83, 752.60, 915.92, 562.66,1066.90, 651.92, 776.93, 756.70,
    795.34, 831.93, 912.97, 872.77, 647.92, 539.44, 880.06, 643.76, 902.06, 906.24,
    990.22, 848.66, 774.84, 772.00,1123.00
])

kl_e4 = np.array([
     2.48,  4.91,  6.44,  9.54, 11.36, 15.61,  6.45,  6.90, 12.89,  4.06,
     6.40, 12.02, 10.87,  8.58,  6.71,  5.74,  6.72,  4.96,  6.19,  5.84,
     5.07,  4.56,  4.63,  4.98,  4.88,  4.81,  5.49,  3.93,  5.28,  4.80,
     4.83,  4.46,  4.03,  5.05,  3.85,  3.92,  4.49,  3.69,  3.90,  4.52,
     4.43,  4.28,  4.50,  4.09,  4.32,  3.17,  4.27,  4.65,  4.45,  3.94,
     4.80,  4.45,  4.39,  4.29,  3.93
])

l_tok, = ax2.plot(steps, valid_tok_k, color="#1D7A67", linestyle="-", linewidth=1.8,
                  label="Valid Tokens (102K→1.12M, 8.1×)", zorder=4)
ax2.fill_between(steps, 0, valid_tok_k, color="#1D7A67", alpha=0.10, zorder=2)

ax2_r = ax2.twinx()
ax2_r.set_ylim(0.0, 22.5)
ax2_r.spines["top"].set_visible(False)
l_kl, = ax2_r.plot(steps, kl_e4, color="#3B6998", linestyle="--", linewidth=1.45,
                   label="Train–Rollout KL (~4.4e-4)", zorder=3)

ax2.set_xticks([1, 15, 30, 45, 55])
ax2.set_xticklabels(["1", "15", "30", "45", "55"], fontsize=7.3)
ax2.set_yticks([0, 400, 800, 1200])
ax2.set_yticklabels(["0", "400K", "800K", "1.2M"], fontsize=7.2, color="#0B4F41")
ax2_r.set_yticks([0, 10, 20])
ax2_r.set_yticklabels(["0", "1e-3", "2e-3"], fontsize=7.2, color="#2D5278")
ax2.set_xlabel("DAG-GRPO Step (4× H200 Trainer)", fontsize=8.0, fontweight=600, color="#18181B", labelpad=3)
ax2.set_ylabel("Valid Tokens / Step", fontsize=8.0, fontweight=600, color="#0B4F41")
ax2_r.set_ylabel("Train–Rollout KL", fontsize=7.9, fontweight=600, color="#2D5278")
ax2.set_title("(b) 4× H200 Packed Token Scaling & KL", fontsize=8.7, fontweight="bold", pad=6, color="#18181B")
ax2.legend([l_tok, l_kl], [l_tok.get_label(), l_kl.get_label()],
           loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#D4D4D8",
           fontsize=6.6, handlelength=1.25, handletextpad=0.45, borderpad=0.25, labelspacing=0.20)

out_pdf = ROOT / "figures/fig7_training_dynamics.pdf"
out_svg = ROOT / "figures/fig7_training_dynamics.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("Saved true-print Figure 7:", out_pdf)
