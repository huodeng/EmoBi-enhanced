import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np

font_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
chosen_font = None
for name in font_candidates:
    for f in fm.fontManager.ttflist:
        if name.lower() in f.name.lower():
            chosen_font = f.name
            break
    if chosen_font:
        break
if chosen_font:
    plt.rcParams["font.family"] = [chosen_font, "sans-serif"]

groups = ["wo_feature\n8B", "wo_feature\n70B", "full\n8B", "full\n70B"]
hyper_f1 = [0.9655, 0.8462, 0.9474, 0.7755]
meta_f1  = [0.6471, 0.7273, 0.6897, 0.7619]

x = np.arange(len(groups))
width = 0.32

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

bars_h = ax.bar(x - width/2, hyper_f1, width, color="#1E88E5", edgecolor="white", linewidth=0.8)
bars_m = ax.bar(x + width/2, meta_f1,  width, color="#26A69A", edgecolor="white", linewidth=0.8)

for bar, val in zip(bars_h, hyper_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012, f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0D47A1")
for bar, val in zip(bars_m, meta_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012, f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#00695C")

title = "模型规模消融对比：HYPO-L (8B vs 70B, Fast Mode, n=30)" if chosen_font else "Model Scale Ablation: HYPO-L (8B vs 70B, Fast Mode, n=30)"
ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
ax.grid(axis="y", linestyle="--", alpha=0.35)

legend_labels = ["Hyperbole F1 (夸张)", "Metaphor F1 (隐喻)"] if chosen_font else ["Hyperbole F1", "Metaphor F1"]
legend_patches = [plt.Rectangle((0,0),1,1,fc="#1E88E5"), plt.Rectangle((0,0),1,1,fc="#26A69A")]
ax.legend(legend_patches, legend_labels, fontsize=11, loc="upper right", framealpha=0.9)

# annotate meta improvements from 8B to 70B
delta_wo = meta_f1[1] - meta_f1[0]
delta_full = meta_f1[3] - meta_f1[2]
if chosen_font:
    ax.annotate(f"Meta +{delta_wo:.4f}\n(8B→70B)", xy=(1, meta_f1[1] + 0.02), xytext=(1, 0.93),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
                fontsize=9, fontweight="bold", color="#D32F2F", ha="center")
    ax.annotate(f"Meta +{delta_full:.4f}\n(8B→70B)", xy=(3, meta_f1[3] + 0.02), xytext=(3, 0.93),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
                fontsize=9, fontweight="bold", color="#D32F2F", ha="center")
else:
    ax.annotate(f"Meta +{delta_wo:.4f}\n(8B→70B)", xy=(1, meta_f1[1] + 0.02), xytext=(1, 0.93),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
                fontsize=9, fontweight="bold", color="#D32F2F", ha="center")
    ax.annotate(f"Meta +{delta_full:.4f}\n(8B→70B)", xy=(3, meta_f1[3] + 0.02), xytext=(3, 0.93),
                arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
                fontsize=9, fontweight="bold", color="#D32F2F", ha="center")

fig.tight_layout()
out_path = "results/tables/ablation_chart.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"saved: {out_path}")
