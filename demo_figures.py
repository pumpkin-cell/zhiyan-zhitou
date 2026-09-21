# demo_figures.py
# =====================================================================
# 离线演示素材：不依赖浏览器，用真实数据生成 Streamlit 三个任务的核心输出图
# 用法：python demo_figures.py
# 产出：./output/demo/ 下 3 张 PNG（可直接放进 PPT）
# =====================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = "./output/demo"
os.makedirs(OUT, exist_ok=True)


def fig1_factor_reproduce():
    """任务①：因子复现——反转因子真实净值曲线。"""
    from reproduce_batch import load_prices, cross_sectional
    from metrics import calc_return_metrics
    prices = load_prices()
    ls = cross_sectional(prices, 20, reverse=True, hold=20)  # 反转因子
    nav = np.cumprod(1 + ls)
    m = calc_return_metrics(ls)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(nav, color='#2b5797', lw=1.5)
    ax.set_title(f"因子复现：反转因子净值（年化 {m['annual_return_pct']:.2f}%，夏普 {m['sharpe_ratio']:.2f}）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_因子复现_净值.png", dpi=150)
    plt.close(fig)
    print("✅ 01_因子复现_净值.png")


def fig2_factor_cards():
    """任务②：因子卡片——IC 一览（真实因子卡片数据）。"""
    df = pd.read_csv("./data/factor_cards.csv")
    df = df.assign(abs_ic=df['ic'].abs()).sort_values('abs_ic', ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ['#2b5797' if v >= 0 else '#c00000' for v in df['ic']]
    ax.barh(df['factor'], df['ic'], color=colors)
    ax.set_title("因子卡片：因子 IC 一览（红=反向指标）")
    ax.axvline(0, color='black', lw=0.8)
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_因子卡片_IC.png", dpi=150)
    plt.close(fig)
    print("✅ 02_因子卡片_IC.png")


def fig3_agent_flow():
    """任务③：自然语言智能体——三步流程示意。"""
    steps = ["① 意图理解\n(自然语言→因子定义)",
             "② 确定性复现\n(回测引擎执行)",
             "③ 自动报告\n(大模型生成)"]
    fig, ax = plt.subplots(figsize=(9, 3))
    for i, s in enumerate(steps):
        ax.add_patch(plt.Rectangle((i * 0.32, 0.3), 0.26, 0.4, color='#2b5797', alpha=0.9))
        ax.text(i * 0.32 + 0.13, 0.5, s, ha='center', va='center', color='white', fontsize=11)
        if i < 2:
            ax.annotate('', xy=(i * 0.32 + 0.27, 0.5), xytext=(i * 0.32 + 0.31, 0.5),
                        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("自然语言智能体：一句话 → 复现 → 报告（全自动）", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_智能体_流程.png", dpi=150)
    plt.close(fig)
    print("✅ 03_智能体_流程.png")


if __name__ == "__main__":
    fig1_factor_reproduce()
    fig2_factor_cards()
    fig3_agent_flow()
    print("✅ 全部演示素材已生成到", OUT)
