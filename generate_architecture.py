# generate_architecture.py
# =====================================================================
# 生成「全流程投研智能体」系统架构图（PNG），供商业计划书/PPT 使用
# 输出：./output/system_architecture.png
# =====================================================================
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配色：数据层灰、因子层蓝、沉淀层绿、组合层橙、回测层紫、知识层金
C = {
    'data': '#e8eef7', 'factor': '#dbe9f9', 'card': '#e2f3e2',
    'comb': '#fdeeda', 'backtest': '#ece4f6', 'knowledge': '#fdf3d1',
}


def box(ax, x, y, w, h, text, color, fs=10):
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03',
                       facecolor=color, edgecolor='#444', linewidth=1.4)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                        mutation_scale=18, color='#555', linewidth=1.8)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 15))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 17)
    ax.axis('off')

    ax.text(6.5, 16.4, '可信金融投研智能体', ha='center',
            va='center', fontsize=16, fontweight='bold')
    ax.text(6.5, 15.7, '让每一份研究结论，都经得起复现（达观数据命题 · 产教协同创新组）', ha='center',
            va='center', fontsize=10, color='#666')

    # ---- 数据层 ----
    ax.text(6.5, 14.9, '① 数据层', ha='center', fontsize=12, fontweight='bold', color='#2b5797')
    box(ax, 1.2, 13.7, 3.4, 0.9, '新闻/舆情文本\n(FinBERT/LLM 情感分析)', C['data'])
    box(ax, 4.8, 13.7, 3.4, 0.9, 'ETF 行情(13 标的)\n价量/动量/波动率', C['data'])
    box(ax, 8.4, 13.7, 3.4, 0.9, '研报/论文 PDF\n(研究假设与因子)', C['data'])

    # ---- 因子构建层 ----
    ax.text(6.5, 12.4, '② 因子构建层（研报复现）', ha='center', fontsize=12, fontweight='bold', color='#2b5797')
    box(ax, 2.3, 11.0, 3.6, 0.9, '情绪因子\n(全市场+大盘聚焦)', C['factor'])
    box(ax, 7.1, 11.0, 3.6, 0.9, '价量因子\n动量/反转/波动率/量能', C['factor'])
    arrow(ax, 6.5, 13.7, 4.1, 12.0)
    arrow(ax, 6.5, 13.7, 8.9, 12.0)

    # ---- 因子知识沉淀层 ----
    ax.text(6.5, 9.9, '③ 因子知识沉淀层（因子卡片）', ha='center', fontsize=12, fontweight='bold', color='#2b5797')
    box(ax, 2.3, 8.3, 8.4, 1.1, 'IC / ICIR / 单调性 / 衰减 / 冗余识别 → 自动判定「有效·冗余·另类」入因子库', C['card'], fs=10.5)
    arrow(ax, 4.1, 11.0, 5.5, 9.5)
    arrow(ax, 8.9, 11.0, 7.5, 9.5)

    # ---- 智能因子组合层 ----
    ax.text(6.5, 7.3, '④ 智能因子组合层', ha='center', fontsize=12, fontweight='bold', color='#2b5797')
    box(ax, 2.3, 5.7, 8.4, 1.1, '多标的轮动 + 趋势择时 + 情绪过滤 + 回撤止损\n（纳入手续费/滑点/换手/仓位/流动性约束）', C['comb'])
    arrow(ax, 6.5, 8.3, 6.5, 6.9)

    # ---- 多层次回测层 ----
    ax.text(6.5, 4.7, '⑤ 多层次回测层', ha='center', fontsize=12, fontweight='bold', color='#2b5797')
    box(ax, 2.3, 3.1, 8.4, 1.1, '样本外验证 + walk-forward + 压力情景 + 生成式多轨迹模拟\n（降低过拟合 / 伪相关风险）', C['backtest'])
    arrow(ax, 6.5, 5.7, 6.5, 4.3)

    # ---- 知识沉淀层 ----
    ax.text(6.5, 2.2, '⑥ 知识沉淀 / 自我进化', ha='center', fontsize=12, fontweight='bold', color='#2b5797')
    box(ax, 2.3, 0.7, 8.4, 1.0, '因子卡片库 + 研究结论库（可复用资产，持续改善研究流程）', C['knowledge'])
    arrow(ax, 6.5, 3.1, 6.5, 1.8)

    os.makedirs('./output', exist_ok=True)
    out = './output/system_architecture.png'
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'✅ 架构图已生成: {out}')


if __name__ == '__main__':
    main()
