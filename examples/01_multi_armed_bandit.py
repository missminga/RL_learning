"""
多臂老虎机 (Multi-Armed Bandit) —— 强化学习最简单的入门例子

=== 什么是多臂老虎机？===
想象你面前有 10 台老虎机（slot machine），每台机器拉下把手后会给你一个随机奖励。
不同的机器给的奖励不同，但你事先不知道哪台最好。
你的目标：在有限次数内，尽可能多地赢取奖励。

=== 核心概念 ===
- 动作 (Action): 选择拉哪台老虎机
- 奖励 (Reward): 拉完之后得到的回报
- 探索 (Exploration): 尝试没怎么拉过的机器，看看它好不好
- 利用 (Exploitation): 拉目前看来最好的那台机器

=== 关键问题 ===
如果你只拉目前看起来最好的机器（纯利用），可能会错过真正最好的机器。
如果你一直随机试（纯探索），又浪费了很多次机会。
所以需要平衡探索和利用 —— 这就是强化学习的核心思想之一！

=== 本例使用的策略：ε-贪心 (Epsilon-Greedy) ===
- 以 ε（比如 10%）的概率随机选一台机器（探索）
- 以 1-ε（比如 90%）的概率选当前估计最好的机器（利用）
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 把项目根目录加入 sys.path，以便从 core/ 导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.bandits import run_comparison


def main():
    """主函数：对比不同 ε 值的效果"""

    print("=" * 50)
    print("多臂老虎机实验 (Multi-Armed Bandit)")
    print("=" * 50)

    # 实验参数
    n_runs = 200       # 重复实验次数（取平均，让结果更稳定）
    steps = 1000       # 每次实验拉 1000 次
    k = 10             # 10 台老虎机
    epsilons = [0, 0.01, 0.1]  # 对比三种 ε 值

    print(f"\n实验设置：{k} 台老虎机，每次实验 {steps} 步，重复 {n_runs} 次取平均")
    print(f"对比的 ε 值：{epsilons}")
    print("\n开始实验...")

    # 调用核心算法
    result = run_comparison(epsilons, k=k, steps=steps, n_runs=n_runs)

    print("实验完成！\n")

    # ===== 打印结果摘要 =====
    print("结果摘要（最后 100 步的平均值）：")
    print("-" * 40)
    for item in result["summary"]:
        eps = item["epsilon"]
        print(f"  ε = {eps:>4}  →  平均奖励: {item['avg_reward']:.2f},  最优动作比例: {item['optimal_pct']:.1f}%")

    print("\n结论：")
    print("  ε = 0（纯贪心）：容易陷入次优选择")
    print("  ε = 0.01（少量探索）：长期效果最好，但学得慢")
    print("  ε = 0.1（适度探索）：学得快，但会持续浪费 10% 在随机探索上")

    # ===== 画图 =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('多臂老虎机实验结果', fontsize=14)

    # 图1：平均奖励
    for eps in epsilons:
        label = f'ε = {eps}' if eps > 0 else 'ε = 0 (纯贪心)'
        ax1.plot(result["rewards"][str(eps)], label=label)
    ax1.set_xlabel('步数')
    ax1.set_ylabel('平均奖励')
    ax1.set_title('不同 ε 值的平均奖励对比')
    ax1.legend()

    # 图2：最优动作比例
    for eps in epsilons:
        label = f'ε = {eps}' if eps > 0 else 'ε = 0 (纯贪心)'
        ax2.plot(result["optimal_pct"][str(eps)], label=label)
    ax2.set_xlabel('步数')
    ax2.set_ylabel('最优动作比例 (%)')
    ax2.set_title('不同 ε 值选择最优动作的比例')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('examples/01_bandit_result.png', dpi=100)
    print("\n图表已保存到 examples/01_bandit_result.png")
    plt.show()


if __name__ == '__main__':
    main()
