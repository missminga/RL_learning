"""
Q-Learning + GridWorld —— 用表格 Q-Learning 解决网格世界问题

=== 从老虎机到网格世界 ===
上一个例子（多臂老虎机）中，没有"状态"的概念——你只需要选拉哪台机器。
现在我们引入"状态"：智能体在一个网格中，每个位置就是一个状态。
智能体需要学会从起点走到终点，同时避开陷阱。

=== 什么是 Q-Learning？ ===
Q-Learning 是最经典的强化学习算法之一。它维护一个 Q 表：
- Q(s, a) 表示"在状态 s 执行动作 a 能获得多少长期奖励"
- 通过不断与环境交互，逐步更新 Q 表
- 学完之后，在每个状态选 Q 值最大的动作，就是最优策略

=== 核心公式 ===
Q(s, a) ← Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
                              └──────── TD 目标 ────────┘
    └──────────── TD 误差（目标 - 当前估计） ────────────┘

- α（学习率）: 每次更新多大幅度
- γ（折扣因子）: 未来奖励打多少折——越接近 1 越重视未来
- TD 误差: 新发现 vs 旧认知的差距——这就是学习的"动力"

=== 本例的网格世界 ===
5×5 的网格：
  S · · · ·      S = 起点 (0,0)
  · ▓ · ✕ ·      ★ = 终点 (4,4)   奖励 +10
  · · · ▓ ·      ✕ = 陷阱          奖励 -10
  · ✕ · · ·      ▓ = 墙壁（不可通行）
  · · · · ★      · = 空地          每步 -0.1
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用非交互后端，避免 display 问题
import matplotlib.pyplot as plt

# 把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.q_learning import (
    GridWorld, q_learning, extract_policy, ACTION_NAMES,
)


def print_grid(env, policy_arrows=None):
    """打印网格世界（带策略箭头）"""
    for r in range(env.rows):
        row_str = ""
        for c in range(env.cols):
            pos = (r, c)
            if pos == env.start:
                cell = " S "
            elif pos == env.goal:
                cell = " ★ "
            elif pos in env.traps:
                cell = " ✕ "
            elif pos in env.walls:
                cell = " ▓ "
            elif policy_arrows is not None:
                cell = f" {policy_arrows[r, c]} "
            else:
                cell = " · "
            row_str += cell
        print(row_str)


def plot_results(episode_rewards, episode_steps, q_table, env):
    """画三张图：学习曲线、步数曲线、策略网格"""
    fig = plt.figure(figsize=(14, 10))

    # --- 图1: 每回合总奖励 ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(episode_rewards, alpha=0.3, color="#5470c6", label="每回合奖励")
    # 滑动平均
    window = 20
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(episode_rewards)), smoothed,
                 color="#ee6666", linewidth=2, label=f"滑动平均 (窗口={window})")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("回合 (Episode)")
    ax1.set_ylabel("总奖励")
    ax1.set_title("学习曲线：每回合总奖励")
    ax1.legend()

    # --- 图2: 每回合步数 ---
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(episode_steps, alpha=0.3, color="#91cc75", label="每回合步数")
    if len(episode_steps) >= window:
        smoothed_steps = np.convolve(episode_steps, np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(episode_steps)), smoothed_steps,
                 color="#fac858", linewidth=2, label=f"滑动平均 (窗口={window})")
    ax2.set_xlabel("回合 (Episode)")
    ax2.set_ylabel("步数")
    ax2.set_title("每回合步数（越少越好）")
    ax2.legend()

    # --- 图3: 策略可视化 ---
    ax3 = fig.add_subplot(2, 2, (3, 4))

    # 画 Q 值热力图
    v_table = np.max(q_table.reshape(env.rows, env.cols, 4), axis=2)
    im = ax3.imshow(v_table, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax3, label="V(s) = max Q(s,a)")

    # 画箭头和特殊标记
    _, policy_arrows = extract_policy(q_table, env)
    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            if pos == env.start:
                text = "S"
                color = "blue"
            elif pos == env.goal:
                text = "G"
                color = "blue"
            elif pos in env.traps:
                text = "X"
                color = "red"
            elif pos in env.walls:
                text = "#"
                color = "black"
            else:
                text = policy_arrows[r, c]
                color = "black"
            ax3.text(c, r, text, ha="center", va="center",
                     fontsize=16, fontweight="bold", color=color)

    ax3.set_xticks(range(env.cols))
    ax3.set_yticks(range(env.rows))
    ax3.set_title("学到的策略（箭头=动作方向，背景色=状态价值）")
    ax3.set_xlabel("列")
    ax3.set_ylabel("行")

    plt.tight_layout()
    plt.savefig("examples/02_gridworld_result.png", dpi=100)
    print("\n图表已保存到 examples/02_gridworld_result.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 55)
    print("Q-Learning + GridWorld 实验")
    print("=" * 55)

    # ===== 环境设置 =====
    rows, cols = 5, 5
    traps = [(1, 3), (3, 1)]      # 陷阱位置
    walls = [(1, 1), (2, 3)]      # 墙壁位置

    env = GridWorld(rows, cols, traps=traps, walls=walls)

    print("\n网格世界地图：")
    print("  S=起点  ★=终点  ✕=陷阱(-10)  ▓=墙壁  ·=空地(-0.1/步)")
    print()
    print_grid(env)

    # ===== 训练参数 =====
    episodes = 500     # 训练 500 个回合
    alpha = 0.1        # 学习率
    gamma = 0.99       # 折扣因子（很重视未来奖励）
    epsilon = 1.0      # 初始探索率（一开始完全随机）
    epsilon_decay = 0.995   # 每回合 ε 乘以这个数
    epsilon_min = 0.01      # ε 的下限

    print(f"\n训练参数：")
    print(f"  回合数: {episodes}")
    print(f"  学习率 α: {alpha}")
    print(f"  折扣因子 γ: {gamma}")
    print(f"  探索率 ε: {epsilon} → {epsilon_min}（衰减率 {epsilon_decay}）")

    # ===== 开始训练 =====
    print("\n开始训练...")
    q_table, episode_rewards, episode_steps = q_learning(
        env, episodes=episodes, alpha=alpha, gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )
    print("训练完成！")

    # ===== 结果分析 =====
    last_50_reward = np.mean(episode_rewards[-50:])
    last_50_steps = np.mean(episode_steps[-50:])
    print(f"\n最后 50 回合平均奖励: {last_50_reward:.2f}")
    print(f"最后 50 回合平均步数: {last_50_steps:.1f}")

    # 打印学到的策略
    _, policy_arrows = extract_policy(q_table, env)
    print("\n学到的最优策略：")
    print_grid(env, policy_arrows)

    # ===== 演示一个回合 =====
    print("\n--- 用学到的策略走一遍 ---")
    state = env.reset()
    path = [state]
    for _ in range(20):
        s_idx = env.state_to_index(state)
        action = np.argmax(q_table[s_idx])
        next_state, reward, done = env.step(action)
        print(f"  {state} → 动作 {ACTION_NAMES[action]} → {next_state}  (奖励: {reward})")
        state = next_state
        path.append(state)
        if done:
            if state == env.goal:
                print("  到达终点！")
            else:
                print("  掉进陷阱！")
            break

    print(f"\n路径长度: {len(path) - 1} 步")

    # ===== 画图 =====
    plot_results(episode_rewards, episode_steps, q_table, env)

    # ===== 教学总结 =====
    print("\n" + "=" * 55)
    print("学到了什么？")
    print("=" * 55)
    print("""
1. 状态 (State): 智能体的位置——多臂老虎机没有状态，GridWorld 有
2. Q 表: Q(s, a) 记录"在 s 做 a 能获得多少长期奖励"
3. TD 学习: 不需要等到回合结束，每一步都能更新——比蒙特卡洛高效
4. ε 衰减: 一开始多探索（ε 大），后期多利用（ε 小）
5. 折扣因子 γ: 控制"眼前利益 vs 长远利益"的平衡

下一步：当状态空间太大时（比如连续状态），Q 表装不下了，
需要用神经网络来近似 Q 函数 → 这就是 Deep Q-Network (DQN)！
""")


if __name__ == "__main__":
    main()
