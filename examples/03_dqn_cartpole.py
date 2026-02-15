"""
DQN + CartPole —— 用深度 Q 网络解决倒立摆问题

=== 从 Q 表到神经网络 ===
上一个例子（GridWorld）中，我们用一张 Q 表存储所有状态-动作的价值。
但 CartPole 的状态是 4 个连续值（位置、速度、角度、角速度），
组合出的状态数量是无限的——Q 表装不下了！

解决方案：用神经网络来"逼近" Q 函数。
输入：状态（4 个数）
输出：每个动作的 Q 值（2 个数：左推 / 右推）

=== DQN 的三个关键技巧 ===

1. 经验回放 (Experience Replay)
   把经历 (s, a, r, s', done) 存在一个"回忆库"里，
   训练时随机抽一批来学习，而不是只学刚发生的那一步。
   好处：打破时间相关性，让训练更稳定。

2. 目标网络 (Target Network)
   用两个结构相同的网络：
   - 策略网络 (policy_net)：负责选动作和训练
   - 目标网络 (target_net)：负责计算目标值，参数定期从策略网络复制
   好处：避免"自己追自己"的不稳定问题。

3. ε-贪心探索
   和 Q-Learning 一样：ε 概率随机，1-ε 概率选最优。
   ε 从 1.0 逐渐衰减到 0.01。

=== CartPole 环境 ===
- 一根杆子通过关节连在小车上
- 目标：通过左右推小车来保持杆子不倒
- 状态：[小车位置, 小车速度, 杆角度, 杆角速度]（4 个连续值）
- 动作：左推(0) / 右推(1)
- 奖励：每活一步 +1，杆倒了或出界就结束
- "解决"标准：连续 100 回合平均 >= 475 步
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.dqn import DQNAgent, train_dqn


def plot_results(episode_rewards, episode_steps, losses):
    """画三张图：奖励曲线、步数曲线、损失曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    window = 20

    # --- 图1: 每回合总奖励 ---
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.3, color="#5470c6", label="每回合奖励")
    if len(episode_rewards) >= window:
        smoothed = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax1.plot(
            range(window - 1, len(episode_rewards)), smoothed,
            color="#ee6666", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax1.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准 (475)")
    ax1.set_xlabel("回合 (Episode)")
    ax1.set_ylabel("总奖励 / 步数")
    ax1.set_title("学习曲线：每回合奖励")
    ax1.legend()

    # --- 图2: 每回合步数 ---
    ax2 = axes[1]
    ax2.plot(episode_steps, alpha=0.3, color="#91cc75", label="每回合步数")
    if len(episode_steps) >= window:
        smoothed_steps = np.convolve(
            episode_steps, np.ones(window) / window, mode="valid"
        )
        ax2.plot(
            range(window - 1, len(episode_steps)), smoothed_steps,
            color="#fac858", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax2.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准")
    ax2.set_xlabel("回合 (Episode)")
    ax2.set_ylabel("步数")
    ax2.set_title("每回合存活步数（越多越好）")
    ax2.legend()

    # --- 图3: 训练损失 ---
    ax3 = axes[2]
    ax3.plot(losses, alpha=0.3, color="#fc8452", label="每回合平均损失")
    if len(losses) >= window:
        smoothed_loss = np.convolve(
            losses, np.ones(window) / window, mode="valid"
        )
        ax3.plot(
            range(window - 1, len(losses)), smoothed_loss,
            color="#9a60b4", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax3.set_xlabel("回合 (Episode)")
    ax3.set_ylabel("MSE 损失")
    ax3.set_title("训练损失（Q 值的预测误差）")
    ax3.legend()

    plt.tight_layout()
    plt.savefig("examples/03_cartpole_result.png", dpi=100)
    print("\n图表已保存到 examples/03_cartpole_result.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 55)
    print("DQN + CartPole 实验")
    print("=" * 55)

    # ===== 创建环境 =====
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    print(f"\n环境: CartPole-v1")
    print(f"  状态维度: {state_dim}  (位置, 速度, 角度, 角速度)")
    print(f"  动作数量: {action_dim}  (0=左推, 1=右推)")
    print(f"  解决标准: 连续 100 回合平均 >= 475 步")

    # ===== 训练参数 =====
    episodes = 300
    hidden_dim = 128
    lr = 1e-3
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    buffer_capacity = 10000
    batch_size = 64
    target_update_freq = 10

    print(f"\n训练参数：")
    print(f"  回合数: {episodes}")
    print(f"  网络隐藏层: {hidden_dim} 个神经元")
    print(f"  学习率: {lr}")
    print(f"  折扣因子 γ: {gamma}")
    print(f"  探索率 ε: {epsilon} → {epsilon_min}")
    print(f"  经验回放容量: {buffer_capacity}")
    print(f"  批大小: {batch_size}")
    print(f"  目标网络更新频率: 每 {target_update_freq} 回合")

    # ===== 创建智能体 =====
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr, gamma=gamma,
        epsilon=epsilon, epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
    )

    # ===== 开始训练 =====
    print(f"\n开始训练（设备: {agent.device}）...")

    def on_episode_end(ep, reward, steps, eps):
        if (ep + 1) % 20 == 0:
            print(f"  回合 {ep+1:3d}/{episodes}  "
                  f"奖励: {reward:6.1f}  步数: {steps:3d}  "
                  f"ε: {eps:.3f}")

    episode_rewards, episode_steps, losses = train_dqn(
        env, agent, episodes=episodes, max_steps=500,
        on_episode_end=on_episode_end,
    )
    print("训练完成！")

    # ===== 结果分析 =====
    last_50_reward = np.mean(episode_rewards[-50:])
    last_50_steps = np.mean(episode_steps[-50:])
    max_reward = np.max(episode_rewards)

    print(f"\n最后 50 回合平均奖励: {last_50_reward:.1f}")
    print(f"最后 50 回合平均步数: {last_50_steps:.1f}")
    print(f"单回合最高奖励: {max_reward:.1f}")

    # 检查是否解决
    solved = False
    if len(episode_rewards) >= 100:
        for i in range(99, len(episode_rewards)):
            window_avg = np.mean(episode_rewards[max(0, i - 99):i + 1])
            if window_avg >= 475:
                print(f"\nCartPole 已解决！在第 {i - 99} 回合达到标准")
                solved = True
                break
    if not solved:
        print(f"\n未完全解决 CartPole（可以增加训练回合数试试）")

    # ===== 演示一个回合 =====
    print("\n--- 用训练好的智能体演示一回合 ---")
    state, _ = env.reset()
    total_reward = 0
    for step in range(500):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            break
    print(f"  演示回合: 存活 {step + 1} 步, 总奖励 {total_reward:.0f}")

    env.close()

    # ===== 画图 =====
    plot_results(episode_rewards, episode_steps, losses)

    # ===== 教学总结 =====
    print("\n" + "=" * 55)
    print("学到了什么？")
    print("=" * 55)
    print("""
1. 连续状态: CartPole 的状态是 4 个浮点数，Q 表装不下 → 用神经网络逼近
2. 经验回放: 把经历存起来随机抽样学习，打破时间相关性
3. 目标网络: 用"慢更新"网络计算目标值，避免训练不稳定
4. 网络结构: 输入状态(4维) → 隐藏层 → 输出各动作Q值(2维)
5. 本质不变: 核心仍是 Q(s,a) ← r + γ * max Q(s',a')，只是用网络代替了表格

对比 Q-Learning (GridWorld) vs DQN (CartPole)：
  Q-Learning: 离散状态 → Q 表 → 直接查表
  DQN:       连续状态 → 神经网络 → 前向传播算 Q 值

下一步：DQN 的改进版——Double DQN、Dueling DQN、Prioritized Replay 等，
或者转向 Policy Gradient 方法（直接学策略，而不是先学 Q 值再选动作）。
""")


if __name__ == "__main__":
    main()
