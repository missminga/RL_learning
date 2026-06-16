"""
PPO (近端策略优化) + CartPole —— 给 A2C 装上"安全带"，反复榨干每批数据

=== 回顾：A2C 的两个遗憾 ===
A2C 跑完一个回合，用 GAE 算优势，然后**只更新一步**就把数据扔了：
  1. 浪费：辛苦采来的一整回合数据只用一次，采样效率低。
  2. 危险：策略梯度只在"当前策略附近"才可靠，一步迈太大就可能把策略推到
     没被验证过的区域，表现突然崩塌（A2C 后期的"暴跌"就是这么来的）。

=== PPO 的两板斧：裁剪 + 数据复用 ===
PPO（Proximal Policy Optimization）在 A2C 之上加两样东西：

1. 重要性采样比值 r = π_新(a|s) / π_旧(a|s)
   —— 衡量"更新后的新策略，比采样时的旧策略，多想/少想选这个动作"。

2. 裁剪目标（clip）：把每次更新的步子焊死在一个小区间里
       L = min( r·A,  clip(r, 1-ε, 1+ε)·A )      （ε 常取 0.2）
   含义："新旧策略对任一动作的概率比，最多变化 ±20%。" 想迈大也迈不出去。

有了这条"安全带"，PPO 就能放心地把**同一批数据反复训练 K 个 epoch**
（分 minibatch），既省数据、又因为有裁剪而不会迈过头 —— 这就是 PPO
比 A2C 又快又稳的根本原因。

=== 一次更新做什么 ===
  采样：用旧策略跑一回合，记录 (state, action, old_log_prob, value, reward)
  算优势：GAE → A_t、returns（和 A2C 完全一样）
  标准化优势：减均值除标准差（PPO 常规操作）
  复用 K 个 epoch，每个 epoch 分 minibatch：
      ratio = exp(new_log_prob - old_log_prob)
      actor_loss  = -min(ratio·A, clip(ratio,1-ε,1+ε)·A).mean()   ← 裁剪
      critic_loss =  (V - returns)²                                ← 同 A2C
      total = actor_loss + 0.5*critic_loss - 0.01*entropy

=== A2C vs PPO 一览 ===
┌─────────────┬──────────────────────┬──────────────────────────┐
│             │ A2C                  │ PPO                      │
├─────────────┼──────────────────────┼──────────────────────────┤
│ 数据复用     │ 一批更新 1 次         │ 一批更新 K 个 epoch       │
│ 限制步长     │ 无（靠小学习率硬撑）   │ clip 锁在 [1-ε, 1+ε]     │
│ 采样效率     │ 低                   │ 高                       │
│ 稳定性       │ 后期易崩溃            │ 裁剪保证不迈过头，稳得多   │
└─────────────┴──────────────────────┴──────────────────────────┘

PPO 是目前最主流的策略优化算法，ChatGPT 的 RLHF 也用它。
理解了 A2C，PPO 只是在它之上加了"裁剪 + 数据复用"两板斧。

=== CartPole 环境 ===
和前几个例子相同：
- 状态：[小车位置, 小车速度, 杆角度, 杆角速度]（4 维连续）
- 动作：左推(0) / 右推(1)
- 奖励：每活一步 +1
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.ppo import PPOAgent, train_ppo


def plot_results(episode_rewards, episode_steps, actor_losses, critic_losses):
    """画四张图：奖励、步数、Actor 损失、Critic 损失"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    window = 20

    # --- 图1: 每回合总奖励 ---
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.3, color="#5470c6", label="每回合奖励")
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, len(episode_rewards)), smoothed,
            color="#ee6666", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax1.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准 (475)")
    ax1.set_xlabel("回合 (Episode)")
    ax1.set_ylabel("总奖励")
    ax1.set_title("PPO 学习曲线")
    ax1.legend()

    # --- 图2: 每回合步数 ---
    ax2 = axes[1]
    ax2.plot(episode_steps, alpha=0.3, color="#91cc75", label="每回合步数")
    if len(episode_steps) >= window:
        smoothed_steps = np.convolve(episode_steps, np.ones(window) / window, mode="valid")
        ax2.plot(
            range(window - 1, len(episode_steps)), smoothed_steps,
            color="#fac858", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax2.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="解决标准")
    ax2.set_xlabel("回合 (Episode)")
    ax2.set_ylabel("步数")
    ax2.set_title("每回合存活步数")
    ax2.legend()

    # --- 图3: Actor 损失 ---
    ax3 = axes[2]
    ax3.plot(actor_losses, alpha=0.3, color="#fc8452", label="Actor 损失")
    if len(actor_losses) >= window:
        smoothed_a = np.convolve(actor_losses, np.ones(window) / window, mode="valid")
        ax3.plot(
            range(window - 1, len(actor_losses)), smoothed_a,
            color="#9a60b4", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax3.set_xlabel("回合 (Episode)")
    ax3.set_ylabel("Actor 损失")
    ax3.set_title("演员损失（裁剪策略部分）")
    ax3.legend()

    # --- 图4: Critic 损失 ---
    ax4 = axes[3]
    ax4.plot(critic_losses, alpha=0.3, color="#73c0de", label="Critic 损失")
    if len(critic_losses) >= window:
        smoothed_c = np.convolve(critic_losses, np.ones(window) / window, mode="valid")
        ax4.plot(
            range(window - 1, len(critic_losses)), smoothed_c,
            color="#3ba272", linewidth=2, label=f"滑动平均 (窗口={window})",
        )
    ax4.set_xlabel("回合 (Episode)")
    ax4.set_ylabel("Critic 损失")
    ax4.set_title("评委损失（价值估计误差）")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("examples/06_ppo_result.png", dpi=100)
    print("\n图表已保存到 examples/06_ppo_result.png")
    plt.show()


def main():
    """主函数"""
    print("=" * 55)
    print("PPO (近端策略优化) + CartPole 实验")
    print("=" * 55)

    # ===== 创建环境 =====
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    print("\n环境: CartPole-v1")
    print(f"  状态维度: {state_dim}")
    print(f"  动作数量: {action_dim}")

    # ===== 训练参数 =====
    episodes = 500
    hidden_dim = 128
    lr = 3e-3
    gamma = 0.99
    gae_lambda = 0.95     # GAE 的 λ，降低优势的方差
    clip_eps = 0.2        # 裁剪范围 ε：新旧策略概率比锁在 [0.8, 1.2]
    value_coef = 0.5      # Critic 损失权重
    entropy_coef = 0.01   # 熵奖励权重
    update_epochs = 10    # 同一批数据反复训练几个 epoch（PPO 的精髓）
    minibatch_size = 64   # 每个 minibatch 的样本数

    print("\n训练参数：")
    print(f"  回合数: {episodes}")
    print(f"  网络隐藏层: {hidden_dim} 个神经元（Actor 与 Critic 各一套独立网络）")
    print(f"  学习率: {lr}")
    print(f"  折扣因子 γ: {gamma}，GAE λ: {gae_lambda}")
    print(f"  裁剪范围 ε: {clip_eps}")
    print(f"  数据复用 epoch 数: {update_epochs}，minibatch 大小: {minibatch_size}")
    print(f"  Critic 损失权重: {value_coef}，熵奖励权重: {entropy_coef}")
    print("\n相比 A2C：用裁剪锁住每步幅度，于是能把每批数据反复用 K 个 epoch，又快又稳。")

    # ===== 创建智能体 =====
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
    )

    # ===== 开始训练 =====
    print(f"\n开始训练（设备: {agent.device}）...")

    def on_episode_end(ep, reward, steps, loss):
        if (ep + 1) % 50 == 0:
            print(f"  回合 {ep+1:3d}/{episodes}  "
                  f"奖励: {reward:6.1f}  步数: {steps:3d}  "
                  f"总损失: {loss:8.3f}")

    episode_rewards, episode_steps, total_losses, actor_losses, critic_losses = train_ppo(
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
        print("\n未完全解决 CartPole（可以多跑几次或增加回合数试试）")

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
    # 演示结束后清空 agent 的记录（不需要更新）
    agent.states = []
    agent.actions = []
    agent.log_probs = []
    agent.values = []
    agent.rewards = []
    print(f"  演示回合: 存活 {step + 1} 步, 总奖励 {total_reward:.0f}")

    env.close()

    # ===== 画图 =====
    plot_results(episode_rewards, episode_steps, actor_losses, critic_losses)

    # ===== 教学总结 =====
    print("\n" + "=" * 55)
    print("学到了什么？")
    print("=" * 55)
    print("""
1. 重要性采样比值 r = π_新/π_旧: 衡量更新前后策略对某动作偏好的变化
2. 裁剪 clip: L = min(r·A, clip(r,1-ε,1+ε)·A)，把每步幅度锁在 ±ε 内，
   想迈大也迈不出去 —— 这是 PPO 稳定的核心
3. 数据复用: 有了裁剪这条"安全带"，同一批数据可以反复训练 K 个 epoch，
   采样效率远高于 A2C 的"用一次就扔"
4. 优势标准化: 减均值除标准差，让裁剪阈值 ε 有稳定一致的意义
5. 其余照搬 A2C: GAE 算优势、Critic 回归 returns、熵奖励鼓励探索、
   截断时用 V(末状态) 自举 —— 这些 PPO 和 A2C 完全一样

对比 A2C vs PPO：
  A2C: 一批数据更新一步，无步长限制 → 采样效率低、后期易崩溃
  PPO: 裁剪锁步长 + 一批数据反复用 → 又快又稳，本实验通常 100 多回合就解决

承上启下：
  - 往前看：PPO = A2C(演员-评委 + GAE) + 重要性采样裁剪 + 多轮数据复用
  - 往后看：PPO 是当今 RLHF（如 ChatGPT 训练）的主力算法；连续动作场景
    可换成高斯策略，或了解同样主流的 SAC、以及更新的 GRPO 等
""")


if __name__ == "__main__":
    main()
