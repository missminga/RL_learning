"""
指令跟随版 Ant —— 从"无脑往前跑"到"听你指挥跑"（真实机器狗的标准形态）

=== 本例在学什么 ===

07 号例子里蚂蚁的奖励是"往 +x 方向跑得越快越好"——它只会向前冲。
真实的四足机器人（比如宇树 Go2）不是这样工作的：它们学的是**速度指令跟踪**，
遥控器摇杆给出 (前进速度 vx*, 侧移速度 vy*, 转向角速度 ω*)，策略负责执行。

实现这件事只需要改"任务"（环境），算法一行不用动：

1. **观测拼接指令**：27 维本体状态 + 3 维指令 → 30 维
   策略网络眼里指令和身体姿态没有区别，它学到的是
   "当指令是 (0.8, 0, 0) 时该怎么迈步，是 (0, 0, 1.0) 时该怎么转"
2. **奖励改跟踪误差**：不再是跑得越快越好，而是"实际速度越接近指令越好"
3. **训练时随机换指令**：每 5 秒随机重采样一个新指令（含"站住"），
   策略见够各种指令后，部署时你给什么它都会

这就是 **command-conditioned policy（指令条件化策略）**，Isaac Lab 里
locomotion 任务名全是 `Isaac-Velocity-xxx` —— Velocity 开头就是这个意思。

=== 实测趟出来的三个坑（比算法本身更值得学）===

v1: 奖励核 σ 太宽 → 站着不动也能蹭分，蚂蚁"躺平"（宁可不跑）
v2: 从零训练 + 摔倒即终止 → "动会摔死、站着安全"，还是躺平；
    σ 放开探索太大 → 直接把预训练的步态洗没
v3（本版）: 迁移学习（加载 07 的会走策略，输入层扩 3 维指令列置零）
    + 关闭摔倒终止 + 小 σ(0.25) + 大批次(8192，条件信号才稳定)
    —— 这套"预训练→微调"正是 LLM 的 SFT→RLHF、机器人预训练→真机微调的配方

=== 训练 vs 部署 ===

训练：指令由环境随机生成（域随机化的思想，见够各种情况）
部署：指令由你给出（本例提供键盘遥控模式 --play，以及自动演示视频）

用法:
    # 训练（默认加载 07 的预训练蚂蚁策略微调，约 1.5M 步）
    uv run python examples/08_ppo_command_ant.py

    # 从零训练（慢且可能躺平，用于体会上面 v1/v2 的坑）
    uv run python examples/08_ppo_command_ant.py --init-from ""

    # 训练后键盘遥控（弹出实时窗口，输入 w/s/a/d/q/e/x 控制）
    uv run python examples/08_ppo_command_ant.py --play
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.ppo_continuous import PPOContinuousAgent, train_ppo_continuous

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 指令维度含义
CMD_NAMES = ["前进vx*", "侧移vy*", "转向ω*"]


class VelocityCommandWrapper(gym.Wrapper):
    """
    速度指令跟踪 wrapper：把 Ant 从"往前跑"改造成"听指令跑"

    - 观测：原 27 维 + 3 维指令 (vx*, vy*, ω*) = 30 维
    - 奖励：跟踪误差（指数形式，误差越小分越高）+ 存活奖励 - 控制成本
    - 每 command_interval 步随机重采样指令（含一定概率全零=站住不动）

    速度真值从哪来：
    - vx, vy：环境 info 里自带（x_velocity / y_velocity）
    - ω（转向角速度）：自由关节的 qvel[5]（z 轴角速度）
    """

    def __init__(
        self,
        env,
        command_interval=100,  # 每 100 步（5 秒 × 20Hz）换一次指令
        vx_range=(-0.5, 2.0),  # 前进/后退速度指令范围 (m/s)，上限含蚂蚁的惯用速度
        vy_range=(-0.75, 0.75),  # 侧移速度指令范围 (m/s)
        wyaw_range=(-1.2, 1.2),  # 转向角速度指令范围 (rad/s)
        stand_prob=0.15,  # 指令全零（站住）的概率
    ):
        super().__init__(env)
        self.command_interval = command_interval
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.wyaw_range = wyaw_range
        self.stand_prob = stand_prob

        self._cmd = np.zeros(3)  # 当前指令 (vx*, vy*, ω*)
        self._steps_since_cmd = 0
        self._cmd_hold = False  # True 时禁止自动换指令（考试/遥控模式用）
        # 课程学习（curriculum learning）：先只练前进速度指令（'forward_only'），
        # 学会调速、读懂 obs 里的指令位之后，再切到全指令（'full'）。
        # 原因：从零/预训练直接上全指令时，Actor 还读不懂指令，
        # "站住"指令段的负优势会无差别惩罚步态 → 把会走的策略洗没
        self.cmd_mode = "full"

        # 观测空间扩展 3 维
        low = np.concatenate(
            [env.observation_space.low, [vx_range[0], vy_range[0], wyaw_range[0]]]
        ).astype(np.float32)
        high = np.concatenate(
            [env.observation_space.high, [vx_range[1], vy_range[1], wyaw_range[1]]]
        ).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _sample_command(self):
        """随机采样一个新指令（stand_prob 概率站住）"""
        if np.random.rand() < self.stand_prob:
            self._cmd = np.zeros(3)
        elif self.cmd_mode == "forward_only":
            # 课程阶段 1：只练前进速度（含慢速/停止附近的低速）
            self._cmd = np.array([np.random.uniform(0.0, 2.0), 0.0, 0.0])
        else:
            self._cmd = np.array(
                [
                    np.random.uniform(*self.vx_range),
                    np.random.uniform(*self.vy_range),
                    np.random.uniform(*self.wyaw_range),
                ]
            )
        self._steps_since_cmd = 0

    def set_command(self, vx, vy, wyaw, hold=True):
        """部署时用：外部（键盘/规划器）设置指令

        hold=True 时"按住不放"：禁止 wrapper 定时重采样，直到下次 reset
        （否则考试/遥控时指令会被自动换掉，测到的就不是指定的指令了）
        """
        self._cmd = np.array([vx, vy, wyaw], dtype=np.float64)
        self._steps_since_cmd = 0
        self._cmd_hold = hold

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._cmd_hold = False
        self._sample_command()
        return np.concatenate([obs, self._cmd]).astype(np.float32), info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # 定时换指令（hold 模式下不换）
        self._steps_since_cmd += 1
        if not self._cmd_hold and self._steps_since_cmd >= self.command_interval:
            self._sample_command()

        # 实际速度：线速度从 info，转向角速度从 qvel[5]（自由关节的 z 轴角速度）
        vx, vy = info["x_velocity"], info["y_velocity"]
        wyaw = float(self.env.unwrapped.data.qvel[5])
        vx_c, vy_c, wyaw_c = self._cmd

        # 跟踪奖励：exp(-误差²/σ²)，σ²=0.5 取宽核 —— 在 0~3 m/s 都有平滑梯度。
        # 实测趟坑记录：σ 太宽 → 站着蹭分躺平（v1）；σ 太窄 → 远离目标梯度为零，
        # 把预训练步态洗没（v2）；线性核在无终止环境下全程负奖励（v4）。
        # 宽核 + 预训练初始化 + 大批次是单环境 CPU 上唯一能出指令响应的组合
        track_lin = np.exp(-((vx - vx_c) ** 2 + (vy - vy_c) ** 2) / 0.5)
        track_yaw = np.exp(-((wyaw - wyaw_c) ** 2) / 0.5)
        # 少量存活奖励防"宁可摔倒"，控制成本防"疯狂抽搐"
        reward = (
            2.0 * track_lin
            + 0.5 * track_yaw
            + 0.1
            - 0.001 * float(np.square(action).sum())
        )

        # 记录跟踪误差（训练曲线用）
        info["cmd_lin_err"] = float(np.hypot(vx - vx_c, vy - vy_c))
        info["cmd_yaw_err"] = float(abs(wyaw - wyaw_c))

        obs_full = np.concatenate([obs, self._cmd]).astype(np.float32)
        return obs_full, reward, terminated, truncated, info


def plot_results(update_rewards, update_steps, actor_losses, critic_losses, out_path):
    """画四张图：批内平均奖励、平均存活步数、Actor 损失、Critic 损失"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    window = 20

    for ax, y, color, label in [
        (axes[0], update_rewards, "#5470c6", "批内平均奖励"),
        (axes[1], update_steps, "#91cc75", "批内平均存活步数"),
        (axes[2], actor_losses, "#fc8452", "Actor 损失"),
        (axes[3], critic_losses, "#73c0de", "Critic 损失"),
    ]:
        ax.plot(y, alpha=0.3, color=color, label=label)
        if len(y) >= window:
            smoothed = np.convolve(y, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, len(y)),
                smoothed,
                color="#ee6666",
                linewidth=2,
                label=f"滑动平均 (窗口={window})",
            )
        ax.set_xlabel("更新次数 (Update)")
        ax.legend()

    axes[0].set_title("指令跟踪奖励")
    axes[1].set_title("存活步数")
    axes[2].set_title("演员损失")
    axes[3].set_title("评委损失")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"\n图表已保存到 {out_path}")


def load_pretrained_locomotion(agent, ckpt_path, obs_body_dim=27):
    """
    迁移学习：把 07 训好的"会走"策略搬进指令跟随网络

    07 的策略输入是 27 维本体状态，本例是 30 维（+3 维指令）。
    做法：拷贝全部共享权重；输入层新增的 3 列（指令维度）置零 ——
    这样初始行为和原策略**完全一致**（指令暂时"看不见"），微调时再学会读它。

    这就是"先会走，再学听指令"：和 LLM 的 SFT→RLHF、机器人的
    预训练→真机微调是同一个配方。
    """
    old = torch.load(ckpt_path, weights_only=True)
    new = agent.net.state_dict()
    for key in new:
        if key in ("actor.0.weight", "critic.0.weight"):
            # [hidden, 30] = [旧的 27 列 | 3 列零]
            new[key] = torch.cat([old[key], torch.zeros(old[key].shape[0], 3)], dim=1)
        elif key in old:
            new[key] = old[key]
    agent.net.load_state_dict(new)
    # 微调要平衡探索噪声：σ 太大（如 0.6）噪声让步态颠簸摔倒，"站着不动"
    # 反而更优，会把会走的策略洗没（实测翻车 v2）；太小则学新行为慢。
    # 取 σ≈0.25：保住预训练步态，留一点探索学指令
    agent.net.log_std.data.fill_(-1.4)
    print(
        f"已加载预训练运动策略 {ckpt_path}（输入层 {obs_body_dim}→30 维，"
        "指令列置零，σ 重置为 0.25）"
    )


def eval_commands(agent, env, episodes_per_cmd=2):
    """用固定指令评估：每条指令下实际速度和指令的差距"""
    test_cmds = {
        "前进 vx=0.8": (0.8, 0.0, 0.0),
        "后退 vx=-0.4": (-0.4, 0.0, 0.0),
        "左移 vy=0.4": (0.0, 0.4, 0.0),
        "转向 ω=0.8": (0.0, 0.0, 0.8),
        "站住 0": (0.0, 0.0, 0.0),
    }
    print("\n===== 指令跟随考试 =====")
    print(f"{'指令':<14}{'实际 vx':>9}{'实际 vy':>9}{'实际 ω':>9}")
    for name, cmd in test_cmds.items():
        vxs, vys, wys = [], [], []
        for _ in range(episodes_per_cmd):
            state, _ = env.reset()
            env.set_command(*cmd)
            for _ in range(400):  # 每条指令跑 20 秒取平均
                state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    mean, _, _ = agent.net(state_t)
                action = np.clip(
                    mean.squeeze(0).cpu().numpy(),
                    env.action_space.low,
                    env.action_space.high,
                )
                state, _, terminated, truncated, info = env.step(action)
                vxs.append(info["x_velocity"])
                vys.append(info["y_velocity"])
                wys.append(float(env.unwrapped.data.qvel[5]))
                if terminated or truncated:
                    break
        print(f"{name:<14}{np.mean(vxs):>9.2f}{np.mean(vys):>9.2f}{np.mean(wys):>9.2f}")


def record_demo_video(agent, env_id, env_kwargs, out_path):
    """
    录一段"指令跟随演示"视频：按剧本切换指令，画面左上角显示当前指令
    """
    env = gym.make(env_id, render_mode="rgb_array", camera_name="track", **env_kwargs)
    env = VelocityCommandWrapper(env)

    # 剧本：(指令, 持续步数) —— 每步 0.05s，60 步 = 3 秒
    script = [
        ("前进 vx=0.8", (0.8, 0.0, 0.0), 60),
        ("左移 vy=0.5", (0.0, 0.5, 0.0), 60),
        ("转向 ω=1.0", (0.0, 0.0, 1.0), 60),
        ("站住", (0.0, 0.0, 0.0), 40),
        ("后退 vx=-0.5", (-0.5, 0.0, 0.0), 60),
    ]

    frames = []
    state, _ = env.reset()
    for label, cmd, steps in script:
        env.set_command(*cmd)
        for _ in range(steps):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                mean, _, _ = agent.net(state_t)
            action = np.clip(
                mean.squeeze(0).cpu().numpy(),
                env.action_space.low,
                env.action_space.high,
            )
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
                env.set_command(*cmd)
            # 把当前指令画在画面左上角
            frame = env.render()
            img = Image.fromarray(frame)
            ImageDraw.Draw(img).text((10, 10), f"CMD: {label}", fill=(255, 60, 60))
            frames.append(np.asarray(img))
    env.close()
    imageio.mimsave(out_path, frames, fps=20)
    print(f"指令跟随演示视频已保存到 {out_path}（{len(frames)} 帧）")


def play_keyboard(agent, env_id, env_kwargs):
    """
    键盘遥控模式：弹出 MuJoCo 实时窗口，**方向键即时遥控**（不用回车）

    控制: ↑=前进 ↓=后退 ←/→=原地左转/右转 a/d=左移/右移 空格或x=站住 Ctrl+C=退出

    实现原理：终端设成 raw 模式逐键读取（方向键是 ESC[A 等转义序列），
    select 非阻塞检查按键——蚂蚁持续运动，你按一下它就换一个指令，
    这就是真实机器狗遥控器摇杆的角色。
    """
    import select
    import termios
    import tty

    keymap = {
        "up": (0.8, 0.0, 0.0, "前进"),
        "down": (-0.5, 0.0, 0.0, "后退"),
        "left": (0.0, 0.0, 0.8, "左转"),
        "right": (0.0, 0.0, -0.8, "右转"),
        "a": (0.0, 0.5, 0.0, "左移"),
        "d": (0.0, -0.5, 0.0, "右移"),
        "x": (0.0, 0.0, 0.0, "站住"),
        " ": (0.0, 0.0, 0.0, "站住"),
    }

    env = gym.make(env_id, render_mode="human", **env_kwargs)
    env = VelocityCommandWrapper(env)
    state, _ = env.reset()
    env.set_command(0.0, 0.0, 0.0)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    print("\n===== 键盘遥控模式（MuJoCo 窗口里看蚂蚁）=====")
    print("↑=前进 ↓=后退 ←/→=转向 a/d=侧移 空格=站住 Ctrl+C=退出")
    try:
        tty.setraw(fd)  # 逐键读取，不用回车
        while True:
            # 非阻塞检查按键
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x03":  # Ctrl+C
                    break
                if ch == "\x1b":  # 方向键的转义序列：ESC [ A/B/C/D
                    seq = sys.stdin.read(2)
                    key = {
                        "[A": "up",
                        "[B": "down",
                        "[C": "right",
                        "[D": "left",
                    }.get(seq)
                else:
                    key = ch
                if key in keymap:
                    vx, vy, wyaw, label = keymap[key]
                    env.set_command(vx, vy, wyaw)
                    print(f"指令 → {label}")

            # 策略推进一步（部署态：取均值不采样）
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                mean, _, _ = agent.net(state_t)
            action = np.clip(
                mean.squeeze(0).cpu().numpy(),
                env.action_space.low,
                env.action_space.high,
            )
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
            time.sleep(0.05)  # 物理步频 20Hz，保持真实速度
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        env.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="指令跟随版 Ant（速度跟踪 PPO）")
    parser.add_argument("--play", action="store_true", help="键盘遥控已训好的策略")
    parser.add_argument("--steps", type=int, default=800_000, help="总采样步数")
    parser.add_argument(
        "--init-from",
        default="examples/07_ant_policy.pt",
        help="预训练运动策略权重（默认用 07 训好的蚂蚁；传空字符串则从零训练）。"
        "先会走、再学听指令：从零训跟踪任务会因'动会摔、站安全'陷入躺平",
    )
    args = parser.parse_args()

    ENV_ID = "Ant-v5"
    slug = "ant_cmd"
    policy_path = f"examples/08_{slug}_policy.pt"
    # 关掉接触力观测（同 07）；并关闭"摔倒即终止" —— 终止对策略是巨额惩罚，
    # 探索噪声下"跑起来会摔死" vs "站着永安全"会让策略宁可站住（实测翻车 v3）。
    # 不终止后摔倒只是少拿分，策略才敢放心练跑
    env_kwargs = {
        "include_cfrc_ext_in_observation": False,
        "terminate_when_unhealthy": False,
    }

    if args.play:
        # 键盘遥控模式：加载已训练的权重
        env = VelocityCommandWrapper(gym.make(ENV_ID, **env_kwargs))
        agent = PPOContinuousAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        )
        agent.net.load_state_dict(torch.load(policy_path, weights_only=True))
        env.close()
        play_keyboard(agent, ENV_ID, env_kwargs)
        return

    print("=" * 55)
    print("指令跟随版 Ant：速度指令跟踪 + PPO")
    print("=" * 55)

    # ===== 创建环境（观测 27+3=30 维）=====
    env = VelocityCommandWrapper(gym.make(ENV_ID, **env_kwargs))
    print(f"\n环境: {ENV_ID} + 速度指令包装")
    print(f"  观测维度: {env.observation_space.shape[0]}（27 本体 + 3 指令）")
    print(f"  动作维度: {env.action_space.shape[0]}")
    print("  指令: 每 5 秒随机换 (vx*, vy*, ω*)，15% 概率站住")

    # ===== 训练（算法和 07 完全一样，core/ppo_continuous.py 原样复用）=====
    agent = PPOContinuousAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dim=128,
        lr=3e-4,
    )
    if args.init_from:
        if Path(args.init_from).exists():
            load_pretrained_locomotion(agent, args.init_from)
            # Critic 预热：先冻结 Actor 练 20 次更新（约 16 万步），
            # 让 V(s) 学会"指令运气"的价值，否则优势被指令噪声主导、
            # 预训练的步态会在前几十次更新里被乱梯度摧毁（实测教训）
            agent.actor_warmup_updates = 20
        else:
            print(f"警告：预训练权重 {args.init_from} 不存在，从零开始训练")
            print("（从零训跟踪任务很难：建议先跑 examples/07_ppo_continuous_ant.py）")

    # 关键超参：batch 远大于 07（2048 → 8192）。
    # 指令跟踪的条件信号（"当前指令下这个动作好不好"）在小批次里方差极大——
    # 每批 2048 步只含约 20 个不同指令，网络分不清该听哪个；
    # 每批 8192 步含约 80 个指令，条件信号才稳定（实测大批次是有效关键）
    batch_size = 8192
    n_updates = args.steps // batch_size
    print(f"\n开始训练（总步数 {args.steps:,}，约 {n_updates} 次更新）...\n")

    def on_update_end(idx, mean_reward, mean_steps, loss, global_step):
        if idx % 10 == 0:
            print(
                f"  更新 {idx:4d}/{n_updates}  步数 {global_step:>9,}  "
                f"批内平均奖励: {mean_reward:7.2f}"
            )

    update_rewards, update_steps, total_losses, actor_losses, critic_losses = (
        train_ppo_continuous(
            env,
            agent,
            total_steps=args.steps,
            batch_size=batch_size,
            on_update_end=on_update_end,
        )
    )
    print("训练完成！")

    # ===== 考试：固定指令下的跟踪精度 =====
    eval_commands(agent, env)
    env.close()

    # ===== 保存 + 录像 + 画图 =====
    torch.save(agent.net.state_dict(), policy_path)
    print(f"\n策略权重已保存到 {policy_path}")
    record_demo_video(agent, ENV_ID, env_kwargs, f"examples/08_{slug}_demo.mp4")
    plot_results(
        update_rewards,
        update_steps,
        actor_losses,
        critic_losses,
        out_path=f"examples/08_{slug}_result.png",
    )

    print("\n" + "=" * 55)
    print("学到了什么？")
    print("=" * 55)
    print("""
1. 指令条件化策略：把"指令"拼进观测，策略学会"看指令行事"
   —— 真实机器狗（宇树 Go2 等）的遥控器底层就是这个
2. 奖励改成跟踪误差：exp(-误差²) 越接近指令分越高，不再是无脑往前
3. 训练时随机换指令（每 5 秒）= 对"指令"做域随机化，见够各种指令
4. 算法零改动：core/ppo_continuous.py 原样复用 —— 变的是任务不是算法

部署方式：
  - 键盘遥控: uv run python examples/08_ppo_command_ant.py --play
  - 程序接管: 把 set_command 的输入换成导航算法的输出（分层架构！）

承上启下：
  - 这个"速度指令接口"就是机器人大脑/小脑分层的接缝：
    小脑（本策略）管怎么走稳，大脑（导航/寻迹/VLA）管往哪走
  - 下一步：MJCF 给自己的 3D 打印机器人建模，把这套流程搬到它身上
""")


if __name__ == "__main__":
    main()
