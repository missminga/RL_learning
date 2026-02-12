"""
Q-Learning + GridWorld 核心算法模块

提供 GridWorld 环境、Q-Learning 训练函数。
CLI 脚本和 Web API 共用此模块。
"""

import numpy as np

# 动作定义：上、下、左、右
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class GridWorld:
    """
    GridWorld 网格世界环境

    一个 rows × cols 的网格，智能体从起点出发，目标是到达终点。
    途中可能有陷阱（负奖励）和墙壁（不可通行）。

    关键概念：
    - 状态 (State): 智能体在网格中的位置 (row, col)
    - 动作 (Action): 上(0)、下(1)、左(2)、右(3)
    - 状态转移: 执行动作后移动到新位置（撞墙则原地不动）
    - 奖励 (Reward): 到达终点 +10，掉进陷阱 -10，每走一步 -0.1
    """

    def __init__(self, rows=5, cols=5, start=None, goal=None,
                 traps=None, walls=None):
        """
        初始化网格世界

        参数:
            rows: 网格行数
            cols: 网格列数
            start: 起点坐标 (row, col)，默认左上角
            goal: 终点坐标 (row, col)，默认右下角
            traps: 陷阱列表 [(row, col), ...]
            walls: 墙壁列表 [(row, col), ...]
        """
        self.rows = rows
        self.cols = cols
        self.start = start or (0, 0)
        self.goal = goal or (rows - 1, cols - 1)
        self.traps = set(traps or [])
        self.walls = set(walls or [])
        self.state = self.start

    def reset(self):
        """重置环境，智能体回到起点"""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        执行一个动作

        参数:
            action: 0=上, 1=下, 2=左, 3=右

        返回:
            next_state: 新的位置
            reward: 获得的奖励
            done: 回合是否结束
        """
        dr, dc = ACTIONS[action]
        new_row = self.state[0] + dr
        new_col = self.state[1] + dc

        # 撞墙检测：出界或碰到墙壁 → 原地不动
        if (new_row < 0 or new_row >= self.rows or
                new_col < 0 or new_col >= self.cols or
                (new_row, new_col) in self.walls):
            new_row, new_col = self.state

        self.state = (new_row, new_col)

        # 判断奖励和是否结束
        if self.state == self.goal:
            return self.state, 10.0, True
        elif self.state in self.traps:
            return self.state, -10.0, True
        else:
            return self.state, -0.1, False

    def state_to_index(self, state):
        """把 (row, col) 转换为整数索引，方便查 Q 表"""
        return state[0] * self.cols + state[1]

    @property
    def n_states(self):
        """状态总数"""
        return self.rows * self.cols

    @property
    def n_actions(self):
        """动作总数"""
        return 4


def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1,
               epsilon_decay=0.995, epsilon_min=0.01, max_steps=100):
    """
    Q-Learning 算法

    === 核心公式 ===
    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    其中：
    - Q(s, a): 在状态 s 执行动作 a 的预期价值
    - α (alpha): 学习率，新经验的权重
    - γ (gamma): 折扣因子，未来奖励的重要程度
    - r: 即时奖励
    - s': 下一个状态
    - max_a' Q(s', a'): 下一个状态的最优价值

    参数:
        env: GridWorld 环境
        episodes: 训练多少个回合
        alpha: 学习率
        gamma: 折扣因子（0~1，越接近 1 越重视未来奖励）
        epsilon: 初始探索概率
        epsilon_decay: 每个回合 ε 衰减的倍率
        epsilon_min: ε 的最小值
        max_steps: 每个回合最多走多少步（防止无限循环）

    返回:
        q_table: 学到的 Q 值表 (n_states × n_actions)
        episode_rewards: 每个回合的总奖励列表
        episode_steps: 每个回合的步数列表
    """
    # 初始化 Q 表：所有 Q 值为 0
    q_table = np.zeros((env.n_states, env.n_actions))
    episode_rewards = []
    episode_steps = []

    current_epsilon = epsilon

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            s_idx = env.state_to_index(state)

            # ε-贪心选动作
            if np.random.random() < current_epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(q_table[s_idx])

            # 执行动作
            next_state, reward, done = env.step(action)
            ns_idx = env.state_to_index(next_state)

            # Q-Learning 更新公式
            best_next = np.max(q_table[ns_idx])
            td_target = reward + gamma * best_next * (1 - done)
            q_table[s_idx, action] += alpha * (td_target - q_table[s_idx, action])

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)

        # 衰减 ε：随着训练进行，逐渐减少探索
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)

    return q_table, episode_rewards, episode_steps


def extract_policy(q_table, env):
    """
    从 Q 表中提取策略（每个状态选 Q 值最大的动作）

    返回:
        policy: rows × cols 的数组，每个格子是最优动作编号
        policy_arrows: rows × cols 的数组，每个格子是箭头符号
    """
    policy = np.zeros((env.rows, env.cols), dtype=int)
    policy_arrows = np.full((env.rows, env.cols), " ", dtype=object)

    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            if pos == env.goal:
                policy_arrows[r, c] = "★"
                continue
            if pos in env.traps:
                policy_arrows[r, c] = "✕"
                continue
            if pos in env.walls:
                policy_arrows[r, c] = "▓"
                continue

            s_idx = env.state_to_index(pos)
            best_action = np.argmax(q_table[s_idx])
            policy[r, c] = best_action
            policy_arrows[r, c] = ACTION_NAMES[best_action]

    return policy, policy_arrows


def run_gridworld_experiment(rows=5, cols=5, traps=None, walls=None,
                             episodes=500, alpha=0.1, gamma=0.99,
                             epsilon=0.1, epsilon_decay=0.995,
                             epsilon_min=0.01, max_steps=100, n_runs=1):
    """
    运行 GridWorld Q-Learning 实验，返回 JSON 可序列化的 dict

    参数:
        rows, cols: 网格大小
        traps: 陷阱坐标列表
        walls: 墙壁坐标列表
        episodes: 训练回合数
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 初始探索率
        epsilon_decay: ε 衰减率
        epsilon_min: 最小 ε
        max_steps: 每回合最大步数
        n_runs: 重复训练次数（取平均）

    返回:
        dict: 包含学习曲线、策略、Q 值等
    """
    if traps is None:
        traps = [(1, 3), (3, 1)]
    if walls is None:
        walls = [(1, 1), (2, 3)]

    # 多次运行取平均
    all_rewards = np.zeros(episodes)
    all_steps = np.zeros(episodes)
    best_q_table = None
    best_final_reward = -float("inf")

    for _ in range(n_runs):
        env = GridWorld(rows, cols, traps=traps, walls=walls)
        q_table, ep_rewards, ep_steps = q_learning(
            env, episodes=episodes, alpha=alpha, gamma=gamma,
            epsilon=epsilon, epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min, max_steps=max_steps,
        )
        all_rewards += np.array(ep_rewards)
        all_steps += np.array(ep_steps)

        final_avg = np.mean(ep_rewards[-50:])
        if final_avg > best_final_reward:
            best_final_reward = final_avg
            best_q_table = q_table.copy()

    avg_rewards = (all_rewards / n_runs).tolist()
    avg_steps = (all_steps / n_runs).tolist()

    # 提取最优策略
    ref_env = GridWorld(rows, cols, traps=traps, walls=walls)
    policy, policy_arrows = extract_policy(best_q_table, ref_env)

    # 构建网格信息
    grid_info = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            pos = (r, c)
            if pos == ref_env.start:
                cell_type = "start"
            elif pos == ref_env.goal:
                cell_type = "goal"
            elif pos in ref_env.traps:
                cell_type = "trap"
            elif pos in ref_env.walls:
                cell_type = "wall"
            else:
                cell_type = "empty"
            row_data.append({
                "type": cell_type,
                "action": int(policy[r, c]),
                "arrow": str(policy_arrows[r, c]),
                "q_values": best_q_table[r * cols + c].tolist(),
            })
        grid_info.append(row_data)

    return {
        "rows": rows,
        "cols": cols,
        "episodes": episodes,
        "avg_rewards": avg_rewards,
        "avg_steps": avg_steps,
        "grid": grid_info,
        "summary": {
            "final_avg_reward": round(float(np.mean(avg_rewards[-50:])), 2),
            "final_avg_steps": round(float(np.mean(avg_steps[-50:])), 1),
            "converged_episode": _find_convergence(avg_rewards),
        },
    }


def _find_convergence(rewards, window=20, threshold=0.5):
    """找到奖励大致收敛的回合数（滑动平均变化小于阈值）"""
    if len(rewards) < window * 2:
        return len(rewards)
    for i in range(window, len(rewards) - window):
        recent = np.mean(rewards[i:i + window])
        if recent > 0:
            return i
    return len(rewards)
