"""Q-Learning + GridWorld 核心模块测试"""

import numpy as np
import pytest

from core.q_learning import (
    ACTIONS,
    ACTION_NAMES,
    GridWorld,
    q_learning,
    extract_policy,
    run_gridworld_experiment,
)


class TestGridWorld:
    """GridWorld 环境测试"""

    def test_default_init(self):
        """默认初始化：5x5，起点(0,0)，终点(4,4)"""
        env = GridWorld()
        assert env.rows == 5
        assert env.cols == 5
        assert env.start == (0, 0)
        assert env.goal == (4, 4)
        assert env.n_states == 25
        assert env.n_actions == 4

    def test_reset(self):
        """reset 应该返回起点"""
        env = GridWorld()
        state = env.reset()
        assert state == (0, 0)
        assert env.state == (0, 0)

    def test_move_basic(self):
        """基本移动：向下、向右"""
        env = GridWorld(rows=3, cols=3)
        env.reset()

        # 向下
        state, reward, done = env.step(1)
        assert state == (1, 0)
        assert reward == -0.1
        assert not done

        # 向右
        state, reward, done = env.step(3)
        assert state == (1, 1)
        assert reward == -0.1
        assert not done

    def test_wall_collision(self):
        """撞墙应该原地不动"""
        env = GridWorld(rows=3, cols=3)
        env.reset()  # (0, 0)

        # 向上撞出界
        state, _, _ = env.step(0)
        assert state == (0, 0)

        # 向左撞出界
        state, _, _ = env.step(2)
        assert state == (0, 0)

    def test_wall_block(self):
        """碰到墙壁方块也应该原地不动"""
        env = GridWorld(rows=3, cols=3, walls=[(0, 1)])
        env.reset()  # (0, 0)

        # 向右碰墙壁
        state, _, _ = env.step(3)
        assert state == (0, 0)

    def test_reach_goal(self):
        """到达终点：奖励 +10，done=True"""
        env = GridWorld(rows=2, cols=2)  # 起点(0,0)，终点(1,1)
        env.reset()
        env.step(1)  # (0,0) → (1,0)
        state, reward, done = env.step(3)  # (1,0) → (1,1)
        assert state == (1, 1)
        assert reward == 10.0
        assert done

    def test_trap(self):
        """踩到陷阱：奖励 -10，done=True"""
        env = GridWorld(rows=3, cols=3, traps=[(0, 1)])
        env.reset()
        state, reward, done = env.step(3)  # (0,0) → (0,1) 陷阱
        assert state == (0, 1)
        assert reward == -10.0
        assert done

    def test_state_to_index(self):
        """坐标转索引"""
        env = GridWorld(rows=5, cols=5)
        assert env.state_to_index((0, 0)) == 0
        assert env.state_to_index((0, 4)) == 4
        assert env.state_to_index((1, 0)) == 5
        assert env.state_to_index((4, 4)) == 24


class TestQLearning:
    """Q-Learning 算法测试"""

    def test_q_table_shape(self):
        """Q 表的形状应该是 (n_states, n_actions)"""
        env = GridWorld(rows=3, cols=3)
        q_table, _, _ = q_learning(env, episodes=10)
        assert q_table.shape == (9, 4)

    def test_returns_rewards_and_steps(self):
        """应返回每回合的奖励和步数"""
        env = GridWorld(rows=3, cols=3)
        _, rewards, steps = q_learning(env, episodes=50)
        assert len(rewards) == 50
        assert len(steps) == 50

    def test_learning_improves(self):
        """训练后期的表现应该比初期好"""
        env = GridWorld(rows=4, cols=4, traps=[(1, 3)], walls=[(1, 1)])
        _, rewards, _ = q_learning(
            env, episodes=300, alpha=0.1, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01,
        )
        # 前 30 回合 vs 后 30 回合，后期应该更好
        early = np.mean(rewards[:30])
        late = np.mean(rewards[-30:])
        assert late > early

    def test_simple_2x2_finds_goal(self):
        """2x2 网格，无障碍，应该能学会到达终点"""
        env = GridWorld(rows=2, cols=2)
        q_table, rewards, _ = q_learning(
            env, episodes=200, alpha=0.2, gamma=0.99,
            epsilon=1.0, epsilon_decay=0.98, epsilon_min=0.01,
        )
        # 最后几回合的奖励应该是正的（到达终点得 +10）
        assert np.mean(rewards[-20:]) > 0


class TestExtractPolicy:
    """策略提取测试"""

    def test_policy_shape(self):
        """策略应该是 rows × cols"""
        env = GridWorld(rows=3, cols=3)
        q_table = np.random.randn(9, 4)
        policy, arrows = extract_policy(q_table, env)
        assert policy.shape == (3, 3)
        assert arrows.shape == (3, 3)

    def test_special_cells(self):
        """终点、陷阱、墙壁应该有特殊标记"""
        env = GridWorld(rows=3, cols=3, traps=[(0, 1)], walls=[(1, 0)])
        q_table = np.zeros((9, 4))
        _, arrows = extract_policy(q_table, env)
        assert arrows[2, 2] == "★"   # 终点
        assert arrows[0, 1] == "✕"   # 陷阱
        assert arrows[1, 0] == "▓"   # 墙壁

    def test_arrow_values(self):
        """普通格子应该显示箭头"""
        env = GridWorld(rows=2, cols=2)
        q_table = np.zeros((4, 4))
        _, arrows = extract_policy(q_table, env)
        # (0,0) 是起点，所有 Q 值为 0 时 argmax 返回 0（↑）
        assert arrows[0, 0] in ACTION_NAMES.values()


class TestRunExperiment:
    """运行实验函数测试"""

    def test_returns_correct_structure(self):
        """返回的字典应包含所有必要的键"""
        result = run_gridworld_experiment(
            rows=3, cols=3, traps=[], walls=[],
            episodes=20, n_runs=1,
        )
        assert "rows" in result
        assert "cols" in result
        assert "episodes" in result
        assert "avg_rewards" in result
        assert "avg_steps" in result
        assert "grid" in result
        assert "summary" in result

    def test_grid_dimensions(self):
        """grid 的维度应该匹配 rows × cols"""
        result = run_gridworld_experiment(
            rows=4, cols=3, traps=[], walls=[],
            episodes=10, n_runs=1,
        )
        assert len(result["grid"]) == 4
        assert len(result["grid"][0]) == 3

    def test_multiple_runs(self):
        """多次运行应该返回平均结果"""
        result = run_gridworld_experiment(
            rows=3, cols=3, traps=[], walls=[],
            episodes=20, n_runs=3,
        )
        assert len(result["avg_rewards"]) == 20
        assert len(result["avg_steps"]) == 20
