"""DQN 模块单元测试"""

import numpy as np
import pytest
import torch
import gymnasium as gym

from core.dqn import (
    QNetwork,
    ReplayBuffer,
    DQNAgent,
    train_dqn,
    run_cartpole_experiment,
)


# ===== QNetwork 测试 =====


class TestQNetwork:
    """Q 网络基础测试"""

    def test_output_shape(self):
        """输出维度应等于动作数"""
        net = QNetwork(state_dim=4, action_dim=2, hidden_dim=32)
        x = torch.randn(1, 4)
        out = net(x)
        assert out.shape == (1, 2)

    def test_batch_output(self):
        """批量输入应返回对应批量输出"""
        net = QNetwork(state_dim=4, action_dim=3, hidden_dim=16)
        x = torch.randn(8, 4)
        out = net(x)
        assert out.shape == (8, 3)

    def test_different_inputs_different_outputs(self):
        """不同输入应产生不同输出（网络不是常函数）"""
        net = QNetwork(state_dim=4, action_dim=2, hidden_dim=32)
        x1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        x2 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        out1 = net(x1)
        out2 = net(x2)
        assert not torch.allclose(out1, out2)


# ===== ReplayBuffer 测试 =====


class TestReplayBuffer:
    """经验回放缓冲区测试"""

    def test_push_and_len(self):
        """存入经验后长度应增加"""
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.push([1, 2, 3, 4], 0, 1.0, [2, 3, 4, 5], False)
        assert len(buf) == 1

    def test_capacity_limit(self):
        """超出容量时应自动丢弃最旧的"""
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.push([i, 0, 0, 0], 0, 1.0, [i + 1, 0, 0, 0], False)
        assert len(buf) == 5

    def test_sample_shape(self):
        """抽样返回的形状应正确"""
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push([i, i, i, i], i % 2, 1.0, [i + 1, i, i, i], i == 19)
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, 4)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 4)
        assert dones.shape == (8,)

    def test_sample_types(self):
        """抽样返回的数据类型应正确"""
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push([0, 0, 0, 0], 1, 0.5, [1, 1, 1, 1], True)
        states, actions, rewards, next_states, dones = buf.sample(4)
        assert states.dtype == np.float32
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32
        assert dones.dtype == np.float32


# ===== DQNAgent 测试 =====


class TestDQNAgent:
    """DQN 智能体测试"""

    @pytest.fixture
    def agent(self):
        return DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=32,
            lr=1e-3, gamma=0.99, epsilon=1.0,
            epsilon_decay=0.99, epsilon_min=0.01,
            buffer_capacity=100, batch_size=8,
            target_update_freq=5,
        )

    def test_select_action_range(self, agent):
        """选出的动作应在有效范围内"""
        state = np.array([0.0, 0.1, -0.05, 0.2], dtype=np.float32)
        for _ in range(20):
            action = agent.select_action(state)
            assert action in [0, 1]

    def test_select_action_greedy(self, agent):
        """epsilon=0 时应完全贪心"""
        agent.epsilon = 0.0
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        actions = [agent.select_action(state) for _ in range(10)]
        # 贪心时对同一状态应始终选同一个动作
        assert len(set(actions)) == 1

    def test_epsilon_decay(self, agent):
        """epsilon 应正确衰减"""
        initial = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(initial * agent.epsilon_decay)

    def test_epsilon_min(self, agent):
        """epsilon 不应低于最小值"""
        agent.epsilon = 0.011
        agent.epsilon_min = 0.01
        agent.epsilon_decay = 0.5
        agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_min

    def test_train_step_no_data(self, agent):
        """缓冲区不够时 train_step 应返回 None"""
        result = agent.train_step()
        assert result is None

    def test_train_step_with_data(self, agent):
        """缓冲区够了之后 train_step 应返回损失值"""
        for i in range(20):
            agent.store_transition(
                np.random.randn(4).astype(np.float32), i % 2,
                1.0, np.random.randn(4).astype(np.float32), False,
            )
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

    def test_target_network_update(self, agent):
        """更新目标网络后两个网络参数应相同"""
        # 先训练几步让 policy_net 参数变化
        for i in range(20):
            agent.store_transition(
                np.random.randn(4).astype(np.float32), i % 2,
                1.0, np.random.randn(4).astype(np.float32), False,
            )
        agent.train_step()

        agent.update_target_network()

        for p, t in zip(agent.policy_net.parameters(),
                        agent.target_net.parameters()):
            assert torch.allclose(p, t)


# ===== 训练集成测试 =====


class TestTraining:
    """训练流程集成测试"""

    def test_train_dqn_runs(self):
        """train_dqn 应能正常运行并返回正确长度的结果"""
        env = gym.make("CartPole-v1")
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=16,
            buffer_capacity=200, batch_size=16,
            target_update_freq=3,
        )
        rewards, steps, losses = train_dqn(
            env, agent, episodes=5, max_steps=50,
        )
        env.close()

        assert len(rewards) == 5
        assert len(steps) == 5
        assert len(losses) == 5
        assert all(r >= 0 for r in rewards)
        assert all(s >= 1 for s in steps)

    def test_run_cartpole_experiment(self):
        """run_cartpole_experiment 应返回正确结构的 dict"""
        result = run_cartpole_experiment(
            episodes=5, hidden_dim=16, buffer_capacity=200,
            batch_size=16, target_update_freq=2, max_steps=50,
        )

        assert "episodes" in result
        assert "avg_rewards" in result
        assert "avg_steps" in result
        assert "avg_losses" in result
        assert "summary" in result
        assert len(result["avg_rewards"]) == 5
        assert "final_avg_reward" in result["summary"]
        assert "max_reward" in result["summary"]

    def test_callback_called(self):
        """on_episode_end 回调应被正确调用"""
        env = gym.make("CartPole-v1")
        agent = DQNAgent(
            state_dim=4, action_dim=2, hidden_dim=16,
            buffer_capacity=200, batch_size=16,
        )
        call_count = [0]

        def callback(ep, reward, steps, eps):
            call_count[0] += 1

        train_dqn(env, agent, episodes=3, max_steps=20,
                   on_episode_end=callback)
        env.close()

        assert call_count[0] == 3
