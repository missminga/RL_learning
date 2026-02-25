# Agent Swarm Setup (OpenClaw + Codex/Claude)

这个目录实现了一个可落地的「编排层 + 多代理」最小系统，适配当前仓库。

## 目录结构

- `active-tasks.json`：任务注册表
- `task_registry.py`：任务注册/更新工具
- `spawn-agent.sh`：创建 worktree + 启动 tmux 代理
- `check-agents.py` / `check-agents.sh`：巡检任务状态（tmux/PR/CI）
- `cleanup-worktrees.sh`：清理已完成任务的 worktree
- `logs/`：代理终端日志
- `worktrees/`：每个任务独立 worktree
- `prompts/`：可复用 prompt 模板

## 1) 先决条件

- 已安装并认证：`gh`、`codex`、`claude`
- 已安装：`tmux`
- 当前仓库已配置 git remote（推荐有 `origin`）

## 2) 启动一个代理任务

```bash
cd /root/project/RL_learning

# 方式 A：直接传 prompt 文本
.clawdbot/spawn-agent.sh \
  feat-policy-gradient \
  codex \
  "实现 policy gradient 示例并补测试" \
  "请在 examples 新增 policy gradient 教学示例，中文注释，补 tests 并确保通过 uv run pytest tests/"

# 方式 B：从 prompt 文件读取（推荐）
.clawdbot/spawn-agent.sh \
  feat-dqn-docs \
  claude \
  "补 DQN 教学文档" \
  @/root/project/RL_learning/.clawdbot/prompts/dqn_docs.txt
```

参数说明：

```text
spawn-agent.sh <task-id> <agent:codex|claude> <description> <prompt|@file> [model] [effort]
```

默认模型：
- codex: `gpt-5.3-codex`
- claude: `claude-opus-4.5`

## 3) 巡检（建议 10 分钟一次）

```bash
cd /root/project/RL_learning
.clawdbot/check-agents.sh
```

巡检逻辑：
- 检查 tmux 会话是否仍在
- 查询该分支关联 PR
- 检查 PR checks 是否全部通过
- 自动更新 `active-tasks.json` 的 `status/note/pr` 等字段

## 4) 清理已完成 worktree

```bash
cd /root/project/RL_learning
.clawdbot/cleanup-worktrees.sh
```

## 5) 可选：crontab 示例

```cron
*/10 * * * * cd /root/project/RL_learning && ./.clawdbot/check-agents.sh >> ./.clawdbot/logs/monitor.log 2>&1
30 2 * * * cd /root/project/RL_learning && ./.clawdbot/cleanup-worktrees.sh >> ./.clawdbot/logs/cleanup.log 2>&1
```

## 建议的任务完成定义（DoD）

建议在每个代理 prompt 中明确：

1. PR 已创建
2. CI 通过（lint/types/tests）
3. 如有 UI 变更附截图
4. 关键风险点在 PR 描述中说明

---

这是偏工程实用的 v1。后续可以继续扩展：
- 自动重试策略（失败后按规则追加上下文再重跑）
- Telegram/Discord 推送
- 多模型路由策略（按任务类型自动选 codex/claude）
