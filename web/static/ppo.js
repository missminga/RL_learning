// PPO (近端策略优化) 前端逻辑

let ppoRewardChart = null;
let ppoLossChart = null;

async function runPPO() {
    const btn = document.getElementById('ppo-run-btn');
    const statusText = document.getElementById('ppo-status-text');

    const episodes = parseInt(document.getElementById('ppo-episodes').value);
    const hidden_dim = parseInt(document.getElementById('ppo-hidden').value);
    const lr = parseFloat(document.getElementById('ppo-lr').value);
    const gamma = parseFloat(document.getElementById('ppo-gamma').value);
    const gae_lambda = parseFloat(document.getElementById('ppo-gae-lambda').value);
    const clip_eps = parseFloat(document.getElementById('ppo-clip-eps').value);
    const update_epochs = parseInt(document.getElementById('ppo-update-epochs').value);
    const minibatch_size = parseInt(document.getElementById('ppo-minibatch').value);
    const value_coef = parseFloat(document.getElementById('ppo-value-coef').value);
    const entropy_coef = parseFloat(document.getElementById('ppo-entropy-coef').value);
    const n_runs = parseInt(document.getElementById('ppo-nruns').value);

    btn.disabled = true;
    statusText.textContent = '训练中，PPO 在裁剪保护下反复复用每批数据，请耐心等待...';
    statusText.style.color = '#666';

    try {
        // 1) 提交异步训练任务
        const resp = await fetch('/api/ppo/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                episodes, hidden_dim, lr, gamma, gae_lambda, clip_eps,
                update_epochs, minibatch_size, value_coef, entropy_coef, n_runs
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || JSON.stringify(err));
        }

        const { task_id } = await resp.json();

        // 2) 轮询任务状态，直到完成
        const data = await pollTask(task_id, statusText);

        renderPPOCharts(data);
        renderPPOSummary(data);

        statusText.textContent = '训练完成！';
        statusText.style.color = '#38a169';
    } catch (e) {
        statusText.textContent = '出错: ' + e.message;
        statusText.style.color = '#e53e3e';
    } finally {
        btn.disabled = false;
    }
}

// 轮询任务状态，返回最终结果
async function pollTask(taskId, statusText) {
    while (true) {
        await new Promise((r) => setTimeout(r, 800));
        const resp = await fetch(`/api/tasks/${taskId}`);
        if (!resp.ok) {
            throw new Error('查询任务状态失败');
        }
        const task = await resp.json();

        if (task.status === 'running' || task.status === 'queued') {
            const pct = Math.round((task.progress || 0) * 100);
            statusText.textContent = `训练中... ${pct}%  ${task.message || ''}`;
            continue;
        }
        if (task.status === 'completed') {
            return task.result;
        }
        // failed / cancelled / timeout
        throw new Error(task.error || `任务${task.status}`);
    }
}

function renderPPOCharts(data) {
    const chartsPanel = document.getElementById('ppo-charts-panel');
    chartsPanel.style.display = 'block';

    if (!ppoRewardChart) {
        ppoRewardChart = echarts.init(document.getElementById('ppo-reward-chart'));
    }
    if (!ppoLossChart) {
        ppoLossChart = echarts.init(document.getElementById('ppo-loss-chart'));
    }

    const xData = Array.from({ length: data.avg_rewards.length }, (_, i) => i + 1);
    const window = 20;
    const smoothRewards = movingAverage(data.avg_rewards, window);
    const smoothActor = movingAverage(data.avg_actor_losses, window);
    const smoothCritic = movingAverage(data.avg_critic_losses, window);

    // 奖励图
    ppoRewardChart.setOption({
        title: { text: '每回合奖励（存活步数）', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '回合' },
        yAxis: { type: 'value', name: '奖励' },
        grid: { left: 60, right: 20, bottom: 60, top: 50 },
        series: [
            {
                name: '每回合奖励',
                type: 'line',
                showSymbol: false,
                data: data.avg_rewards,
                lineStyle: { width: 1, opacity: 0.3 },
                itemStyle: { color: '#5470c6' }
            },
            {
                name: '滑动平均',
                type: 'line',
                showSymbol: false,
                data: smoothRewards,
                lineStyle: { width: 2 },
                itemStyle: { color: '#ee6666' }
            },
            {
                name: '解决标准 (475)',
                type: 'line',
                showSymbol: false,
                data: xData.map(() => 475),
                lineStyle: { width: 1, type: 'dashed', color: '#38a169' },
                itemStyle: { color: '#38a169' }
            }
        ]
    }, true);

    // 损失图：同时画 Actor（裁剪策略）和 Critic（评委）两条损失
    ppoLossChart.setOption({
        title: { text: 'Actor / Critic 损失', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '回合' },
        yAxis: { type: 'value', name: '损失' },
        grid: { left: 60, right: 20, bottom: 60, top: 50 },
        series: [
            {
                name: 'Actor 损失（滑动平均）',
                type: 'line',
                showSymbol: false,
                data: smoothActor,
                lineStyle: { width: 2 },
                itemStyle: { color: '#fc8452' }
            },
            {
                name: 'Critic 损失（滑动平均）',
                type: 'line',
                showSymbol: false,
                data: smoothCritic,
                lineStyle: { width: 2 },
                itemStyle: { color: '#73c0de' }
            }
        ]
    }, true);
}

function renderPPOSummary(data) {
    const panel = document.getElementById('ppo-summary-panel');
    panel.style.display = 'block';

    const content = document.getElementById('ppo-summary-content');
    const s = data.summary;

    const solvedText = s.solved_episode !== null
        ? `<span style="color:#38a169;font-weight:bold;">已解决！（第 ${s.solved_episode} 回合）</span>`
        : '<span style="color:#e53e3e;">未解决（可增加训练回合数或多跑几次）</span>';

    content.innerHTML = `
        <div class="summary-stats">
            <div class="stat-item">
                <span class="stat-value">${s.final_avg_reward}</span>
                <span class="stat-label">最后50回合平均奖励</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${s.final_avg_steps}</span>
                <span class="stat-label">最后50回合平均步数</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${s.max_reward}</span>
                <span class="stat-label">单回合最高奖励</span>
            </div>
        </div>
        <p style="text-align:center;margin-top:16px;font-size:1rem;">
            CartPole 状态：${solvedText}
        </p>
    `;
}

// 滑动平均
function movingAverage(arr, window) {
    const result = new Array(arr.length).fill(null);
    for (let i = window - 1; i < arr.length; i++) {
        let sum = 0;
        for (let j = i - window + 1; j <= i; j++) {
            sum += arr[j];
        }
        result[i] = sum / window;
    }
    return result;
}

// 窗口缩放自适应
window.addEventListener('resize', () => {
    if (ppoRewardChart) ppoRewardChart.resize();
    if (ppoLossChart) ppoLossChart.resize();
});
