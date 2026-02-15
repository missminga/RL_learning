// DQN CartPole 前端逻辑

let cpRewardChart = null;
let cpLossChart = null;

async function runCartPole() {
    const btn = document.getElementById('cp-run-btn');
    const statusText = document.getElementById('cp-status-text');

    const episodes = parseInt(document.getElementById('cp-episodes').value);
    const hidden_dim = parseInt(document.getElementById('cp-hidden').value);
    const lr = parseFloat(document.getElementById('cp-lr').value);
    const gamma = parseFloat(document.getElementById('cp-gamma').value);
    const epsilon = parseFloat(document.getElementById('cp-epsilon').value);
    const epsilon_decay = parseFloat(document.getElementById('cp-decay').value);
    const buffer_capacity = parseInt(document.getElementById('cp-buffer').value);
    const batch_size = parseInt(document.getElementById('cp-batch').value);
    const target_update_freq = parseInt(document.getElementById('cp-target-freq').value);

    btn.disabled = true;
    statusText.textContent = '训练中，DQN 需要较长时间，请耐心等待...';
    statusText.style.color = '#666';

    try {
        const resp = await fetch('/api/cartpole/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                episodes, hidden_dim, lr, gamma,
                epsilon, epsilon_decay,
                epsilon_min: 0.01,
                buffer_capacity, batch_size,
                target_update_freq,
                n_runs: 1
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || JSON.stringify(err));
        }

        const data = await resp.json();
        renderCPCharts(data);
        renderCPSummary(data);

        statusText.textContent = '训练完成！';
        statusText.style.color = '#38a169';
    } catch (e) {
        statusText.textContent = '出错: ' + e.message;
        statusText.style.color = '#e53e3e';
    } finally {
        btn.disabled = false;
    }
}

function renderCPCharts(data) {
    const chartsPanel = document.getElementById('cp-charts-panel');
    chartsPanel.style.display = 'block';

    if (!cpRewardChart) {
        cpRewardChart = echarts.init(document.getElementById('cp-reward-chart'));
    }
    if (!cpLossChart) {
        cpLossChart = echarts.init(document.getElementById('cp-loss-chart'));
    }

    const xData = Array.from({ length: data.avg_rewards.length }, (_, i) => i + 1);
    const window = 20;
    const smoothRewards = movingAverage(data.avg_rewards, window);
    const smoothLosses = movingAverage(data.avg_losses, window);

    // 奖励图
    cpRewardChart.setOption({
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

    // 损失图
    cpLossChart.setOption({
        title: { text: '训练损失 (MSE)', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '回合' },
        yAxis: { type: 'value', name: '损失' },
        grid: { left: 60, right: 20, bottom: 60, top: 50 },
        series: [
            {
                name: '每回合平均损失',
                type: 'line',
                showSymbol: false,
                data: data.avg_losses,
                lineStyle: { width: 1, opacity: 0.3 },
                itemStyle: { color: '#fc8452' }
            },
            {
                name: '滑动平均',
                type: 'line',
                showSymbol: false,
                data: smoothLosses,
                lineStyle: { width: 2 },
                itemStyle: { color: '#9a60b4' }
            }
        ]
    }, true);
}

function renderCPSummary(data) {
    const panel = document.getElementById('cp-summary-panel');
    panel.style.display = 'block';

    const content = document.getElementById('cp-summary-content');
    const s = data.summary;

    const solvedText = s.solved_episode !== null
        ? `<span style="color:#38a169;font-weight:bold;">已解决！（第 ${s.solved_episode} 回合）</span>`
        : '<span style="color:#e53e3e;">未解决（可增加训练回合数）</span>';

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
    if (cpRewardChart) cpRewardChart.resize();
    if (cpLossChart) cpLossChart.resize();
});
