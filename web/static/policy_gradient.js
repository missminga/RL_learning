// REINFORCE Policy Gradient 前端逻辑

let pgRewardChart = null;
let pgLossChart = null;

async function runPolicyGradient() {
    const btn = document.getElementById('pg-run-btn');
    const statusText = document.getElementById('pg-status-text');

    const episodes = parseInt(document.getElementById('pg-episodes').value);
    const hidden_dim = parseInt(document.getElementById('pg-hidden').value);
    const lr = parseFloat(document.getElementById('pg-lr').value);
    const gamma = parseFloat(document.getElementById('pg-gamma').value);
    const n_runs = parseInt(document.getElementById('pg-nruns').value);

    btn.disabled = true;
    statusText.textContent = '训练中，REINFORCE 需要跑完整个回合才更新，请耐心等待...';
    statusText.style.color = '#666';

    try {
        const resp = await fetch('/api/policy-gradient/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ episodes, hidden_dim, lr, gamma, n_runs })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || JSON.stringify(err));
        }

        const data = await resp.json();
        renderPGCharts(data);
        renderPGSummary(data);

        statusText.textContent = '训练完成！';
        statusText.style.color = '#38a169';
    } catch (e) {
        statusText.textContent = '出错: ' + e.message;
        statusText.style.color = '#e53e3e';
    } finally {
        btn.disabled = false;
    }
}

function renderPGCharts(data) {
    const chartsPanel = document.getElementById('pg-charts-panel');
    chartsPanel.style.display = 'block';

    if (!pgRewardChart) {
        pgRewardChart = echarts.init(document.getElementById('pg-reward-chart'));
    }
    if (!pgLossChart) {
        pgLossChart = echarts.init(document.getElementById('pg-loss-chart'));
    }

    const xData = Array.from({ length: data.avg_rewards.length }, (_, i) => i + 1);
    const window = 20;
    const smoothRewards = movingAverage(data.avg_rewards, window);
    const smoothLosses = movingAverage(data.avg_losses, window);

    // 奖励图
    pgRewardChart.setOption({
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
    pgLossChart.setOption({
        title: { text: '策略梯度损失', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '回合' },
        yAxis: { type: 'value', name: '损失' },
        grid: { left: 60, right: 20, bottom: 60, top: 50 },
        series: [
            {
                name: '每回合损失',
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

function renderPGSummary(data) {
    const panel = document.getElementById('pg-summary-panel');
    panel.style.display = 'block';

    const content = document.getElementById('pg-summary-content');
    const s = data.summary;

    const solvedText = s.solved_episode !== null
        ? `<span style="color:#38a169;font-weight:bold;">已解决！（第 ${s.solved_episode} 回合）</span>`
        : '<span style="color:#e53e3e;">未解决（REINFORCE 方差较大，可多跑几次或增加回合数）</span>';

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
    if (pgRewardChart) pgRewardChart.resize();
    if (pgLossChart) pgLossChart.resize();
});
