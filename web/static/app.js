// 多臂老虎机实验前端逻辑

// ECharts 颜色方案
const COLORS = [
    '#5470c6', '#91cc75', '#fac858', '#ee6666',
    '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
    '#ea7ccc', '#48b8d0'
];

let rewardChart = null;
let optimalChart = null;

async function runExperiment() {
    const btn = document.getElementById('run-btn');
    const statusText = document.getElementById('status-text');

    // 解析参数
    const epsilonsStr = document.getElementById('epsilons').value;
    const epsilons = epsilonsStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
    const steps = parseInt(document.getElementById('steps').value);
    const n_runs = parseInt(document.getElementById('n_runs').value);
    const k = parseInt(document.getElementById('k').value);

    // 简单校验
    if (epsilons.length === 0) {
        statusText.textContent = '请输入至少一个 ε 值';
        statusText.style.color = '#e53e3e';
        return;
    }

    // 禁用按钮，显示状态
    btn.disabled = true;
    statusText.textContent = '计算中，请稍候...';
    statusText.style.color = '#666';

    try {
        const resp = await fetch('/api/bandit/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epsilons, steps, n_runs, k })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || JSON.stringify(err));
        }

        const data = await resp.json();
        renderCharts(data);
        renderSummary(data.summary);

        statusText.textContent = '完成！';
        statusText.style.color = '#38a169';
    } catch (e) {
        statusText.textContent = '出错: ' + e.message;
        statusText.style.color = '#e53e3e';
    } finally {
        btn.disabled = false;
    }
}

function renderCharts(data) {
    const chartsPanel = document.getElementById('charts-panel');
    chartsPanel.style.display = 'block';

    // 初始化或获取 ECharts 实例
    if (!rewardChart) {
        rewardChart = echarts.init(document.getElementById('reward-chart'));
    }
    if (!optimalChart) {
        optimalChart = echarts.init(document.getElementById('optimal-chart'));
    }

    const epsilons = data.epsilons;
    const xData = Array.from({ length: Object.values(data.rewards)[0].length }, (_, i) => i + 1);

    // 平均奖励图表
    const rewardSeries = epsilons.map((eps, idx) => ({
        name: eps === 0 ? 'ε = 0 (纯贪心)' : `ε = ${eps}`,
        type: 'line',
        showSymbol: false,
        data: data.rewards[String(eps)],
        lineStyle: { width: 2 },
        itemStyle: { color: COLORS[idx % COLORS.length] }
    }));

    rewardChart.setOption({
        title: { text: '不同 ε 值的平均奖励对比', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '步数', axisLabel: { interval: 'auto' } },
        yAxis: { type: 'value', name: '平均奖励' },
        dataZoom: [
            { type: 'inside', xAxisIndex: 0 },
            { type: 'slider', xAxisIndex: 0, bottom: 30 }
        ],
        grid: { left: 60, right: 20, bottom: 80, top: 50 },
        series: rewardSeries
    }, true);

    // 最优动作比例图表
    const optimalSeries = epsilons.map((eps, idx) => ({
        name: eps === 0 ? 'ε = 0 (纯贪心)' : `ε = ${eps}`,
        type: 'line',
        showSymbol: false,
        data: data.optimal_pct[String(eps)],
        lineStyle: { width: 2 },
        itemStyle: { color: COLORS[idx % COLORS.length] }
    }));

    optimalChart.setOption({
        title: { text: '不同 ε 值选择最优动作的比例', left: 'center' },
        tooltip: { trigger: 'axis', valueFormatter: v => v.toFixed(1) + '%' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '步数', axisLabel: { interval: 'auto' } },
        yAxis: { type: 'value', name: '最优动作比例 (%)', max: 100 },
        dataZoom: [
            { type: 'inside', xAxisIndex: 0 },
            { type: 'slider', xAxisIndex: 0, bottom: 30 }
        ],
        grid: { left: 60, right: 20, bottom: 80, top: 50 },
        series: optimalSeries
    }, true);
}

function renderSummary(summary) {
    const panel = document.getElementById('summary-panel');
    panel.style.display = 'block';

    const tbody = document.querySelector('#summary-table tbody');
    tbody.innerHTML = '';

    for (const item of summary) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${item.epsilon}</td>
            <td>${item.avg_reward.toFixed(3)}</td>
            <td>${item.optimal_pct.toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
    }
}

// 窗口缩放时自适应图表大小
window.addEventListener('resize', () => {
    if (rewardChart) rewardChart.resize();
    if (optimalChart) optimalChart.resize();
});
