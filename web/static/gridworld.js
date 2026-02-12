// Q-Learning GridWorld 前端逻辑

let gwRewardChart = null;
let gwStepsChart = null;

// 颜色方案
const CELL_COLORS = {
    start:  '#4f7df9',
    goal:   '#38a169',
    trap:   '#e53e3e',
    wall:   '#4a5568',
    empty:  '#fff'
};

const ARROW_MAP = {
    0: '↑', 1: '↓', 2: '←', 3: '→'
};

async function runGridWorld() {
    const btn = document.getElementById('gw-run-btn');
    const statusText = document.getElementById('gw-status-text');

    const size = parseInt(document.getElementById('grid-size').value);
    const episodes = parseInt(document.getElementById('gw-episodes').value);
    const alpha = parseFloat(document.getElementById('gw-alpha').value);
    const gamma = parseFloat(document.getElementById('gw-gamma').value);
    const epsilon = parseFloat(document.getElementById('gw-epsilon').value);
    const epsilon_decay = parseFloat(document.getElementById('gw-decay').value);

    // 默认陷阱和墙壁（根据网格大小调整）
    const traps = size >= 5 ? [[1,3],[3,1]] : [[1,2]];
    const walls = size >= 5 ? [[1,1],[2,3]] : [[1,1]];

    btn.disabled = true;
    statusText.textContent = '训练中，请稍候...';
    statusText.style.color = '#666';

    try {
        const resp = await fetch('/api/gridworld/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rows: size, cols: size,
                traps, walls,
                episodes, alpha, gamma,
                epsilon, epsilon_decay,
                epsilon_min: 0.01,
                n_runs: 1
            })
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || JSON.stringify(err));
        }

        const data = await resp.json();
        renderGWCharts(data);
        renderPolicyGrid(data);
        renderGWSummary(data);

        statusText.textContent = '训练完成！';
        statusText.style.color = '#38a169';
    } catch (e) {
        statusText.textContent = '出错: ' + e.message;
        statusText.style.color = '#e53e3e';
    } finally {
        btn.disabled = false;
    }
}

function renderGWCharts(data) {
    const chartsPanel = document.getElementById('gw-charts-panel');
    chartsPanel.style.display = 'block';

    if (!gwRewardChart) {
        gwRewardChart = echarts.init(document.getElementById('gw-reward-chart'));
    }
    if (!gwStepsChart) {
        gwStepsChart = echarts.init(document.getElementById('gw-steps-chart'));
    }

    const xData = Array.from({ length: data.avg_rewards.length }, (_, i) => i + 1);

    // 计算滑动平均
    const window = 20;
    const smoothRewards = movingAverage(data.avg_rewards, window);
    const smoothSteps = movingAverage(data.avg_steps, window);

    // 奖励图
    gwRewardChart.setOption({
        title: { text: '每回合总奖励', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '回合' },
        yAxis: { type: 'value', name: '总奖励' },
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
            }
        ]
    }, true);

    // 步数图
    gwStepsChart.setOption({
        title: { text: '每回合步数（越少越好）', left: 'center' },
        tooltip: { trigger: 'axis' },
        legend: { bottom: 0 },
        xAxis: { type: 'category', data: xData, name: '回合' },
        yAxis: { type: 'value', name: '步数' },
        grid: { left: 60, right: 20, bottom: 60, top: 50 },
        series: [
            {
                name: '每回合步数',
                type: 'line',
                showSymbol: false,
                data: data.avg_steps,
                lineStyle: { width: 1, opacity: 0.3 },
                itemStyle: { color: '#91cc75' }
            },
            {
                name: '滑动平均',
                type: 'line',
                showSymbol: false,
                data: smoothSteps,
                lineStyle: { width: 2 },
                itemStyle: { color: '#fac858' }
            }
        ]
    }, true);
}

function renderPolicyGrid(data) {
    const panel = document.getElementById('gw-policy-panel');
    panel.style.display = 'block';

    const container = document.getElementById('gw-grid-container');
    container.innerHTML = '';

    const { rows, cols, grid } = data;

    // 收集所有 Q 值来计算 V(s) 的范围（用于着色）
    let vMin = Infinity, vMax = -Infinity;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = grid[r][c];
            if (cell.type === 'wall') continue;
            const v = Math.max(...cell.q_values);
            if (v < vMin) vMin = v;
            if (v > vMax) vMax = v;
        }
    }

    // 设置 grid 布局
    container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = grid[r][c];
            const div = document.createElement('div');
            div.className = 'grid-cell';

            if (cell.type === 'start') {
                div.innerHTML = '<span class="cell-label">S</span>';
                div.style.background = valueToColor(Math.max(...cell.q_values), vMin, vMax);
                div.innerHTML += `<span class="cell-arrow">${cell.arrow}</span>`;
            } else if (cell.type === 'goal') {
                div.innerHTML = '<span class="cell-label goal-label">G</span>';
                div.style.background = '#38a169';
                div.style.color = '#fff';
            } else if (cell.type === 'trap') {
                div.innerHTML = '<span class="cell-label trap-label">X</span>';
                div.style.background = '#e53e3e';
                div.style.color = '#fff';
            } else if (cell.type === 'wall') {
                div.innerHTML = '<span class="cell-label wall-label">#</span>';
                div.style.background = '#4a5568';
                div.style.color = '#fff';
            } else {
                div.style.background = valueToColor(Math.max(...cell.q_values), vMin, vMax);
                div.innerHTML = `<span class="cell-arrow">${cell.arrow}</span>`;
            }

            // tooltip 显示 Q 值
            const qStr = cell.q_values.map((v, i) =>
                `${ARROW_MAP[i]}: ${v.toFixed(2)}`
            ).join('\n');
            div.title = `(${r},${c}) ${cell.type}\nQ 值:\n${qStr}`;

            container.appendChild(div);
        }
    }
}

function renderGWSummary(data) {
    const panel = document.getElementById('gw-summary-panel');
    panel.style.display = 'block';

    const content = document.getElementById('gw-summary-content');
    const s = data.summary;
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
                <span class="stat-value">${s.converged_episode}</span>
                <span class="stat-label">大约收敛回合</span>
            </div>
        </div>
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

// 将 V 值映射到红→黄→绿的颜色
function valueToColor(v, vMin, vMax) {
    if (vMax === vMin) return '#f0f0f0';
    const t = (v - vMin) / (vMax - vMin); // 0~1
    // 红(0) → 黄(0.5) → 绿(1)
    const r = Math.round(t < 0.5 ? 255 : 255 * (1 - (t - 0.5) * 2));
    const g = Math.round(t < 0.5 ? 255 * t * 2 : 255);
    const b = 80;
    return `rgba(${r}, ${g}, ${b}, 0.35)`;
}

// 窗口缩放时自适应图表大小
window.addEventListener('resize', () => {
    if (gwRewardChart) gwRewardChart.resize();
    if (gwStepsChart) gwStepsChart.resize();
});
