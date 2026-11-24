"use strict";

const get = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("zh-CN");
const count = (value) => number.format(value);
const percent = (ratio) => (ratio * 100).toFixed(1) + "%";
const signed = (value) => (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(1);
const opponentName = (name) => name === "computer ai" ? "内置 Bot" : name.toUpperCase();

const colors = { original: "#a8b8ee", trained: "#65e0c4", delta: "#65e0c4" };
const labels = { original: "原始 VDN", trained: "训练后 VDN", delta: "训练后 − 原始" };
const state = { mode: "rates", selected: 0, report: null, geometry: null };

function element(tag, content, className) {
  const el = document.createElement(tag);
  if (content !== undefined) el.textContent = content;
  if (className) el.className = className;
  return el;
}

function policyCounts(row, phase) {
  return ["win", "loss", "draw"].map((key) => row[phase + "_" + key]);
}

function rateCell(ratio, outcomes, phase) {
  const td = element("td", undefined, phase === "trained" ? "trained-cell" : "");
  td.append(
    element("span", percent(ratio), "result-rate"),
    element("span", outcomes.map(count).join(" / "), "result-count")
  );
  return td;
}

function resultRow(name, row) {
  const tr = element("tr");
  const header = element("th", name);
  header.scope = "row";
  tr.append(
    header,
    rateCell(row.original_win_rate, policyCounts(row, "original"), "original"),
    rateCell(row.trained_win_rate, policyCounts(row, "trained"), "trained"),
    element("td", signed(row.win_rate_delta_pp), row.win_rate_delta_pp < 0 ? "delta-negative" : "delta-positive")
  );
  return tr;
}

function renderScores(report, comparisons) {
  const total = {
    original_win_rate: report.original.win_rate,
    trained_win_rate: report.trained.win_rate,
    win_rate_delta_pp: 100 * (report.trained.win_rate - report.original.win_rate)
  };
  for (const phase of ["original", "trained"]) {
    for (const outcome of ["win", "loss", "draw"]) {
      const key = phase + "_" + outcome;
      total[key] = comparisons.reduce((sum, row) => sum + row[key], 0);
    }
    get(phase + "-score").textContent = percent(report[phase].win_rate);
    get(phase + "-games").textContent = count(total[phase + "_win"]) + " / " + count(report[phase].episodes) + " 胜";
  }
  get("gain-score").textContent = signed(total.win_rate_delta_pp);
  get("training-steps").textContent = count(report.training_steps);
  get("training-games").textContent = count(report.episodes);
  get("test-games").textContent = count(report.original.episodes + report.trained.episodes);
  get("release-time").dateTime = report.completed_at;
  get("release-time").textContent = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai", dateStyle: "medium", timeStyle: "short", hour12: false
  }).format(new Date(report.completed_at));

  get("results-note").textContent =
    "每个对手、每版 VDN 各 " + count(report.test_layouts) + " 局 · 次行依次为胜 / 负 / 平。";
  get("matchup-rows").replaceChildren(...comparisons.map((row) => resultRow(opponentName(row.opponent), row)));
  get("matchup-total").replaceChildren(resultRow("合计", total));
}

function renderTraining(report, comparisons) {
  const facts = [
    ["参数更新", count(report.updates) + " 次"],
    ["采样与更新", report.actors + " 个 SC2 实例 · RTX 4070"],
    ["训练种子", String(report.seed)],
    ["经验回放", count(report.replay_size) + " 局"],
    ["批量大小", count(report.batch_size) + " 局"],
    ["游戏版本", report.game_version]
  ].map(([label, value]) => {
    const item = element("div");
    item.append(element("dt", label), element("dd", value));
    return item;
  });
  get("training-facts").replaceChildren(...facts);
  get("opponents").replaceChildren(...comparisons.map((row) => {
    const tag = element("span", opponentName(row.opponent));
    tag.title = "训练中出场 " + count(row.training_episodes) + " 局";
    return tag;
  }));
  const exposure = Array.from(new Set(comparisons.map((row) => row.training_episodes)));
  get("opponent-note").textContent = "8 种 3m 预训练模型和 1 个内置 Bot。"
    + (exposure.length === 1 ? "每位对手出场 " + count(exposure[0]) + " 局。": "出场次数见各对手标签。")
    + "测试使用相同的固定对手库和留出出生点，A 始终为红方。";
}

function renderValidation(report) {
  const checks = report.validation.slice(-report.plateau.checks);
  get("validation-rows").replaceChildren(...checks.map((row) => {
    const tr = element("tr");
    const header = element("th", count(row.training_steps));
    header.scope = "row";
    tr.append(header, element("td", count(row.episodes)), element("td", percent(row.win_rate)), element("td", row.mean_return.toFixed(2)));
    return tr;
  }));
  get("validation-note").textContent = "达到训练预算和预设收敛条件后冻结模型，再运行最终测试。";
  get("plateau-note").textContent = "至少 " + count(report.minimum_steps) + " 环境步。最近 " + checks.length
    + " 轮验证的总胜率极差 ≤ " + (100 * report.plateau.win_rate_spread)
    + " 个百分点、平均回报极差 ≤ " + report.plateau.return_spread
    + "、每个对手的胜率极差 ≤ " + (100 * report.plateau.opponent_win_rate_spread) + " 个百分点。";
}

function drawBox(ctx, x, scale, width, summary, color) {
  const y1 = scale(summary.q3);
  const y2 = scale(summary.q1);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(x, scale(summary.high));
  ctx.lineTo(x, y1);
  ctx.moveTo(x, y2);
  ctx.lineTo(x, scale(summary.low));
  for (const end of [summary.high, summary.low]) {
    ctx.moveTo(x - width * 0.32, scale(end));
    ctx.lineTo(x + width * 0.32, scale(end));
  }
  ctx.stroke();
  ctx.fillStyle = color + "36";
  ctx.fillRect(x - width / 2, y1, width, Math.max(1, y2 - y1));
  ctx.strokeRect(x - width / 2, y1, width, Math.max(1, y2 - y1));
  ctx.fillStyle = color;
  ctx.fillRect(x - width / 2, scale(summary.median) - 1.1, width, 2.2);

  ctx.fillStyle = color + "86";
  for (const outlier of summary.outliers) {
    ctx.beginPath();
    ctx.arc(x, scale(outlier), 1.6, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function phases() {
  return state.mode === "rates" ? ["original", "trained"] : ["delta"];
}

function updatePlotSummary() {
  const active = state.report.comparisons[state.selected];
  const unit = state.mode === "rates" ? "%" : " 个百分点";
  get("plot-summary").textContent = phases().map((phase) => {
    const box = active.box[phase];
    return labels[phase] + " 中位数 " + box.median.toFixed(1) + unit
      + " · Q1–Q3 " + box.q1.toFixed(1) + "–" + box.q3.toFixed(1) + unit;
  }).join("  ｜  ");
}

function drawChart() {
  if (!state.report) return;
  const canvas = get("chart");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx || width < 1) return;
  ctx.scale(dpr, dpr);

  const comparisons = state.report.comparisons;
  const plot = { left: 52, right: width - 17, top: 26, bottom: height - 39 };
  const groups = (plot.right - plot.left) / comparisons.length;
  const scaleY = (value) => plot.bottom - value / 100 * (plot.bottom - plot.top);
  const centers = comparisons.map((_, i) => plot.left + groups * (i + 0.5));
  const boxWidth = Math.min(23, groups * 0.26);
  state.geometry = { plot, groups };

  ctx.fillStyle = "#192a3e";
  ctx.fillRect(plot.left + groups * state.selected + 1, plot.top, groups - 2, plot.bottom - plot.top);
  ctx.font = "11px Consolas, monospace";
  ctx.textBaseline = "middle";
  for (const tick of [0, 20, 40, 60, 80, 100]) {
    const y = scaleY(tick);
    ctx.strokeStyle = tick === 0 ? "#485e74" : "#25374d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillStyle = "#9eafc8";
    ctx.fillText(tick, plot.left - 9, y);
  }
  ctx.textAlign = "left";
  ctx.font = "11px Inter, Microsoft YaHei, sans-serif";
  ctx.fillText(state.mode === "rates" ? "胜率 (%)" : "胜率差 (百分点)", plot.left, 10);

  comparisons.forEach((row, index) => {
    const center = centers[index];
    ctx.textAlign = "center";
    ctx.fillStyle = index === state.selected ? "#edf3fc" : "#adbed3";
    ctx.fillText(opponentName(row.opponent), center, plot.bottom + 19);
    phases().forEach((phase, offset) => {
      const x = state.mode === "rates" ? center + (offset === 0 ? -1 : 1) * (boxWidth / 2 + 3) : center;
      drawBox(ctx, x, scaleY, boxWidth, row.box[phase], colors[phase]);
    });
  });
}

function setMode(mode) {
  state.mode = mode;
  get("mode-rates").setAttribute("aria-pressed", String(mode === "rates"));
  get("mode-delta").setAttribute("aria-pressed", String(mode === "delta"));
  get("chart").setAttribute("aria-label", mode === "rates" ? "各对手的重抽样胜率分布" : "各对手的配对胜率提升分布");
  get("chart-legend").replaceChildren(...phases().map((phase) => {
    const swatch = element("i");
    swatch.setAttribute("aria-hidden", "true");
    swatch.style.setProperty("--legend-color", colors[phase]);
    const label = element("span");
    label.append(swatch, labels[phase]);
    return label;
  }));
  updatePlotSummary();
  drawChart();
}

function setSelected(index) {
  state.selected = index;
  get("chart-opponent").value = String(index);
  updatePlotSummary();
  drawChart();
}

function setUpChart(report) {
  const picker = get("chart-opponent");
  picker.replaceChildren(...report.comparisons.map((row, index) => {
    const option = element("option", opponentName(row.opponent));
    option.value = String(index);
    return option;
  }));
  picker.addEventListener("change", () => setSelected(Number(picker.value)));
  get("mode-rates").addEventListener("click", () => setMode("rates"));
  get("mode-delta").addEventListener("click", () => setMode("delta"));
  const canvas = get("chart");
  canvas.addEventListener("click", (event) => {
    if (!state.geometry) return;
    const index = Math.floor((event.clientX - canvas.getBoundingClientRect().left - state.geometry.plot.left) / state.geometry.groups);
    if (index >= 0 && index < report.comparisons.length) setSelected(index);
  });
  new ResizeObserver(drawChart).observe(canvas.parentElement);
  get("box-method").textContent =
    "箱体为 Q1–Q3，线为中位数，须为 1.5 倍 IQR 内的样本范围。每次按 "
    + count(report.test_layouts) + " 个留出出生点成对重抽样，共 "
    + count(report.bootstrap.draws) + " 次，平局不计胜局。分布反映出生点抽样波动，实验使用单个训练种子。";
  setMode("rates");
}

async function loadReport() {
  const response = await fetch("./data/report.json");
  if (!response.ok) throw new Error("HTTP " + response.status);
  const report = await response.json();
  const comparisons = [...report.comparisons].sort((a, b) => Number(a.opponent === "computer ai") - Number(b.opponent === "computer ai"));
  state.report = { ...report, comparisons };
  renderScores(report, comparisons);
  renderTraining(report, comparisons);
  renderValidation(report);
  setUpChart(state.report);
  // Results loaded above the active section can move its original anchor.
  const target = get(location.hash.slice(1));
  if (target) requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));

  get("copy-link").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const url = new URL(location.href);
    url.searchParams.set("batch", report.batch);
    url.searchParams.set("version", report.version);
    url.hash = "release";
    try {
      await navigator.clipboard.writeText(url.href);
      button.textContent = "已复制";
    } catch {
      window.prompt("复制实验链接", url.href);
    }
    setTimeout(() => { button.textContent = "复制模块链接"; }, 1600);
  });
}

loadReport().catch((error) => {
  const message = get("load-error");
  message.hidden = false;
  message.textContent = "无法读取实验数据（" + error.message + "）。请通过本机 HTTP 服务或 GitHub Pages 打开。";
});
