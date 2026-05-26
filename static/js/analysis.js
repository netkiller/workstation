(() => {
  const root = document.querySelector(".analysis-dashboard");
  if (!root) return;

  const fileInput = root.querySelector("#fileInput");
  const uploadZone = root.querySelector("#uploadZone");
  const picker = root.querySelector("[data-file-picker]");
  const dashboard = root.querySelector("#dashboard");
  const emptyHint = root.querySelector("#empty-hint");
  let allData = [];
  let currentSlice = [];
  let charts = {};

  const colors = {
    cyan: "#1667c7",
    green: "#10b981",
    purple: "#7c3aed",
    amber: "#f59e0b",
    red: "#ef4444",
    blue: "#3b82f6",
    pink: "#ec4899",
    teal: "#14b8a6",
  };

  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  function chartTheme() {
    return {
      text: cssVar("--analysis-text") || "#1f2937",
      muted: cssVar("--analysis-muted") || "#64748b",
      border: cssVar("--analysis-border") || "#d9e1ec",
      surface2: cssVar("--analysis-surface2") || "#eef6ff",
    };
  }

  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (!lines.length) return {headers: [], rows: []};
    const headers = lines[0].split(",").map((item) => item.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i += 1) {
      const values = lines[i].split(",");
      if (values.length < headers.length) continue;
      const row = {};
      headers.forEach((header, index) => {
        const raw = String(values[index] ?? "").trim();
        const parsed = Number.parseFloat(raw);
        row[header] = Number.isFinite(parsed) ? parsed : raw;
      });
      rows.push(row);
    }
    return {headers, rows};
  }

  function getValue(row, key) {
    const value = Number.parseFloat(row?.[key]);
    return Number.isFinite(value) ? value : 0;
  }

  function getEpoch(row, fallback) {
    const value = Number.parseFloat(row?.epoch);
    return Number.isFinite(value) ? value : fallback + 1;
  }

  function destroyChart(id) {
    if (charts[id]) {
      charts[id].destroy();
      delete charts[id];
    }
  }

  function makeDataset(label, data, color, dashed = false) {
    return {
      label,
      data,
      borderColor: color,
      backgroundColor: `${color}18`,
      borderWidth: 2,
      pointRadius: data.length > 100 ? 0 : 2,
      pointHoverRadius: 4,
      tension: 0.3,
      fill: false,
      borderDash: dashed ? [6, 3] : [],
    };
  }

  function commonOpts(yLabel = "", pct = false) {
    const theme = chartTheme();
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {mode: "index", intersect: false},
      plugins: {
        legend: {
          labels: {
            color: theme.muted,
            font: {family: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace", size: 11},
            boxWidth: 14,
          },
        },
        tooltip: {
          backgroundColor: theme.surface2,
          borderColor: theme.border,
          borderWidth: 1,
          titleColor: theme.text,
          bodyColor: theme.muted,
          callbacks: pct ? {label: (ctx) => ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(2)}%`} : {},
        },
      },
      scales: {
        x: {
          grid: {color: theme.border},
          ticks: {color: theme.muted, font: {size: 10}, maxTicksLimit: 12},
          title: {display: true, text: "Epoch", color: theme.muted, font: {size: 11}},
        },
        y: {
          grid: {color: theme.border},
          ticks: {
            color: theme.muted,
            font: {size: 10},
            callback: pct ? (value) => `${(value * 100).toFixed(0)}%` : (value) => Number(value).toFixed(4),
          },
          title: {display: Boolean(yLabel), text: yLabel, color: theme.muted, font: {size: 11}},
        },
      },
    };
  }

  function makeChart(id, type, labels, datasets, options) {
    destroyChart(id);
    const canvas = root.querySelector(`#${id}`);
    if (!canvas || !window.Chart) return;
    charts[id] = new Chart(canvas.getContext("2d"), {type, data: {labels, datasets}, options});
  }

  function renderStats(data) {
    const statsGrid = root.querySelector("#statsGrid");
    if (!statsGrid || !data.length) return;
    const last = data[data.length - 1];
    const best = data.reduce(
      (bestRow, row) => getValue(row, "metrics/mAP50(B)") > getValue(bestRow, "metrics/mAP50(B)") ? row : bestRow,
      data[0],
    );
    const precision = getValue(last, "metrics/precision(B)");
    const recall = getValue(last, "metrics/recall(B)");
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const totalSeconds = getValue(last, "time");
    const cards = [
      {label: "Total Epochs", value: data.length, sub: `第 ${getEpoch(data[0], 0)} → ${getEpoch(last, data.length - 1)} 轮`, color: colors.cyan},
      {label: "Best mAP50", value: `${(getValue(best, "metrics/mAP50(B)") * 100).toFixed(2)}%`, sub: `Epoch ${getEpoch(best, 0)}`, color: colors.green},
      {label: "Best mAP50-95", value: `${(data.reduce((max, row) => Math.max(max, getValue(row, "metrics/mAP50-95(B)")), 0) * 100).toFixed(2)}%`, sub: "综合精度", color: colors.purple},
      {label: "Final Precision", value: `${(precision * 100).toFixed(2)}%`, sub: "最终精确率", color: colors.amber},
      {label: "Final Recall", value: `${(recall * 100).toFixed(2)}%`, sub: "最终召回率", color: colors.red},
      {label: "Final F1", value: `${(f1 * 100).toFixed(2)}%`, sub: "调和均值", color: colors.cyan},
      {label: "Val Box Loss", value: getValue(last, "val/box_loss").toFixed(4), sub: "最终验证", color: colors.red},
      {label: "Train Time", value: totalSeconds > 3600 ? `${(totalSeconds / 3600).toFixed(1)}h` : `${(totalSeconds / 60).toFixed(0)}m`, sub: "累计训练时间", color: colors.teal},
    ];
    statsGrid.innerHTML = cards.map((card) => `
      <div class="stat-card" style="--card-color:${card.color}">
        <div class="stat-label">${card.label}</div>
        <div class="stat-value">${card.value}</div>
        <div class="stat-sub">${card.sub}</div>
      </div>
    `).join("");
  }

  function renderCharts(data) {
    if (!window.Chart || !data.length) return;
    const epochs = data.map((row, index) => getEpoch(row, index));
    const get = (key) => data.map((row) => getValue(row, key));

    makeChart("chartMAP", "line", epochs, [
      makeDataset("mAP50", get("metrics/mAP50(B)"), colors.cyan),
      makeDataset("mAP50-95", get("metrics/mAP50-95(B)"), colors.purple),
    ], commonOpts("mAP", true));

    makeChart("chartPrecision", "line", epochs, [
      makeDataset("Precision", get("metrics/precision(B)"), colors.green),
    ], commonOpts("Precision", true));

    makeChart("chartRecall", "line", epochs, [
      makeDataset("Recall", get("metrics/recall(B)"), colors.purple),
    ], commonOpts("Recall", true));

    makeChart("chartPROverEpoch", "line", epochs, [
      makeDataset("Precision", get("metrics/precision(B)"), colors.green),
      makeDataset("Recall", get("metrics/recall(B)"), colors.amber),
    ], commonOpts("", true));

    makeChart("chartTotalLoss", "line", epochs, [
      makeDataset("Train Total", data.map((row) => getValue(row, "train/box_loss") + getValue(row, "train/cls_loss") + getValue(row, "train/dfl_loss")), colors.amber),
      makeDataset("Val Total", data.map((row) => getValue(row, "val/box_loss") + getValue(row, "val/cls_loss") + getValue(row, "val/dfl_loss")), colors.red, true),
    ], commonOpts("Loss"));

    makeChart("chartBoxLoss", "line", epochs, [
      makeDataset("Train Box", get("train/box_loss"), colors.amber),
      makeDataset("Val Box", get("val/box_loss"), colors.red, true),
    ], commonOpts("Box Loss"));

    makeChart("chartClsLoss", "line", epochs, [
      makeDataset("Train Cls", get("train/cls_loss"), colors.amber),
      makeDataset("Val Cls", get("val/cls_loss"), colors.red, true),
    ], commonOpts("Cls Loss"));

    makeChart("chartDflLoss", "line", epochs, [
      makeDataset("Train DFL", get("train/dfl_loss"), colors.purple),
      makeDataset("Val DFL", get("val/dfl_loss"), colors.pink, true),
    ], commonOpts("DFL Loss"));

    const lrKeys = Object.keys(data[0]).filter((key) => key.startsWith("lr/"));
    const lrColors = [colors.cyan, colors.green, colors.amber, colors.red, colors.purple, colors.blue, colors.pink, colors.teal];
    makeChart("chartLR", "line", epochs, lrKeys.map((key, index) => makeDataset(key, get(key), lrColors[index % lrColors.length])), commonOpts("Learning Rate"));

    renderPRScatter(data);

    const f1 = data.map((row) => {
      const precision = getValue(row, "metrics/precision(B)");
      const recall = getValue(row, "metrics/recall(B)");
      return precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    });
    makeChart("chartF1", "line", epochs, [makeDataset("F1 Score", f1, colors.amber)], commonOpts("F1", true));
    makeChart("chartFPR", "line", epochs, [makeDataset("误报率 (1-P)", get("metrics/precision(B)").map((value) => 1 - value), colors.red)], commonOpts("误报率", true));
  }

  function renderPRScatter(data) {
    destroyChart("chartPRScatter");
    const canvas = root.querySelector("#chartPRScatter");
    if (!canvas || !window.Chart) return;
    const theme = chartTheme();
    const points = data.map((row, index) => ({
      x: getValue(row, "metrics/recall(B)"),
      y: getValue(row, "metrics/precision(B)"),
      epoch: getEpoch(row, index),
    }));
    charts.chartPRScatter = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [{
          label: "P-R per Epoch",
          data: points.map((point) => ({x: point.x, y: point.y})),
          pointBackgroundColor: points.map((_, index) => `hsl(${180 + (index / Math.max(points.length, 1)) * 60}, 90%, ${40 + (index / Math.max(points.length, 1)) * 25}%)`),
          pointRadius: 4,
          pointHoverRadius: 7,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {display: false},
          tooltip: {
            backgroundColor: theme.surface2,
            borderColor: theme.border,
            borderWidth: 1,
            titleColor: theme.text,
            bodyColor: theme.muted,
            callbacks: {label: (ctx) => `Epoch ${points[ctx.dataIndex]?.epoch}  P=${(ctx.parsed.y * 100).toFixed(1)}%  R=${(ctx.parsed.x * 100).toFixed(1)}%`},
          },
        },
        scales: {
          x: {min: 0, max: 1, grid: {color: theme.border}, ticks: {color: theme.muted, callback: (value) => `${value * 100}%`}, title: {display: true, text: "Recall", color: theme.muted}},
          y: {min: 0, max: 1, grid: {color: theme.border}, ticks: {color: theme.muted, callback: (value) => `${value * 100}%`}, title: {display: true, text: "Precision", color: theme.muted}},
        },
      },
    });
  }

  function renderTable(data) {
    const tableContainer = root.querySelector("#tableContainer");
    const tableDesc = root.querySelector("#tableDesc");
    if (!tableContainer || !data.length) return;
    const best = data.reduce(
      (bestRow, row) => getValue(row, "metrics/mAP50(B)") > getValue(bestRow, "metrics/mAP50(B)") ? row : bestRow,
      data[0],
    );
    const cols = ["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"];
    const labels = ["Epoch", "Precision", "Recall", "mAP50", "mAP50-95", "Train Box↓", "Val Box↓", "Train Cls↓", "Val Cls↓"];
    const pct = [false, true, true, true, true, false, false, false, false];
    tableContainer.innerHTML = `
      <table>
        <thead><tr>${labels.map((label) => `<th>${label}</th>`).join("")}</tr></thead>
        <tbody>${data.map((row, index) => {
          const isBest = getEpoch(row, index) === getEpoch(best, 0);
          const prev = data[index - 1];
          const cells = cols.map((col, colIndex) => {
            const value = col === "epoch" ? getEpoch(row, index) : getValue(row, col);
            const formatted = pct[colIndex] ? `${(value * 100).toFixed(2)}%` : Number(value).toFixed(colIndex === 0 ? 0 : 4);
            let className = "";
            if (prev && colIndex > 0) {
              const delta = value - getValue(prev, col);
              if (Math.abs(delta) > 0.0001) {
                className = colIndex <= 4 ? (delta > 0 ? "up" : "dn") : (delta < 0 ? "up" : "dn");
              }
            }
            return `<td class="${className}">${formatted}</td>`;
          }).join("");
          return `<tr class="${isBest ? "best-row" : ""}">${cells}</tr>`;
        }).join("")}</tbody>
      </table>`;
    if (tableDesc) {
      tableDesc.textContent = `共 ${data.length} 行 · 最佳 mAP50 在 Epoch ${getEpoch(best, 0)} · 绿色=提升 红色=退步`;
    }
  }

  function renderAll(data) {
    currentSlice = data;
    renderStats(data);
    renderCharts(data);
    renderTable(data);
  }

  function onRangeChange() {
    let start = Number(root.querySelector("#epochStart")?.value ?? 0);
    let end = Number(root.querySelector("#epochEnd")?.value ?? 0);
    if (start > end) [start, end] = [end, start];
    root.querySelector("#epochStartLabel").textContent = `Ep ${getEpoch(allData[start], start)}`;
    root.querySelector("#epochEndLabel").textContent = `Ep ${getEpoch(allData[end], end)}`;
    renderAll(allData.slice(start, end + 1));
  }

  function resetRange() {
    if (!allData.length) return;
    root.querySelector("#epochStart").value = 0;
    root.querySelector("#epochEnd").value = allData.length - 1;
    onRangeChange();
  }

  function initDashboard() {
    if (!dashboard) return;
    dashboard.hidden = false;
    if (emptyHint) emptyHint.hidden = true;
    if (uploadZone) uploadZone.hidden = true;
    const startSlider = root.querySelector("#epochStart");
    const endSlider = root.querySelector("#epochEnd");
    startSlider.max = allData.length - 1;
    startSlider.value = 0;
    endSlider.max = allData.length - 1;
    endSlider.value = allData.length - 1;
    startSlider.addEventListener("input", onRangeChange);
    endSlider.addEventListener("input", onRangeChange);
    onRangeChange();
  }

  function loadFile(file) {
    if (!file || !file.name.toLowerCase().endsWith(".csv")) {
      alert("请选择 results.csv 文件");
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      const {rows} = parseCSV(String(event.target?.result ?? ""));
      if (!rows.length) {
        alert("CSV 解析失败，请检查文件格式");
        return;
      }
      allData = rows;
      initDashboard();
    };
    reader.readAsText(file);
  }

  function showTab(name, button) {
    root.querySelectorAll(".tab-content").forEach((item) => {
      item.hidden = true;
    });
    root.querySelectorAll(".tab").forEach((item) => {
      item.classList.remove("active");
    });
    const panel = root.querySelector(`#tab-${name}`);
    if (panel) panel.hidden = false;
    button?.classList.add("active");
  }

  picker?.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput?.click();
  });
  uploadZone?.addEventListener("click", () => fileInput?.click());
  fileInput?.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) loadFile(file);
  });
  uploadZone?.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadZone.classList.add("drag-over");
  });
  uploadZone?.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
  uploadZone?.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadZone.classList.remove("drag-over");
    const file = event.dataTransfer?.files?.[0];
    if (file) loadFile(file);
  });

  window.addEventListener("yolows-theme-change", () => {
    if (currentSlice.length) renderAll(currentSlice);
  });

  if (!window.Chart) {
    if (emptyHint) emptyHint.textContent = "图表库加载失败，请检查网络后刷新页面。";
  }

  const embeddedRows = root.querySelector("[data-analysis-rows]");
  if (embeddedRows) {
    try {
      const rows = JSON.parse(embeddedRows.textContent || "[]");
      if (Array.isArray(rows) && rows.length) {
        allData = rows;
        initDashboard();
      }
    } catch (error) {
      if (emptyHint) emptyHint.textContent = "results.csv 数据读取失败。";
    }
  }

  window.resetRange = resetRange;
  window.showTab = showTab;
})();
