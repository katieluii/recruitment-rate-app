/* Plotly chart renderers */

const COLORS = {
  accent:   '#58a6ff',
  accentDim:'rgba(88,166,255,0.18)',
  warn:     '#d29922',
  muted:    '#8b949e',
  bg:       '#161b22',
  grid:     '#21262d',
  text:     '#e6edf3',
};

const LAYOUT_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:  { family: 'Inter, system-ui, sans-serif', color: COLORS.text, size: 11 },
  margin: { t: 10, r: 20, b: 60, l: 170 },
  xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, color: COLORS.muted },
  yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid, color: COLORS.muted },
  legend: { bgcolor: 'transparent', font: { color: COLORS.text } },
};

const CONFIG = { displayModeBar: false, responsive: true };

// ── Chart 1: Distribution box plot by therapeutic area ────────────────────────
export function renderBoxPlot(containerId, data) {
  const sorted = [...data].sort((a, b) => a.median - b.median);

  const trace = {
    type: 'bar',
    orientation: 'h',
    x: sorted.map(d => d.median / 30.44),         // convert to months
    y: sorted.map(d => d.therapeutic_area),
    error_x: {
      type: 'data',
      symmetric: false,
      array:      sorted.map(d => (d.q75 - d.median) / 30.44),
      arrayminus: sorted.map(d => (d.median - d.q25) / 30.44),
      color: COLORS.accentDim,
      thickness: 6,
      width: 4,
    },
    marker: { color: COLORS.accent, opacity: 0.75 },
    hovertemplate:
      '<b>%{y}</b><br>Median: %{x:.1f} mo<br>' +
      'IQR: ' +
      sorted.map(d =>
        `${(d.q25/30.44).toFixed(1)} – ${(d.q75/30.44).toFixed(1)} mo`
      ).join('<br>') +
      '<extra></extra>',
    customdata: sorted.map(d => ({
      q25:  (d.q25  / 30.44).toFixed(1),
      q75:  (d.q75  / 30.44).toFixed(1),
      q10:  (d.q10  / 30.44).toFixed(1),
      q90:  (d.q90  / 30.44).toFixed(1),
      n:    d.n,
    })),
  };

  // Override hovertemplate with customdata
  trace.hovertemplate = sorted.map((d, i) => {
    const q25m = (d.q25/30.44).toFixed(1);
    const q75m = (d.q75/30.44).toFixed(1);
    const nm   = (d.median/30.44).toFixed(1);
    return `<b>${d.therapeutic_area}</b><br>Median: ${nm} mo<br>IQR: ${q25m}–${q75m} mo<br>n=${d.n}<extra></extra>`;
  });

  const layout = {
    ...LAYOUT_BASE,
    margin: { t: 10, r: 30, b: 45, l: 210 },
    xaxis: { ...LAYOUT_BASE.xaxis, title: { text: 'Duration (months)', font: { color: COLORS.muted } } },
    yaxis: { ...LAYOUT_BASE.yaxis, automargin: true },
    bargap: 0.35,
  };

  Plotly.newPlot(containerId, [trace], layout, CONFIG);
}

// ── Chart 2: Phase comparison bar chart ───────────────────────────────────────
export function renderPhaseComparison(containerId, phaseSummaries) {
  // phaseSummaries: [{ phase, label, median_months, q25_months, q75_months }]
  const trace = {
    type: 'bar',
    x: phaseSummaries.map(p => p.label),
    y: phaseSummaries.map(p => p.median_months),
    error_y: {
      type: 'data',
      symmetric: false,
      array:      phaseSummaries.map(p => p.q75_months - p.median_months),
      arrayminus: phaseSummaries.map(p => p.median_months - p.q25_months),
      color: COLORS.accentDim,
      thickness: 3,
      width: 8,
    },
    marker: {
      color: [COLORS.accent, '#3fb950', '#d29922', '#f85149'].slice(0, phaseSummaries.length),
      opacity: 0.8,
    },
    hovertemplate: '<b>%{x}</b><br>Median: %{y:.1f} mo<extra></extra>',
  };

  const layout = {
    ...LAYOUT_BASE,
    margin: { t: 10, r: 20, b: 60, l: 55 },
    yaxis: {
      ...LAYOUT_BASE.yaxis,
      title: { text: 'Duration (months)', font: { color: COLORS.muted } },
    },
    bargap: 0.4,
  };

  Plotly.newPlot(containerId, [trace], layout, CONFIG);
}

// ── Prediction error bar (inline SVG-based) ───────────────────────────────────
export function renderErrorBar(lower, predicted, upper, containerId, confidencePct = 80) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const range = upper - lower;
  const fillLeft  = ((predicted - lower) / range * 100).toFixed(1);
  const fillRight = ((upper - predicted) / range * 100).toFixed(1);
  const centerPct = ((predicted - lower) / range * 100).toFixed(1);

  el.innerHTML = `
    <div class="error-bar-label">
      ${confidencePct}% prediction interval &nbsp;·&nbsp;
      <span style="color:var(--accent)">${lower.toFixed(1)} – ${upper.toFixed(1)} months</span>
    </div>
    <div class="error-bar-track">
      <div class="error-bar-fill" style="left:0;width:100%"></div>
      <div class="error-bar-center" style="left:${centerPct}%"></div>
    </div>
    <div class="error-bar-ticks">
      <span>${lower.toFixed(1)} mo</span>
      <span style="color:var(--accent);font-weight:600">▲ ${predicted.toFixed(1)} mo</span>
      <span>${upper.toFixed(1)} mo</span>
    </div>
  `;
}
