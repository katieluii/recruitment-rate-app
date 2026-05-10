import { api } from './api.js';
import { renderBoxPlot, renderPhaseComparison, renderErrorBar } from './charts.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const phaseSelect  = document.getElementById('phase-select');
const taSelect     = document.getElementById('ta-select');
const predictBtn   = document.getElementById('predict-btn');
const resultCard   = document.getElementById('result-card');
const chartsSection= document.getElementById('charts-section');
const errorNotice  = document.getElementById('error-notice');
const warnNotice   = document.getElementById('warn-notice');

// ── Bootstrap app ─────────────────────────────────────────────────────────────
async function init() {
  try {
    const [phases, areas] = await Promise.all([api.getPhases(), api.getTherapeuticAreas()]);

    phases.forEach(p => {
      const opt = new Option(p.label + (p.trained ? '' : ' (model pending)'), p.key);
      if (!p.trained) opt.style.color = '#8b949e';
      phaseSelect.add(opt);
    });

    areas.forEach(ta => taSelect.add(new Option(ta, ta)));

    predictBtn.disabled = false;
  } catch (err) {
    showError(`Could not reach backend: ${err.message}. Is the server running?`);
  }
}

// ── Predict ───────────────────────────────────────────────────────────────────
predictBtn.addEventListener('click', async () => {
  clearNotices();
  const phase = phaseSelect.value;
  const ta    = taSelect.value;
  if (!phase || !ta) return;

  setLoading(true);

  try {
    const result = await api.predict({ phase, therapeutic_area: ta });
    displayResult(result);
    await loadCharts(phase, result);
  } catch (err) {
    if (err.message.includes('model')) {
      showWarn(err.message + ' Run <code>python -m scripts.train_models</code> to train.');
    } else {
      showError(`Prediction failed: ${err.message}`);
    }
  } finally {
    setLoading(false);
  }
});

// ── Display result ────────────────────────────────────────────────────────────
function displayResult(r) {
  document.getElementById('result-months').textContent = r.predicted_months.toFixed(1);
  document.getElementById('result-phase-label').textContent = r.phase_label;
  document.getElementById('result-ta').textContent = r.therapeutic_area;

  document.getElementById('stat-days').textContent   = `${r.predicted_days.toFixed(0)} d`;
  document.getElementById('stat-rmse').textContent   = `±${r.rmse_days.toFixed(0)} d`;
  document.getElementById('stat-ntrain').textContent = r.n_train.toLocaleString();

  renderErrorBar(r.lower_months, r.predicted_months, r.upper_months, 'error-bar-container');

  resultCard.classList.add('visible');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Charts ────────────────────────────────────────────────────────────────────
let _analyticsCache = {};

async function loadCharts(phase, result) {
  chartsSection.classList.add('visible');

  // Chart 1: distribution for selected phase
  if (!_analyticsCache[phase]) {
    _analyticsCache[phase] = await api.getAnalytics(phase).catch(() => null);
  }
  const analytics = _analyticsCache[phase];
  if (analytics?.data?.length) {
    renderBoxPlot('chart-distribution', analytics.data);
  }

  // Chart 2: phase comparison (use median from result + typical values)
  const phaseSummaries = buildPhaseSummaries(result);
  renderPhaseComparison('chart-phases', phaseSummaries);

  chartsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function buildPhaseSummaries(result) {
  // Typical median durations (months) per phase, from published literature
  const typical = {
    P1HV: { label: 'Ph 1 (HV)',      median: 6,  q25: 4,  q75: 9  },
    P1:   { label: 'Ph 1 (Patient)', median: 12, q25: 8,  q75: 18 },
    P2:   { label: 'Phase 2',        median: 24, q25: 16, q75: 36 },
    P3:   { label: 'Phase 3',        median: 36, q25: 24, q75: 54 },
  };
  // Replace the selected phase with model prediction
  const summary = { ...typical };
  summary[result.phase_key] = {
    label:         typical[result.phase_key]?.label || result.phase_label,
    median:        result.predicted_months,
    q25:           result.lower_months,
    q75:           result.upper_months,
  };
  return Object.entries(summary).map(([k, v]) => ({ phase: k, ...v,
    median_months: v.median, q25_months: v.q25, q75_months: v.q75 }));
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const target = btn.dataset.target;
    document.querySelectorAll('.chart-panel').forEach(p => {
      p.style.display = p.id === target ? 'block' : 'none';
    });
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function setLoading(on) {
  predictBtn.classList.toggle('loading', on);
  predictBtn.disabled = on;
}

function clearNotices() {
  errorNotice.classList.remove('visible');
  warnNotice.classList.remove('visible');
}

function showError(msg) {
  errorNotice.innerHTML = msg;
  errorNotice.classList.add('visible');
}

function showWarn(msg) {
  warnNotice.innerHTML = msg;
  warnNotice.classList.add('visible');
}

init();
