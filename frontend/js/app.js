import { api } from './api.js';
import { renderErrorBar } from './charts.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const phaseSelect = document.getElementById('phase-select');
const taSelect    = document.getElementById('ta-select');
const predictBtn  = document.getElementById('predict-btn');
const resultCard  = document.getElementById('result-card');
const errorNotice = document.getElementById('error-notice');
const warnNotice  = document.getElementById('warn-notice');

// ── Bootstrap app ─────────────────────────────────────────────────────────────
async function init() {
  try {
    const [phases, areas] = await Promise.all([api.getPhases(), api.getTherapeuticAreas()]);

    phaseSelect.options.length = 0;
    phaseSelect.add(new Option('Select a phase…', '', true, true));
    phaseSelect.options[0].disabled = true;

    phases.forEach(p => {
      const opt = new Option(p.label + (p.trained ? '' : ' (model pending)'), p.key);
      if (!p.trained) opt.style.color = '#8b949e';
      phaseSelect.add(opt);
    });

    taSelect.options.length = 0;
    taSelect.add(new Option('Select a therapeutic area…', '', true, true));
    taSelect.options[0].disabled = true;

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
