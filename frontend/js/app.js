import { api } from './api.js?v=5';
import { renderErrorBar } from './charts.js?v=5';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const phaseSelect = document.getElementById('phase-select');
const taSelect    = document.getElementById('ta-select');
const endpointSel = document.getElementById('endpoint-select');
const enrolInput  = document.getElementById('enrollment-input');
const sitesInput  = document.getElementById('sites-input');
const predictBtn  = document.getElementById('predict-btn');
const resultCard  = document.getElementById('result-card');
const errorNotice = document.getElementById('error-notice');
const warnNotice  = document.getElementById('warn-notice');

// ── Bootstrap app ─────────────────────────────────────────────────────────────
async function init() {
  try {
    const [phases, areas, archetypes] = await Promise.all([
      api.getPhases(), api.getTherapeuticAreas(), api.getEndpointArchetypes(),
    ]);

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

    // Endpoint type is the strongest driver of duration after phase: on Phase 3
    // a survival endpoint runs a median 39.6 months against 10.6 for
    // immunogenicity. Left blank, the model assumes what is typical for the area.
    archetypes.forEach(a => endpointSel.add(new Option(prettyArchetype(a), a)));

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
    const payload = { phase, therapeutic_area: ta };
    if (endpointSel.value) payload.endpoint_archetype = endpointSel.value;
    if (enrolInput.value)  payload.enrollment = Number(enrolInput.value);
    if (sitesInput.value)  payload.num_sites  = Number(sitesInput.value);

    const result = await api.predict(payload);
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
  // The interval is now a genuine conformalised quantile range, so show the
  // range itself rather than a single global RMSE that applied to every input.
  document.getElementById('stat-interval').textContent =
    `${r.lower_months.toFixed(1)}–${r.upper_months.toFixed(1)} mo`;
  document.getElementById('stat-ntrain').textContent = r.n_train.toLocaleString();
  document.getElementById('result-model').textContent =
    `${r.model_used} · ${r.confidence_pct}% prediction interval`;

  renderErrorBar(r.lower_months, r.predicted_months, r.upper_months, 'error-bar-container');
  displayRate(r);

  if (r.extrapolation_warnings && r.extrapolation_warnings.length) {
    showWarn('Outside the trained range — treat as indicative only:<br>' +
             r.extrapolation_warnings.map(w => `· ${escapeHtml(w)}`).join('<br>'));
  }

  resultCard.classList.add('visible');
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Recruitment rate ──────────────────────────────────────────────────────────
function displayRate(r) {
  const panel = document.getElementById('rate-panel');
  if (r.recruitment_rate == null) { panel.hidden = true; return; }

  panel.hidden = false;
  document.getElementById('rate-value').textContent = r.recruitment_rate.toFixed(2);
  document.getElementById('rate-range').textContent =
    `80% interval ${r.recruitment_rate_lower.toFixed(2)}–${r.recruitment_rate_upper.toFixed(2)}`;
  // The caveat ships with the number, not in a footnote. This figure is modelled
  // from trial-level data, not observed per-site enrolment.
  document.getElementById('rate-note').textContent = r.rate_note || '';
}

function prettyArchetype(a) {
  return a.replace(/_/g, ' ').toLowerCase()
          .replace(/\b\w/g, c => c.toUpperCase());
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
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
