import { api } from './api.js?v=9';
import { renderErrorBar } from './charts.js?v=9';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const phaseSelect = document.getElementById('phase-select');
const taSelect    = document.getElementById('ta-select');
const enrolInput  = document.getElementById('enrollment-input');
const sitesInput  = document.getElementById('sites-input');
const predictBtn  = document.getElementById('predict-btn');
const enrolField  = document.getElementById('enrollment-field');
const sitesField  = document.getElementById('sites-field');
const enrolValue  = document.getElementById('enrollment-value');
const sitesValue  = document.getElementById('sites-value');
const sliderCaption = document.getElementById('slider-caption');
const resultCard  = document.getElementById('result-card');
const errorNotice = document.getElementById('error-notice');
const warnNotice  = document.getElementById('warn-notice');

// ── Bootstrap app ─────────────────────────────────────────────────────────────
async function init() {
  try {
    const [phases, areas] = await Promise.all([
      api.getPhases(), api.getTherapeuticAreas(),
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

    // The endpoint input is deliberately absent. It exposed WSi's internal
    // 11-value archetype vocabulary (SURVIVAL, BIOMARKER, …) and asked the user
    // to supply the strongest driver of duration after phase — which is a large
    // part of what they came to find out. It returns as a SELECTION from the
    // endpoint combinations that actually occur in this phase x area, once WS21
    // ships that clustering (docs/WS21_CONTRACT.md). Until then the model uses
    // what is typical for the area, which is what the blank option did anyway.

    predictBtn.disabled = false;
  } catch (err) {
    showError(`Could not reach backend: ${err.message}. Is the server running?`);
  }
}

// ── Predict ───────────────────────────────────────────────────────────────────
// Phase and therapeutic area stay behind an explicit action: they decide WHICH
// model answers, so changing them is a different question, not a refinement of
// the current one. Enrolment and sites are refinements and update live.
predictBtn.addEventListener('click', async () => {
  clearNotices();
  const phase = phaseSelect.value;
  const ta    = taSelect.value;
  if (!phase || !ta) return;

  setLoading(true);
  try {
    await armSliders(phase, ta);
    await runPrediction({ scroll: true });
  } finally {
    setLoading(false);
  }
});

function buildPayload() {
  const payload = { phase: phaseSelect.value, therapeutic_area: taSelect.value };
  if (!enrolField.hidden && enrolInput.value) payload.enrollment = Number(enrolInput.value);
  if (!sitesField.hidden && sitesInput.value) payload.num_sites  = Number(sitesInput.value);
  return payload;
}

async function runPrediction({ scroll = false } = {}) {
  try {
    errorNotice.classList.remove('visible');
    const result = await api.predict(buildPayload());
    displayResult(result, { scroll });
    return true;
  } catch (err) {
    if (err.message.includes('model')) {
      showWarn(err.message + ' Run <code>python -m scripts.train_models</code> to train.');
    } else {
      showError(`Prediction failed: ${err.message}`);
    }
    return false;
  }
}

// Bounds come from the model's TRAINED range for this phase and area, so the
// slider cannot be dragged somewhere the model has no evidence. The starting
// value is the therapeutic area's own median where one exists — an oncology
// Phase 3 runs a median 89 sites and a dermatology Phase 3 runs 33, and opening
// both at the same number describes a trial neither resembles.
async function armSliders(phase, ta) {
  try {
    const { inputs } = await api.getInputRanges(phase, ta);
    applyRange(inputs.enrollment, enrolField, enrolInput, enrolValue);
    applyRange(inputs.num_sites,  sitesField, sitesInput, sitesValue);
    sliderCaption.hidden = enrolField.hidden && sitesField.hidden;
  } catch (err) {
    // Sliders are a refinement; losing them must not cost the prediction.
    enrolField.hidden = true;
    sitesField.hidden = true;
    sliderCaption.hidden = true;
  }
}

function applyRange(range, field, input, output) {
  if (!range) { field.hidden = true; return; }
  input.min = range.min;
  input.max = range.max;
  input.step = range.max > 2000 ? 10 : 1;
  input.value = Math.min(Math.max(range.default, range.min), range.max);
  output.textContent = fmtCount(input.value);
  field.hidden = false;
}

function fmtCount(n) { return Number(n).toLocaleString(); }

// One in-flight request at a time. A drag fires an input event per pixel, and
// without this the answer displayed is whichever response happens to land last
// rather than the one matching where the handle stopped.
let pending = null;
let queued = false;

async function liveUpdate() {
  if (pending) { queued = true; return; }
  resultCard.classList.add('updating');
  do {
    queued = false;
    pending = runPrediction();
    await pending;
    pending = null;
  } while (queued);
  resultCard.classList.remove('updating');
}

[[enrolInput, enrolValue], [sitesInput, sitesValue]].forEach(([input, output]) => {
  // 'input' updates the READOUT on every pixel so the control feels direct;
  // the prediction itself is debounced, because it is a model call.
  input.addEventListener('input', () => { output.textContent = fmtCount(input.value); });
  let timer = null;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(liveUpdate, 180);
  });
  // A keyboard user tabbing away or releasing the handle should not wait out
  // the debounce.
  input.addEventListener('change', () => { clearTimeout(timer); liveUpdate(); });
});

// ── Display result ────────────────────────────────────────────────────────────
function displayResult(r, { scroll = false } = {}) {
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
  displaySplit(r);
  displayProvenance(r.provenance);

  // Set the warning state in ONE write. Warnings are per-input — a slider
  // position can be outside the trained range and the next one inside it — so
  // this both raises and clears them, without an intermediate hidden state that
  // would reflow the page between every pair of responses during a drag.
  if (r.extrapolation_warnings && r.extrapolation_warnings.length) {
    showWarn('Outside the trained range — treat as indicative only:<br>' +
             r.extrapolation_warnings.map(w => `· ${escapeHtml(w)}`).join('<br>'));
  } else {
    warnNotice.classList.remove('visible');
  }

  resultCard.classList.add('visible');
  // Scroll ONLY on the explicit Predict action, which reveals the card for the
  // first time. Doing it on every update yanked the page out from under a drag:
  // the slider fires continuously, and each response re-scrolled the window
  // while the handle was still held.
  if (scroll) resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Duration split: recruiting vs follow-up ───────────────────────────────────
function displaySplit(r) {
  const panel = document.getElementById('split-panel');
  if (r.enrolment_months == null || r.followup_months == null) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const total = Math.max(r.enrolment_months + r.followup_months, 0.01);
  document.getElementById('split-enrol').textContent = `${r.enrolment_months.toFixed(1)} mo`;
  document.getElementById('split-fu').textContent = `${r.followup_months.toFixed(1)} mo`;
  document.getElementById('split-enrol-bar').style.width =
    `${(r.enrolment_months / total) * 100}%`;
  document.getElementById('split-fu-bar').style.width =
    `${(r.followup_months / total) * 100}%`;
}

// ── Provenance: the working behind every number ───────────────────────────────
function displayProvenance(p) {
  const box = document.getElementById('working');
  if (!p) { box.hidden = true; return; }
  box.hidden = false;

  document.getElementById('prov-values').innerHTML =
    Object.entries(p.values || {}).map(([key, v]) => {
      const val = Array.isArray(v.value) ? v.value.join(' to ') : v.value;
      return `<div><dt>${labelFor(key)} — ${val} ${escapeHtml(v.unit || '')}</dt>
              <dd>${escapeHtml(v.derivation || '')}</dd></div>`;
    }).join('');

  // Origin is the point of this table: a number the user gave and a number
  // filled from an area median are not the same kind of input.
  document.getElementById('prov-inputs').innerHTML =
    Object.entries(p.inputs || {}).map(([key, v]) => {
      const n = v.evidence_n_trials ? ` from ${v.evidence_n_trials} trials` : '';
      return `<tr><td>${labelFor(key)}</td>
        <td>${v.value == null ? '—' : escapeHtml(String(v.value))}</td>
        <td><span class="origin ${v.origin}">${v.origin.replace('_', ' ')}</span>${n}</td></tr>`;
    }).join('');

  document.getElementById('prov-sources').innerHTML =
    (p.sources || []).map(s => {
      const n = s.n_trials || s.n_fit;
      const extra = n ? ` — ${n.toLocaleString()} trials` : '';
      const built = s.built ? ` · built ${s.built}` : '';
      return `<li><b>${escapeHtml(s.label)}</b>${extra}${built}
        ${s.selection ? `<br><span style="font-size:.75rem">${escapeHtml(s.selection)}</span>` : ''}</li>`;
    }).join('');

  document.getElementById('prov-gaps').innerHTML =
    (p.gaps || []).map(g => `<li>${escapeHtml(g)}</li>`).join('');
}

function labelFor(key) {
  return ({
    predicted_months: 'Total duration',
    enrolment_months: 'Recruiting window',
    followup_months: 'Follow-up',
    interval: 'Prediction interval',
    recruitment_rate: 'Recruitment rate',
    enrollment: 'Target enrolment',
    num_sites: 'Number of sites',
    drug_type: 'Intervention type',
    region: 'Region',
    endpoint_archetype: 'Primary endpoint type',
  })[key] || key.replace(/_/g, ' ');
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
