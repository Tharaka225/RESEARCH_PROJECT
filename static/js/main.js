/* ═══════════════════════════════════════════════════════════════
   NeuroShield — Main JavaScript
   All API endpoints and variable names preserved exactly.
   ═══════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────
let uploadedModelId = null;
let selectedImageFile = null;
let currentResults = null;
let activeTab = 'demo';

const CLASS_EMOJI = {
  airplane: '✈️', automobile: '🚗', bird: '🐦', cat: '🐱',
  deer: '🦌', dog: '🐕', frog: '🐸', horse: '🐴', ship: '🚢', truck: '🚛'
};

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initImageUpload();
  initModelUpload();
  checkSystemStatus();
});

// ── System Status ─────────────────────────────────────────────
async function checkSystemStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const el = document.getElementById('systemStatus');
    if (data.status === 'ok') {
      el.textContent = 'SYSTEM ONLINE';
      el.style.color = 'var(--green)';
    }
  } catch {
    const el = document.getElementById('systemStatus');
    el.textContent = 'OFFLINE';
    el.style.color = 'var(--red)';
  }
}

// ── Tab Switching ─────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  document.getElementById('tabDemo').classList.toggle('active', tab === 'demo');
  document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
  document.getElementById('demoPanel').style.display   = tab === 'demo'   ? '' : 'none';
  document.getElementById('uploadPanel').style.display = tab === 'upload' ? '' : 'none';
}

// ── Image Upload ──────────────────────────────────────────────
function initImageUpload() {
  const zone  = document.getElementById('imageDropZone');
  const input = document.getElementById('imageInput');

  zone.addEventListener('click', (e) => {
    if (!e.target.closest('button') && !e.target.closest('.image-preview')) {
      input.click();
    }
  });

  input.addEventListener('change', () => {
    if (input.files[0]) handleImageFile(input.files[0]);
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleImageFile(file);
  });

  document.getElementById('clearImageBtn').addEventListener('click', (e) => {
    e.stopPropagation();
    clearImage();
  });
}

function handleImageFile(file) {
  selectedImageFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('imageName').textContent = file.name;
    document.getElementById('imageSize').textContent =
      `${(file.size / 1024).toFixed(1)} KB`;
    document.getElementById('uploadContent').style.display = 'none';
    document.getElementById('imagePreview').style.display  = '';
  };
  reader.readAsDataURL(file);
  updateAnalyzeBtn();
}

function clearImage() {
  selectedImageFile = null;
  document.getElementById('imageInput').value = '';
  document.getElementById('uploadContent').style.display = '';
  document.getElementById('imagePreview').style.display  = 'none';
  updateAnalyzeBtn();
}

// ── Model Upload ──────────────────────────────────────────────
function initModelUpload() {
  const zone  = document.getElementById('modelDropZone');
  const input = document.getElementById('modelInput');

  if (!zone || !input) return;

  zone.addEventListener('click', (e) => {
    if (!e.target.closest('button')) input.click();
  });

  input.addEventListener('change', () => {
    if (input.files[0]) uploadModel(input.files[0]);
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) uploadModel(file);
  });
}

async function uploadModel(file) {
  const content  = document.getElementById('modelUploadContent');
  const loaded   = document.getElementById('modelLoaded');
  const nameEl   = document.getElementById('modelFilename');

  content.innerHTML = `<span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-3)">Uploading…</span>`;

  const form = new FormData();
  form.append('model', file);

  try {
    const res  = await fetch('/api/upload-model', { method: 'POST', body: form });
    const data = await res.json();

    if (data.model_id) {
      uploadedModelId = data.model_id;
      content.style.display = 'none';
      loaded.style.display  = '';
      nameEl.textContent    = file.name;
      updateAnalyzeBtn();
    } else {
      content.innerHTML = `<span style="color:var(--red);font-size:0.75rem">${data.error || 'Upload failed'}</span>`;
    }
  } catch {
    content.innerHTML = `<span style="color:var(--red);font-size:0.75rem">Network error</span>`;
  }
}

// ── Analyse Button State ──────────────────────────────────────
function updateAnalyzeBtn() {
  const ready = selectedImageFile &&
    (activeTab === 'demo' || uploadedModelId);
  document.getElementById('analyzeBtn').disabled = !ready;
}

// ── Analysis ──────────────────────────────────────────────────
async function startAnalysis() {
  if (!selectedImageFile) return;

  showState('progress');
  resetSteps();
  setProgress(0);

  const form = new FormData();
  form.append('image', selectedImageFile);
  form.append('model_id',   activeTab === 'upload' && uploadedModelId ? uploadedModelId : 'demo');
  form.append('image_size', document.getElementById('imageSizeSelect').value);
  form.append('num_classes', document.getElementById('numClasses')?.value || '10');

  const arch = document.getElementById('architectureSelect')?.value;
  if (arch) form.append('architecture', arch);

  // Simulate step progression
  const stepTimings = [
    { step: 1, progress: 15, delay: 800 },
    { step: 2, progress: 45, delay: 2400 },
    { step: 3, progress: 70, delay: 4000 },
    { step: 4, progress: 90, delay: 5500 },
  ];

  let stepIdx = 0;
  const stepTimer = setInterval(() => {
    if (stepIdx < stepTimings.length) {
      const { step, progress } = stepTimings[stepIdx];
      setStepActive(step);
      setProgress(progress);
      stepIdx++;
    }
  }, stepTimings[0]?.delay || 800);

  // Actually run timings sequentially
  const runTimings = async () => {
    for (let i = 0; i < stepTimings.length; i++) {
      await sleep(i === 0 ? stepTimings[0].delay : stepTimings[i].delay - stepTimings[i-1].delay);
      setStepActive(stepTimings[i].step);
      setProgress(stepTimings[i].progress);
    }
  };

  clearInterval(stepTimer);
  runTimings();

  try {
    const res  = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await res.json();

    setProgress(100);
    markAllDone();

    await sleep(500);

    if (data.status === 'error') {
      showError(data.error || 'Analysis failed');
      return;
    }

    currentResults = data;
    renderResults(data);
    showState('results');

  } catch (err) {
    showError(err.message || 'Network error');
  }
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── Step Management ───────────────────────────────────────────
function resetSteps() {
  [1,2,3,4].forEach(n => {
    const el = document.getElementById(`step${n}`);
    if (el) {
      el.classList.remove('active', 'done');
      const badge = el.querySelector('.step-badge');
      if (badge) badge.textContent = 'PENDING';
    }
  });
}

function setStepActive(n) {
  // Mark previous as done
  if (n > 1) setStepDone(n - 1);

  const el = document.getElementById(`step${n}`);
  if (!el) return;
  el.classList.add('active');
  el.classList.remove('done');
  const badge = el.querySelector('.step-badge');
  if (badge) badge.textContent = 'RUNNING';
}

function setStepDone(n) {
  const el = document.getElementById(`step${n}`);
  if (!el) return;
  el.classList.remove('active');
  el.classList.add('done');
  const badge = el.querySelector('.step-badge');
  if (badge) badge.textContent = '✓ DONE';
}

function markAllDone() {
  [1,2,3,4].forEach(n => setStepDone(n));
}

// ── Progress Bar ──────────────────────────────────────────────
function setProgress(pct) {
  const bar   = document.getElementById('progressBar');
  const label = document.getElementById('progressLabel');
  if (bar)   bar.style.width   = `${pct}%`;
  if (label) label.textContent = `${Math.round(pct)}%`;
}

// ── State Manager ─────────────────────────────────────────────
function showState(state) {
  document.getElementById('idleState').style.display     = state === 'idle'     ? '' : 'none';
  document.getElementById('progressState').style.display = state === 'progress' ? '' : 'none';
  document.getElementById('resultsState').style.display  = state === 'results'  ? '' : 'none';
  document.getElementById('errorState').style.display    = state === 'error'    ? '' : 'none';
}

function showError(msg) {
  showState('error');
  const el = document.getElementById('errorMessage');
  if (el) el.textContent = msg;
}

// ── Render Results ────────────────────────────────────────────
function renderResults(data) {
  const c1 = data.components?.component1 || {};
  const c2 = data.components?.component2 || {};
  const c3 = data.components?.component3 || {};
  const c4 = data.components?.component4 || {};

  // Verdict banner
  const banner  = document.getElementById('verdictBanner');
  const verdict = data.verdict || '—';
  banner.className = `verdict-banner ${verdict.toLowerCase()}`;

  document.getElementById('verdictText').textContent = verdict;
  document.getElementById('anomalyScoreDisplay').textContent =
    data.anomaly_score != null ? `${(data.anomaly_score * 100).toFixed(1)}%` : '—';
  document.getElementById('stabilityVal').textContent =
    c4.mean_stability != null ? `${(c4.mean_stability * 100).toFixed(1)}%` : '—';
  document.getElementById('certRadius').textContent =
    c4.certified_radius != null ? c4.certified_radius.toFixed(4) : '—';

  // Component 1
  renderPredictionCard(c1);
  renderC1Metrics(c1);

  // Component 2
  renderC2Metrics(c2);

  // Component 3 — feature chart
  renderFeatureChart(c3);

  // Component 4
  renderC4Metrics(c4);
}

function renderPredictionCard(c1) {
  const wrap = document.getElementById('predictionCard');
  if (!wrap) return;

  const cls   = c1.original_class_name || 'unknown';
  const conf  = c1.original_conf != null ? (c1.original_conf * 100).toFixed(1) : '—';
  const emoji = CLASS_EMOJI[cls] || '🔷';

  wrap.innerHTML = `
    <div class="pred-class-icon">${emoji}</div>
    <div class="pred-info">
      <div class="pred-class-name">${cls}</div>
      <div class="pred-class-label">PREDICTED CLASS</div>
    </div>
    <div class="pred-conf">
      <div class="pred-conf-val">${conf}%</div>
      <div class="pred-conf-label">CONFIDENCE</div>
    </div>
  `;

  // Confidence bar
  const barFill = document.getElementById('confBarFill');
  const barPct  = document.getElementById('confBarPct');
  if (barFill) barFill.style.width = `${conf}%`;
  if (barFill) barFill.style.background =
    parseFloat(conf) > 90 ? 'var(--red)' :
    parseFloat(conf) > 70 ? 'var(--amber)' : 'var(--green)';
  if (barPct) barPct.textContent = `${conf}%`;
}

function renderC1Metrics(c1) {
  const grid = document.getElementById('c1Metrics');
  if (!grid) return;

  const metrics = [
    { label: 'FGSM Δ',          val: c1.delta_fgsm,        fmt: v => v.toFixed(4) },
    { label: 'I-FGSM Δ',        val: c1.delta_ifgsm,       fmt: v => v.toFixed(4) },
    { label: 'Min Flip ε',       val: c1.min_flip_epsilon,  fmt: v => v.toFixed(4),
      cls: v => v > 0.4 ? 'highlight-bad' : v < 0.1 ? 'highlight-good' : '' },
    { label: 'FGSM Conf',        val: c1.fgsm_conf,         fmt: v => `${(v*100).toFixed(1)}%` },
    { label: 'I-FGSM Conf',      val: c1.ifgsm_conf,        fmt: v => `${(v*100).toFixed(1)}%` },
    { label: 'Conf Margin',      val: c1.conf_margin,       fmt: v => v.toFixed(4) },
    { label: 'Gradient L2',      val: c1.grad_l2,           fmt: v => v.toFixed(4) },
    { label: 'Gradient Max',     val: c1.grad_max,          fmt: v => v.toFixed(4) },
    { label: 'Gradient Var',     val: c1.gradient_variance, fmt: v => v.toFixed(4) },
    { label: 'Loss Sensitivity', val: c1.loss_sensitivity,  fmt: v => v.toFixed(4) },
  ];

  grid.innerHTML = metrics.map(m => {
    const val    = m.val != null ? m.fmt(m.val) : '—';
    const cls    = m.cls && m.val != null ? m.cls(m.val) : '';
    return `
      <div class="metric-card">
        <div class="metric-card-label">${m.label}</div>
        <div class="metric-card-val ${cls}">${val}</div>
      </div>`;
  }).join('');
}

function renderC2Metrics(c2) {
  const grid = document.getElementById('c2Metrics');
  if (!grid) return;

  const metrics = [
    { label: 'Black-Box Δ',    val: c2.delta_blackbox,    fmt: v => v.toFixed(4),
      cls: v => v > 0.4 ? 'highlight-bad' : v < 0.1 ? 'highlight-good' : '' },
    { label: 'FD Sensitivity', val: c2.fd_sensitivity,    fmt: v => v.toFixed(4) },
    { label: 'Boundary Dist',  val: c2.boundary_distance, fmt: v => v.toFixed(4) },
    { label: 'Query Count',    val: c2.query_count,       fmt: v => v.toString() },
    { label: 'BB Confidence',  val: c2.bb_confidence,     fmt: v => `${(v*100).toFixed(1)}%` },
    { label: 'Decision Score', val: c2.decision_score,    fmt: v => v.toFixed(4) },
  ];

  grid.innerHTML = metrics.filter(m => m.val != null).map(m => {
    const val = m.fmt(m.val);
    const cls = m.cls ? m.cls(m.val) : '';
    return `
      <div class="metric-card">
        <div class="metric-card-label">${m.label}</div>
        <div class="metric-card-val ${cls}">${val}</div>
      </div>`;
  }).join('') || '<p style="color:var(--text-3);font-size:0.8rem">No data available</p>';
}

function renderFeatureChart(c3) {
  const wrap = document.getElementById('featureChart');
  if (!wrap) return;

  const vec = c3.feature_vector || [];
  if (!vec.length) {
    wrap.innerHTML = '<p style="color:var(--text-3);font-size:0.8rem">No feature data</p>';
    return;
  }

  const labels = [
    'orig_conf','delta_fgsm','delta_ifgsm','grad_norm','loss_sens',
    'fgsm_conf','ifgsm_conf','conf_margin','grad_var','grad_max',
    'flip_eps','bb_delta','fd_sens','boundary','bb_conf',
    'decision','stability','cert_radius','stable_flag','high_stab',
    'mean_stab','min_stab','fgsm_chg','ifgsm_chg','feat_25'
  ];

  const maxAbs = Math.max(...vec.map(v => Math.abs(v)), 1);

  wrap.innerHTML = `<div class="feature-chart-wrap">` +
    vec.slice(0, 22).map((v, i) => {
      const pct  = Math.min(Math.abs(v) / maxAbs * 100, 100).toFixed(1);
      const name = labels[i] || `feat_${i+1}`;
      const color = v < 0 ? 'var(--red)' : 'linear-gradient(90deg, var(--accent), var(--green))';
      return `
        <div class="feature-row">
          <div class="feature-name">${name}</div>
          <div class="feature-bar-bg">
            <div class="feature-bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="feature-val">${v.toFixed(3)}</div>
        </div>`;
    }).join('') + `</div>`;
}

function renderC4Metrics(c4) {
  const grid = document.getElementById('c4Metrics');
  if (!grid) return;

  const metrics = [
    { label: 'Verdict',          val: c4.verdict,            fmt: v => v,
      cls: v => v === 'TROJAN' ? 'highlight-bad' : 'highlight-good' },
    { label: 'Anomaly Score',    val: c4.anomaly_score,      fmt: v => `${(v*100).toFixed(1)}%` },
    { label: 'Certified Radius', val: c4.certified_radius,   fmt: v => v.toFixed(4) },
    { label: 'Mean Stability',   val: c4.mean_stability,     fmt: v => `${(v*100).toFixed(1)}%` },
    { label: 'Min Stability',    val: c4.min_stability,      fmt: v => `${(v*100).toFixed(1)}%` },
    { label: 'Stable Flag',      val: c4.stable_flag,        fmt: v => v ? '✓ YES' : '✗ NO',
      cls: v => v ? 'highlight-good' : 'highlight-bad' },
    { label: 'High Stability',   val: c4.high_stability_flag,fmt: v => v ? '✓ YES' : '✗ NO',
      cls: v => v ? 'highlight-good' : 'highlight-bad' },
    { label: 'IF Model Used',    val: c4.used_trained_model, fmt: v => v ? 'YES' : 'NO (heuristic)' },
  ];

  grid.innerHTML = metrics.filter(m => m.val != null).map(m => {
    const val = m.fmt(m.val);
    const cls = m.cls ? m.cls(m.val) : '';
    return `
      <div class="metric-card">
        <div class="metric-card-label">${m.label}</div>
        <div class="metric-card-val ${cls}">${val}</div>
      </div>`;
  }).join('');
}

// ── Result Tabs ───────────────────────────────────────────────
function showResultTab(id, btn) {
  document.querySelectorAll('.rtab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.rtab').forEach(el => el.classList.remove('active'));
  document.getElementById(`rtab-${id}`)?.classList.add('active');
  btn?.classList.add('active');
}

// ── Reset ─────────────────────────────────────────────────────
function resetAnalysis() {
  currentResults = null;
  showState('idle');
  clearImage();
  resetSteps();
  setProgress(0);
}

// ── Export ────────────────────────────────────────────────────
function exportResults() {
  if (!currentResults) return;
  const blob = new Blob([JSON.stringify(currentResults, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `neuroshield_${currentResults.session_id || 'results'}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
