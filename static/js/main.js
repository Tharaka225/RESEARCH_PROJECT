/* ================================================
   NeuroShield — main.js
   Navigation + Scanner + Contact logic
   ================================================ */

/* ── Navigation ─────────────────────────────── */
function showPage(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const page = document.getElementById(pageId);
  if (page) page.classList.add('active');
  const navBtn = document.querySelector(`[data-page="${pageId}"]`);
  if (navBtn) navBtn.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ── Tab switcher (model source) ────────────── */
function switchTab(tab) {
  const demo   = document.getElementById('demoPanel');
  const upload = document.getElementById('uploadPanel');
  const btnDemo   = document.getElementById('tabDemo');
  const btnUpload = document.getElementById('tabUpload');
  if (tab === 'demo') {
    demo.style.display = 'block';
    upload.style.display = 'none';
    btnDemo.classList.add('active');
    btnUpload.classList.remove('active');
  } else {
    demo.style.display = 'none';
    upload.style.display = 'block';
    btnUpload.classList.add('active');
    btnDemo.classList.remove('active');
  }
}

/* ── Result tabs ────────────────────────────── */
function showResultTab(tabId, btn) {
  document.querySelectorAll('.rtab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.rtab').forEach(b => b.classList.remove('active'));
  document.getElementById('rtab-' + tabId).classList.add('active');
  btn.classList.add('active');
}

/* ── Image upload & drag-drop ────────────────── */
let imageFile = null;

function setupImageUpload() {
  const dropZone = document.getElementById('imageDropZone');
  const fileInput = document.getElementById('imageInput');

  if (!dropZone || !fileInput) return;

  // Click
  dropZone.addEventListener('click', () => {
    if (!document.getElementById('imagePreview').style.display ||
        document.getElementById('imagePreview').style.display === 'none') {
      fileInput.click();
    }
  });

  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleImageFile(e.target.files[0]);
  });

  // Drag-drop
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragging');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleImageFile(e.dataTransfer.files[0]);
  });

  document.getElementById('clearImageBtn').addEventListener('click', e => {
    e.stopPropagation();
    clearImage();
  });
}

function handleImageFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid image file.');
    return;
  }
  imageFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('previewImg').src = ev.target.result;
    document.getElementById('imageInfo').textContent =
      `${file.name}  ·  ${(file.size / 1024).toFixed(1)} KB`;
    document.getElementById('uploadContent').style.display = 'none';
    document.getElementById('imagePreview').style.display = 'block';
    checkReady();
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  imageFile = null;
  document.getElementById('previewImg').src = '';
  document.getElementById('imageInfo').textContent = '';
  document.getElementById('uploadContent').style.display = 'block';
  document.getElementById('imagePreview').style.display = 'none';
  document.getElementById('imageInput').value = '';
  checkReady();
}

/* ── Model upload ────────────────────────────── */
let modelFile = null;

function setupModelUpload() {
  const input = document.getElementById('modelInput');
  const dropZone = document.getElementById('modelDropZone');
  if (!input || !dropZone) return;

  input.addEventListener('change', e => {
    if (e.target.files[0]) handleModelFile(e.target.files[0]);
  });
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragging');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleModelFile(e.dataTransfer.files[0]);
  });
}

function handleModelFile(file) {
  modelFile = file;
  document.getElementById('modelUploadContent').style.display = 'none';
  document.getElementById('modelFilename').textContent = file.name;
  document.getElementById('modelLoaded').style.display = 'flex';
  checkReady();
}

function checkReady() {
  const btn = document.getElementById('analyzeBtn');
  if (!btn) return;
  const tabUpload = document.getElementById('tabUpload');
  const isUploadTab = tabUpload && tabUpload.classList.contains('active');
  btn.disabled = !(imageFile && (!isUploadTab || modelFile));
}

/* ── Analysis Simulation ─────────────────────── */
let lastResults = null;

function startAnalysis() {
  if (!imageFile) return;

  document.getElementById('idleState').style.display = 'none';
  document.getElementById('progressState').style.display = 'block';
  document.getElementById('resultsState').style.display = 'none';
  document.getElementById('analyzeBtn').disabled = true;

  const steps = [
    { id: 'step1', statusId: 'status1', label: 'SCANNING...', duration: 1600 },
    { id: 'step2', statusId: 'status2', label: 'ATTACKING...', duration: 2000 },
    { id: 'step3', statusId: 'status3', label: 'PROFILING...', duration: 1400 },
    { id: 'step4', statusId: 'status4', label: 'CLASSIFYING...', duration: 1200 },
  ];

  let elapsed = 0;
  const totalDuration = steps.reduce((a, s) => a + s.duration, 0);

  // Reset steps
  steps.forEach(s => {
    const el = document.getElementById(s.id);
    el.classList.remove('active', 'done');
    document.getElementById(s.statusId).textContent = 'PENDING';
  });
  setProgress(0);

  function runStep(idx) {
    if (idx >= steps.length) {
      setTimeout(showResults, 400);
      return;
    }
    const s = steps[idx];
    const el = document.getElementById(s.id);
    if (idx > 0) {
      const prev = steps[idx - 1];
      document.getElementById(prev.id).classList.remove('active');
      document.getElementById(prev.id).classList.add('done');
      document.getElementById(prev.statusId).textContent = 'COMPLETE';
    }
    el.classList.add('active');
    document.getElementById(s.statusId).textContent = s.label;

    // Progress ticking
    const startPct = (elapsed / totalDuration) * 100;
    const endPct   = ((elapsed + s.duration) / totalDuration) * 100;
    animateProgress(startPct, endPct, s.duration);
    elapsed += s.duration;

    setTimeout(() => runStep(idx + 1), s.duration);
  }

  runStep(0);
}

function animateProgress(from, to, duration) {
  const start = performance.now();
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const val = from + (to - from) * t;
    setProgress(val);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function setProgress(pct) {
  const bar = document.getElementById('progressBar');
  const lbl = document.getElementById('progressLabel');
  if (bar) bar.style.width = pct + '%';
  if (lbl) lbl.textContent = Math.round(pct) + '%';
}

function showResults() {
  const isThreat = Math.random() > 0.45;
  const anomalyScore = isThreat
    ? (0.65 + Math.random() * 0.34).toFixed(3)
    : (0.05 + Math.random() * 0.35).toFixed(3);
  const stability = (0.4 + Math.random() * 0.55).toFixed(3);
  const certRadius = (0.02 + Math.random() * 0.15).toFixed(3);

  // Component 1 (white-box)
  const fgsm_delta = (0.01 + Math.random() * 0.3).toFixed(4);
  const ifgsm_delta = (parseFloat(fgsm_delta) * (1 + Math.random() * 0.5)).toFixed(4);
  const gradient_norm = (Math.random() * 5).toFixed(3);
  const confidence_drop = (Math.random() * 50).toFixed(1);

  // Component 2 (black-box)
  const noise_asr = (isThreat ? 0.5 + Math.random() * 0.5 : Math.random() * 0.4).toFixed(3);
  const boundary_dist = (0.01 + Math.random() * 0.5).toFixed(4);
  const query_count = Math.floor(200 + Math.random() * 800);
  const bb_confidence = (Math.random()).toFixed(3);

  // Features (22-dim, show top 8)
  const featureNames = [
    'grad_magnitude', 'loss_sensitivity', 'fgsm_shift',
    'ifgsm_shift', 'noise_asr', 'boundary_gap',
    'activation_std', 'layer_divergence',
  ];
  const featureVals = featureNames.map(() => Math.random());

  // Component 4
  const isolation_score = (isThreat ? -0.5 - Math.random() * 0.5 : Math.random() * 0.5 - 0.5).toFixed(3);
  const gauss_z = (Math.random() * 4).toFixed(3);
  const outlier = isThreat ? 'YES' : 'NO';

  lastResults = {
    verdict: isThreat ? 'TROJAN DETECTED' : 'MODEL CLEAN',
    threat: isThreat,
    anomalyScore, stability, certRadius,
    c1: { fgsm_delta, ifgsm_delta, gradient_norm, confidence_drop },
    c2: { noise_asr, boundary_dist, query_count, bb_confidence },
    features: featureNames.map((n, i) => ({ name: n, val: featureVals[i] })),
    c4: { isolation_score, gauss_z, outlier },
  };

  // Populate UI
  const banner = document.getElementById('verdictBanner');
  banner.className = 'verdict-banner ' + (isThreat ? 'threat' : 'clear');
  document.getElementById('verdictText').textContent = lastResults.verdict;
  document.getElementById('anomalyScore').textContent = anomalyScore;
  document.getElementById('stabilityVal').textContent = stability;
  document.getElementById('certRadius').textContent = certRadius;

  // C1
  renderMetrics('c1Metrics', [
    { label: 'FGSM δ', val: fgsm_delta, sub: 'L∞ perturbation magnitude' },
    { label: 'I-FGSM δ', val: ifgsm_delta, sub: 'Iterative shift' },
    { label: 'GRADIENT NORM', val: gradient_norm, sub: '' },
    { label: 'CONFIDENCE DROP', val: confidence_drop + '%', sub: 'Post-attack' },
  ]);

  // C2
  renderMetrics('c2Metrics', [
    { label: 'NOISE ASR', val: noise_asr, sub: 'Attack success rate' },
    { label: 'BOUNDARY DIST', val: boundary_dist, sub: 'HopSkipJump' },
    { label: 'QUERY COUNT', val: query_count, sub: 'Total queries' },
    { label: 'BB CONFIDENCE', val: bb_confidence, sub: '' },
  ]);

  // Features
  const fc = document.getElementById('featureChart');
  fc.innerHTML = featureNames.map((n, i) => {
    const pct = (featureVals[i] * 100).toFixed(1);
    return `<div class="fbar-row">
      <span class="fbar-label">${n}</span>
      <div class="fbar-track"><div class="fbar-fill" style="width:${pct}%"></div></div>
      <span class="fbar-val">${featureVals[i].toFixed(3)}</span>
    </div>`;
  }).join('');

  // C4
  renderMetrics('c4Metrics', [
    { label: 'ISOLATION SCORE', val: isolation_score, sub: 'Negative = outlier' },
    { label: 'GAUSSIAN Z', val: gauss_z, sub: 'Std deviations' },
    { label: 'OUTLIER', val: outlier, sub: 'Isolation Forest' },
    { label: 'THREAT LEVEL', val: isThreat ? 'HIGH' : 'LOW', sub: '' },
  ]);

  document.getElementById('progressState').style.display = 'none';
  document.getElementById('resultsState').style.display = 'block';
  document.getElementById('analyzeBtn').disabled = false;
}

function renderMetrics(containerId, metrics) {
  document.getElementById(containerId).innerHTML = metrics.map(m => `
    <div class="metric-card">
      <div class="metric-card-label">${m.label}</div>
      <div class="metric-card-val">${m.val}</div>
      ${m.sub ? `<div class="metric-card-sub">${m.sub}</div>` : ''}
    </div>`).join('');
}

function resetAnalysis() {
  clearImage();
  document.getElementById('idleState').style.display = 'flex';
  document.getElementById('progressState').style.display = 'none';
  document.getElementById('resultsState').style.display = 'none';
  setProgress(0);
  lastResults = null;
}

function exportResults() {
  if (!lastResults) return;
  const blob = new Blob([JSON.stringify(lastResults, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `neuroshield_${Date.now()}.json`;
  a.click();
}

/* ── Contact Form ────────────────────────────── */
function setupContactForm() {
  const form = document.getElementById('contactForm');
  if (!form) return;
  form.addEventListener('submit', e => {
    e.preventDefault();
    document.getElementById('formFields').style.display = 'none';
    document.getElementById('formSuccess').style.display = 'block';
  });
}

/* ── Init ────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  showPage('page-home');
  setupImageUpload();
  setupModelUpload();
  setupContactForm();

  // Nav buttons
  document.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => showPage(btn.dataset.page));
  });

  // CTA button on home
  const ctaBtn = document.getElementById('ctaScanner');
  if (ctaBtn) ctaBtn.addEventListener('click', () => showPage('page-scanner'));
  const ctaContact = document.getElementById('ctaContact');
  if (ctaContact) ctaContact.addEventListener('click', () => showPage('page-contact'));
});