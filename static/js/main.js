/* NeuroShield — Frontend Controller */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  imageFile: null,
  modelId: "demo",
  lastResults: null,
};

// ── Init ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  setupDragDrop("imageDropZone", "imageInput", handleImageFile);
  setupDragDrop("modelDropZone", "modelInput", handleModelFile);
  document.getElementById("imageInput").addEventListener("change", e =>
    e.target.files[0] && handleImageFile(e.target.files[0])
  );
  document.getElementById("modelInput").addEventListener("change", e =>
    e.target.files[0] && handleModelFile(e.target.files[0])
  );
  document.getElementById("clearImageBtn").addEventListener("click", clearImage);
  checkHealth();
});

// ── Drag & Drop ───────────────────────────────────────────────────────────────

function setupDragDrop(zoneId, inputId, handler) {
  const zone = document.getElementById(zoneId);
  if (!zone) return;

  zone.addEventListener("click", () => document.getElementById(inputId).click());
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handler(file);
  });
}

// ── Image Handling ────────────────────────────────────────────────────────────

function handleImageFile(file) {
  state.imageFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById("previewImg");
    const info = document.getElementById("imageInfo");
    img.src = e.target.result;
    img.onload = () => {
      info.textContent = `${img.naturalWidth}×${img.naturalHeight} · ${formatBytes(file.size)} · ${file.type}`;
    };
    document.getElementById("uploadContent").style.display = "none";
    document.getElementById("imagePreview").style.display  = "flex";
  };
  reader.readAsDataURL(file);
  updateAnalyzeBtn();
}

function clearImage() {
  state.imageFile = null;
  document.getElementById("uploadContent").style.display = "block";
  document.getElementById("imagePreview").style.display  = "none";
  document.getElementById("imageInput").value = "";
  updateAnalyzeBtn();
}

// ── Model Handling ────────────────────────────────────────────────────────────

function switchTab(tab) {
  document.getElementById("tabDemo").classList.toggle("active", tab === "demo");
  document.getElementById("tabUpload").classList.toggle("active", tab === "upload");
  document.getElementById("demoPanel").style.display   = tab === "demo"   ? "block" : "none";
  document.getElementById("uploadPanel").style.display = tab === "upload" ? "block" : "none";

  if (tab === "demo") {
    state.modelId = "demo";
    updateAnalyzeBtn();
  }
}

async function handleModelFile(file) {
  if (!file.name.match(/\.(pt|pth)$/i)) {
    alert("Please select a .pt or .pth model file.");
    return;
  }

  const formData = new FormData();
  formData.append("model", file);

  try {
    const resp = await fetch("/api/upload-model", { method: "POST", body: formData });
    const data = await resp.json();

    if (data.error) throw new Error(data.error);

    state.modelId = data.model_id;
    document.getElementById("modelUploadContent").style.display = "none";
    document.getElementById("modelLoaded").style.display        = "flex";
    document.getElementById("modelFilename").textContent        = data.filename;
    updateAnalyzeBtn();
  } catch (err) {
    alert("Model upload failed: " + err.message);
  }
}

// ── Analysis ──────────────────────────────────────────────────────────────────

async function startAnalysis() {
  if (!state.imageFile) return;

  showProgressState();

  const formData = new FormData();
  formData.append("image", state.imageFile);
  formData.append("model_id", state.modelId);
  formData.append("image_size", document.getElementById("imageSizeSelect").value);
  formData.append("architecture", document.getElementById("architectureSelect")?.value || "");
  formData.append("num_classes", document.getElementById("numClasses")?.value || "10");

  // Animate steps in sequence while waiting
  animateSteps();

  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await resp.json();

    if (data.error) throw new Error(data.error);

    state.lastResults = data;
    showResults(data);
  } catch (err) {
    showError(err.message);
  }
}

let stepTimers = [];

function animateSteps() {
  stepTimers.forEach(clearTimeout);
  stepTimers = [];

  const delays = [0, 1200, 2600, 4200]; // approximate timing hints
  delays.forEach((delay, i) => {
    stepTimers.push(setTimeout(() => activateStep(i + 1), delay));
  });
}

function activateStep(n) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`step${i}`);
    const st = document.getElementById(`status${i}`);
    if (i < n) {
      el.classList.remove("active");
      el.classList.add("done");
      st.textContent = "COMPLETE ✓";
    } else if (i === n) {
      el.classList.add("active");
      el.classList.remove("done");
      st.textContent = "RUNNING...";
    } else {
      el.classList.remove("active", "done");
      st.textContent = "PENDING";
    }
  }

  // Progress bar hint
  const pct = [10, 35, 65, 80][n - 1];
  setProgress(pct);
}

function setProgress(pct) {
  document.getElementById("progressBar").style.width  = pct + "%";
  document.getElementById("progressLabel").textContent = pct + "%";
}

// ── Render Results ────────────────────────────────────────────────────────────

function showResults(data) {
  // Complete all steps
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById(`step${i}`);
    const st = document.getElementById(`status${i}`);
    el.classList.add("done");
    el.classList.remove("active");
    st.textContent = "COMPLETE ✓";
  }
  setProgress(100);

  setTimeout(() => {
    document.getElementById("progressState").style.display = "none";
    document.getElementById("resultsState").style.display  = "block";

    const c4 = data.components.component4;
    const verdict = c4.verdict;

    // Verdict banner
    const banner = document.getElementById("verdictBanner");
    banner.className = "verdict-banner " + verdict.toLowerCase();
    document.getElementById("verdictText").textContent = verdict;
    document.getElementById("anomalyScore").textContent =
      (c4.anomaly_score * 100).toFixed(1) + "%";
    document.getElementById("stabilityVal").textContent =
      (c4.mean_stability * 100).toFixed(1) + "%";
    document.getElementById("certRadius").textContent =
      c4.certified_radius.toFixed(4);

    // ── NEW: Render Component 1 ────────────────────────────────────────
    renderComponent1(data.components.component1);

    // ── Other components (unchanged) ───────────────────────────────────
    renderMetrics("c1Metrics", data.components.component1, [
      ["delta_fgsm", "δ FGSM"],
      ["delta_ifgsm", "δ I-FGSM"],
      ["min_flip_epsilon", "Min Flip ε"],
      ["original_conf", "Original Conf"],
      ["fgsm_conf", "Post-FGSM Conf"],
      ["ifgsm_conf", "Post-IFGSM Conf"],
      ["gradient_norm", "Gradient Norm"],
      ["gradient_variance", "Grad Variance"],
      ["conf_margin", "Conf Margin"],
      ["loss_sensitivity", "Loss Value"],
    ]);

    renderMetrics("c2Metrics", data.components.component2, [
      ["delta_blackbox", "δ Black-Box"],
      ["noise_delta", "Noise Delta"],
      ["hsj_delta", "HSJ Delta"],
      ["fd_sensitivity", "FD Sensitivity"],
      ["mean_conf_drop", "Avg Conf Drop"],
      ["max_conf_drop", "Max Conf Drop"],
      ["conf_drop_variance", "Drop Variance"],
    ]);

    renderFeatureChart(data.components.component3);

    renderMetrics("c4Metrics", data.components.component4, [
      ["verdict", "Verdict"],
      ["anomaly_score", "Anomaly Score"],
      ["certified_radius", "Certified Radius"],
      ["mean_stability", "Mean Stability"],
      ["min_stability", "Min Stability"],
      ["stable_flag", "Stable Flag"],
      ["high_stability_flag", "High Stability"],
      ["used_trained_model", "Used IF Model"],
    ]);
  }, 500);
}

// ── NEW: Dedicated render function for Component 1 ────────────────────────
function renderComponent1(c1) {
  // Original prediction & confidence
  const origClassEl = document.getElementById("c1OriginalClass");
  const origConfEl  = document.getElementById("c1OriginalConf");

  origClassEl.textContent = c1.original_class_name || `Class ${c1.original_pred}`;
  origConfEl.textContent  = (c1.original_conf * 100).toFixed(1) + "%";

  // Noise stability table
  const tbody = document.getElementById("c1NoiseTableBody");
  tbody.innerHTML = "";

  if (c1.noise_test_results && Array.isArray(c1.noise_test_results) && c1.noise_test_results.length > 0) {
    c1.noise_test_results.forEach(item => {
      const row = document.createElement("tr");

      const sigmaCell = document.createElement("td");
      sigmaCell.textContent = item.noise_sigma;

      const classCell = document.createElement("td");
      classCell.textContent = item.predicted_class;

      const changedCell = document.createElement("td");
      changedCell.textContent = item.changed ? "Yes" : "No";
      changedCell.className = item.changed ? "changed yes" : "no";

      row.appendChild(sigmaCell);
      row.appendChild(classCell);
      row.appendChild(changedCell);

      tbody.appendChild(row);
    });
  } else {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="3" style="text-align:center; color:#888;">No noise stability data available</td>';
    tbody.appendChild(row);
  }
}

// ── Existing render functions (unchanged) ─────────────────────────────────

function renderMetrics(containerId, data, fields) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  fields.forEach(([key, label]) => {
    if (!(key in data)) return;
    const val = data[key];

    const card = document.createElement("div");
    card.className = "metric-card";

    const lEl = document.createElement("div");
    lEl.className = "metric-card-label";
    lEl.textContent = label.toUpperCase();

    const vEl = document.createElement("div");
    vEl.className = "metric-card-value";

    if (typeof val === "number") {
      vEl.textContent = val < 0.001 && val > 0 ? val.toExponential(3) : val.toFixed(4);
    } else if (typeof val === "boolean" || val === 0 || val === 1) {
      vEl.textContent = Boolean(val) ? "YES" : "NO";
      vEl.classList.add(val ? "changed" : "ok");
    } else {
      vEl.textContent = String(val);
      if (val === "TROJAN") vEl.style.color = "var(--trojan-color)";
      if (val === "CLEAN")  vEl.style.color = "var(--clean-color)";
    }

    card.appendChild(lEl);
    card.appendChild(vEl);
    container.appendChild(card);
  });
}

function renderFeatureChart(c3) {
  const container = document.getElementById("featureChart");
  container.innerHTML = "";

  const names  = c3.feature_names;
  const values = c3.feature_vector;

  if (!names || !values) return;

  // Normalise for bar widths
  const maxVal = Math.max(...values.map(Math.abs), 1e-8);

  names.forEach((name, i) => {
    const rawVal = values[i];
    const pct    = Math.abs(rawVal) / maxVal * 100;

    const row = document.createElement("div");
    row.className = "feat-row";

    const nameEl = document.createElement("div");
    nameEl.className = "feat-name";
    nameEl.textContent = name;
    nameEl.title = name;

    const barBg = document.createElement("div");
    barBg.className = "feat-bar-bg";

    const barFill = document.createElement("div");
    barFill.className = "feat-bar-fill";
    barFill.style.width = "0%";
    setTimeout(() => barFill.style.width = pct + "%", 50 + i * 20);

    barBg.appendChild(barFill);

    const valEl = document.createElement("div");
    valEl.className = "feat-val";
    valEl.textContent = rawVal < 0.001 && rawVal > 0
      ? rawVal.toExponential(2)
      : rawVal.toFixed(4);

    row.appendChild(nameEl);
    row.appendChild(barBg);
    row.appendChild(valEl);
    container.appendChild(row);
  });
}

function showResultTab(tab, btn) {
  document.querySelectorAll(".rtab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".rtab").forEach(el => el.classList.remove("active"));
  document.getElementById("rtab-" + tab).classList.add("active");
  btn.classList.add("active");
}

// ── UI State Helpers ──────────────────────────────────────────────────────────

function showProgressState() {
  document.getElementById("idleState").style.display    = "none";
  document.getElementById("progressState").style.display = "block";
  document.getElementById("resultsState").style.display  = "none";
  document.getElementById("analyzeBtn").disabled = true;
}

function showError(msg) {
  document.getElementById("progressState").style.display = "none";
  document.getElementById("idleState").style.display     = "block";
  document.getElementById("idleState").innerHTML = `
    <div class="idle-icon" style="color:var(--accent2)">✕</div>
    <p>Analysis Failed</p>
    <p class="idle-sub" style="color:var(--accent2)">${escapeHtml(msg)}</p>
    <button class="btn-secondary" style="margin-top:16px" onclick="resetAnalysis()">Try Again</button>
  `;
}

function resetAnalysis() {
  document.getElementById("idleState").style.display    = "block";
  document.getElementById("progressState").style.display = "none";
  document.getElementById("resultsState").style.display  = "none";
  document.getElementById("idleState").innerHTML = `
    <div class="idle-icon">◈</div>
    <p>Configure inputs and run analysis</p>
    <p class="idle-sub">System awaiting target</p>
  `;
  updateAnalyzeBtn();
  // Reset step styles
  for (let i = 1; i <= 4; i++) {
    document.getElementById(`step${i}`).classList.remove("active", "done");
    document.getElementById(`status${i}`).textContent = "PENDING";
  }
  setProgress(0);
}

function updateAnalyzeBtn() {
  const btn = document.getElementById("analyzeBtn");
  const ready = !!state.imageFile;
  btn.disabled = !ready;
  document.getElementById("analyzeBtnText").textContent = ready
    ? "▶ RUN ANALYSIS"
    : "▶ SELECT AN IMAGE FIRST";
}

// ── Export ────────────────────────────────────────────────────────────────────

function exportResults() {
  if (!state.lastResults) return;
  const blob = new Blob([JSON.stringify(state.lastResults, null, 2)],
    { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `neuroshield_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function formatBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 ** 2) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 ** 2).toFixed(2) + " MB";
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function checkHealth() {
  try {
    const resp = await fetch("/api/health");
    const data = await resp.json();
    const el = document.getElementById("systemStatus");
    if (data.status === "ok") {
      el.textContent = data.cuda ? "CUDA ONLINE" : "CPU MODE";
    }
  } catch { /* ignore */ }
}