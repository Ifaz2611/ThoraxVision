/* ==========================================================================
   ThoraxVision — Minimal Modern Script
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const form = document.getElementById('uploadForm');
  const imageInput = document.getElementById('imageInput');
  const dropzone = document.getElementById('dropzone');
  const browseTrigger = document.getElementById('browseTrigger');
  const sampleBtns = document.querySelectorAll('.sample-btn');

  // Viewport & Results
  const workspace = document.getElementById('workspace');
  const previewImage = document.getElementById('previewImage');
  const fileMeta = document.getElementById('fileMeta');
  const loading = document.getElementById('loading');
  const errorBox = document.getElementById('errorBox');
  const results = document.getElementById('results');

  // Prediction Fields
  const predictionSection = document.getElementById('predictionSection');
  const predictionResult = document.getElementById('predictionResult');
  const predictionBadge = document.getElementById('predictionBadge');
  const confidenceText = document.getElementById('confidenceText');
  const predictionDescription = document.getElementById('predictionDescription');
  const probabilitiesSection = document.getElementById('probabilitiesSection');
  const probabilitiesResult = document.getElementById('probabilitiesResult');
  const analysisResults = document.getElementById('analysisResults');

  // Contrast Buttons
  const contrastBtns = document.querySelectorAll('.contrast-btn');

  // Action Buttons
  const copyReportBtn = document.getElementById('copyReportBtn');
  const resetBtn = document.getElementById('resetBtn');
  const toast = document.getElementById('toast');
  const toastMsg = document.getElementById('toastMsg');

  // State
  let lastAnalysis = null;
  let activeFilename = 'scan.png';

  // --------------------------------------------------------------------------
  // Health Check
  // --------------------------------------------------------------------------
  async function loadModelInfo() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        const modelEl = document.getElementById('modelStatus');
        if (data.model_type) {
          const formatted = data.model_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
          modelEl.textContent = `${formatted} (${(data.classes || []).length} classes)`;
        }
      }
    } catch (_) {}
  }
  loadModelInfo();

  // --------------------------------------------------------------------------
  // File Drop & Selection
  // --------------------------------------------------------------------------
  browseTrigger.addEventListener('click', e => {
    e.stopPropagation();
    imageInput.click();
  });

  dropzone.addEventListener('click', () => {
    imageInput.click();
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-active');
    });
  });

  ['dragleave', 'dragend', 'drop'].forEach(name => {
    dropzone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-active');
    });
  });

  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      imageInput.files = files;
      handleFile(files[0]);
    }
  });

  imageInput.addEventListener('change', () => {
    if (imageInput.files && imageInput.files[0]) {
      handleFile(imageInput.files[0]);
    }
  });

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function handleFile(file) {
    activeFilename = file.name;
    sampleBtns.forEach(b => b.classList.remove('active'));

    const ext = file.name.split('.').pop().toLowerCase();
    const isDicom = ext === 'dcm' || ext === 'dicom' || file.type === 'application/dicom';

    fileMeta.textContent = `${file.name} · ${formatBytes(file.size)}`;

    if (isDicom) {
      previewImage.style.display = 'none';
      workspace.style.display = 'grid';
    } else {
      const reader = new FileReader();
      reader.onload = e => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        workspace.style.display = 'grid';
      };
      reader.readAsDataURL(file);
    }

    submitForm();
  }

  // --------------------------------------------------------------------------
  // Sample Cases
  // --------------------------------------------------------------------------
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      sampleBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const fileName = btn.dataset.file;
      try {
        const res = await fetch(`/static/samples/${fileName}`);
        if (!res.ok) throw new Error('Sample not found');
        const blob = await res.blob();
        const file = new File([blob], fileName, { type: 'image/png' });

        const dt = new DataTransfer();
        dt.items.add(file);
        imageInput.files = dt.files;

        handleFile(file);
      } catch (err) {
        showError('Could not load sample case.');
      }
    });
  });

  // --------------------------------------------------------------------------
  // Submit & Analysis
  // --------------------------------------------------------------------------
  form.addEventListener('submit', e => {
    e.preventDefault();
    submitForm();
  });

  async function submitForm() {
    if (!imageInput.files || !imageInput.files[0]) return;

    loading.style.display = 'flex';
    errorBox.style.display = 'none';
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
      const res = await fetch('/upload', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();

      if (res.ok) {
        lastAnalysis = data;
        if (data.dicom_preview) {
          previewImage.src = data.dicom_preview;
          previewImage.style.display = 'block';
        }
        renderResults(data);
      } else {
        showError(data.error || 'Failed to analyze image.');
      }
    } catch (err) {
      showError('Network error connecting to analysis service.');
    } finally {
      loading.style.display = 'none';
    }
  }

  // --------------------------------------------------------------------------
  // Render Results
  // --------------------------------------------------------------------------
  function renderResults(data) {
    const isNormal = (data.model_prediction || '').toLowerCase() === 'normal';
    const predLabel = data.predicted_label || data.model_prediction || 'Unknown';

    predictionResult.textContent = predLabel;

    if (isNormal) {
      predictionBadge.className = 'status-pill status-clean';
      predictionBadge.textContent = 'Normal';
    } else {
      predictionBadge.className = 'status-pill status-alert';
      predictionBadge.textContent = 'Malignancy';
    }

    // Top Probability
    let topProb = 0;
    if (Array.isArray(data.probabilities)) {
      const match = data.probabilities.find(p => p.class_name === data.model_prediction);
      topProb = (match ? match.probability : data.probabilities[0]?.probability || 0) * 100;
    }
    confidenceText.textContent = `${topProb.toFixed(1)}% confidence`;

    if (data.prediction_description) {
      predictionDescription.textContent = data.prediction_description;
      predictionDescription.style.display = 'block';
    } else {
      predictionDescription.style.display = 'none';
    }

    // Probabilities
    renderProbabilities(data.probabilities, data.model_prediction);

    // Radiomic Metrics & Findings
    if (data.analysis) {
      renderAnalysis(data.analysis);
    }

    predictionSection.style.display = 'flex';
    results.style.display = 'flex';
  }

  function renderProbabilities(probabilities, predictedClass) {
    if (!Array.isArray(probabilities) || probabilities.length === 0) {
      probabilitiesSection.style.display = 'none';
      return;
    }

    probabilitiesResult.innerHTML = probabilities.map((p, idx) => {
      const pct = Math.max(0, Math.min(100, p.probability * 100));
      const isTop = p.class_name === predictedClass || idx === 0;

      return `
        <div class="prob-row ${isTop ? 'top' : ''}">
          <div class="prob-meta">
            <span class="prob-name">${escapeHtml(p.label || p.class_name)}</span>
            <span class="prob-percent">${pct.toFixed(1)}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-bar" style="width: ${pct.toFixed(1)}%"></div>
          </div>
        </div>
      `;
    }).join('');

    probabilitiesSection.style.display = 'flex';
  }

  function renderAnalysis(analysis) {
    let html = '';

    if (analysis.mean_intensity !== undefined) {
      html += `
        <div class="stats-grid">
          <div class="stat-item">
            <span class="stat-val">${analysis.mean_intensity.toFixed(1)}</span>
            <span class="stat-lbl">Mean Intensity</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">${analysis.std_intensity.toFixed(1)}</span>
            <span class="stat-lbl">Std Deviation</span>
          </div>
          <div class="stat-item">
            <span class="stat-val">${analysis.edge_count}</span>
            <span class="stat-lbl">Edge Count</span>
          </div>
        </div>
      `;
    }

    if (analysis.findings && analysis.findings.length > 0) {
      const clean = analysis.findings.filter(f => !f.includes('IMPORTANT') && !f.includes('Consult'));
      if (clean.length > 0) {
        html += '<ul class="findings-list">';
        clean.forEach(f => {
          html += `<li class="finding-bullet">${escapeHtml(f.replace(/⚠️/g, '').trim())}</li>`;
        });
        html += '</ul>';
      }
    }

    analysisResults.innerHTML = html;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = 'block';
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // --------------------------------------------------------------------------
  // Contrast Presets
  // --------------------------------------------------------------------------
  contrastBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      contrastBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const filter = btn.dataset.filter;
      previewImage.className = `filter-${filter}`;
    });
  });

  // --------------------------------------------------------------------------
  // Actions: Copy & Clear
  // --------------------------------------------------------------------------
  copyReportBtn.addEventListener('click', () => {
    if (!lastAnalysis) return;

    const d = lastAnalysis;
    const predLabel = d.predicted_label || d.model_prediction;
    let txt = `ThoraxVision Report\n`;
    txt += `File: ${activeFilename}\n`;
    txt += `Prediction: ${predLabel}\n`;
    if (d.prediction_description) txt += `Summary: ${d.prediction_description}\n`;

    if (Array.isArray(d.probabilities)) {
      txt += `\nProbabilities:\n`;
      d.probabilities.forEach(p => {
        txt += `  ${p.label || p.class_name}: ${(p.probability * 100).toFixed(1)}%\n`;
      });
    }

    if (d.analysis) {
      txt += `\nStatistics:\n`;
      txt += `  Mean: ${d.analysis.mean_intensity?.toFixed(1) || 'N/A'}\n`;
      txt += `  Std Dev: ${d.analysis.std_intensity?.toFixed(1) || 'N/A'}\n`;
      txt += `  Edges: ${d.analysis.edge_count || 'N/A'}\n`;
    }

    navigator.clipboard.writeText(txt).then(() => {
      showToast('Copied summary to clipboard');
    }).catch(() => {
      showToast('Failed to copy');
    });
  });

  function clearScan() {
    form.reset();
    imageInput.value = '';
    previewImage.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    previewImage.style.display = 'none';
    workspace.style.display = 'none';
    results.style.display = 'none';
    errorBox.style.display = 'none';
    fileMeta.textContent = '';
    sampleBtns.forEach(b => b.classList.remove('active'));
    contrastBtns[0].click();
    lastAnalysis = null;
  }

  resetBtn.addEventListener('click', clearScan);

  function showToast(msg) {
    toastMsg.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.display = 'none';
    }, 2500);
  }
});
