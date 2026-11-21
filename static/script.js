document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('uploadForm');
  const imageInput = document.getElementById('imageInput');
  const previewImage = document.getElementById('previewImage');
  const previewSection = document.getElementById('preview');
  const loadingSection = document.getElementById('loading');
  const resultsSection = document.getElementById('results');
  const analysisResults = document.getElementById('analysisResults');
  const submitButton = form.querySelector('button[type="submit"]');

  // Image preview functionality (DICOM files cannot be previewed by browsers)
  imageInput.addEventListener('change', () => {
    const file = imageInput.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = e => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        previewSection.style.display = 'block';
      };
      reader.readAsDataURL(file);
    } else {
      previewImage.style.display = 'none';
      previewSection.style.display = 'none';
    }
  });

  // Form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!imageInput.files[0]) {
      alert('Please select an image file.');
      return;
    }

    // Show loading, hide results
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    submitButton.disabled = true;
    submitButton.textContent = 'Analyzing...';

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (response.ok) {
        displayResults(data);
      } else {
        displayError(data.error || 'An error occurred while analyzing the image.');
      }
    } catch (error) {
      console.error('Error:', error);
      displayError('Network error. Please try again.');
    } finally {
      // Hide loading, restore button
      loadingSection.style.display = 'none';
      submitButton.disabled = false;
      submitButton.textContent = 'Analyze Image';
    }
  });


  function displayResults(data) {
    // Display analysis results
    if (data.analysis) {
      displayAnalysis(data.analysis);
    }

    // Display prediction result
    const predictionResult = document.getElementById('predictionResult');
    if (data.model_prediction !== undefined) {
      const predText = data.predicted_label || data.model_prediction;
      const description = data.prediction_description || '';
      predictionResult.innerHTML =
        `<span class="prediction-value">${escapeHtml(predText)}</span>` +
        (description
          ? `<div class="prediction-description">${escapeHtml(description)}</div>`
          : '');
      document.getElementById('predictionSection').style.display = 'block';
    } else {
      predictionResult.innerHTML = '';
      document.getElementById('predictionSection').style.display = 'none';
    }

    displayProbabilities(data.probabilities);

    // Show results section
    resultsSection.style.display = 'block';
  }

  function displayProbabilities(probabilities) {
    const section = document.getElementById('probabilitiesSection');
    const container = document.getElementById('probabilitiesResult');

    if (!Array.isArray(probabilities) || probabilities.length === 0) {
      section.style.display = 'none';
      container.innerHTML = '';
      return;
    }

    container.innerHTML = probabilities.map(item => {
      const percentage = Math.max(0, Math.min(100, item.probability * 100));
      const label = item.label || item.class_name || 'Unknown';
      return `
        <div class="probability-row">
          <div class="probability-label">
            <span>${escapeHtml(label)}</span>
            <span>${percentage.toFixed(1)}%</span>
          </div>
          <div class="probability-bar">
            <div class="probability-fill" style="width: ${percentage.toFixed(1)}%"></div>
          </div>
        </div>
      `;
    }).join('');

    section.style.display = 'block';
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function displayAnalysis(analysis) {
    let html = '';

    // Statistics
    if (analysis.mean_intensity !== undefined) {
      html += `
        <div class="analysis-stats">
          <div class="stat-card">
            <div class="stat-value">${analysis.mean_intensity.toFixed(1)}</div>
            <div class="stat-label">Mean Intensity</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${analysis.std_intensity.toFixed(1)}</div>
            <div class="stat-label">Standard Deviation</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">${analysis.edge_count}</div>
            <div class="stat-label">Edge Structures</div>
          </div>
        </div>
      `;
    }

    // Findings
    if (analysis.findings && analysis.findings.length > 0) {
      html += '<h4>Analysis Findings:</h4>';
      html += '<ul class="findings-list">';
      
      analysis.findings.forEach(finding => {
        const isWarning = finding.includes('⚠️') || finding.includes('IMPORTANT') || finding.includes('Consult');
        const className = isWarning ? 'warning' : '';
        html += `<li class="${className}">${escapeHtml(finding)}</li>`;
      });
      
      html += '</ul>';
    }

    // Disclaimer
    if (analysis.disclaimer) {
      html += `<div class="disclaimer"><p><strong>Disclaimer:</strong> ${escapeHtml(analysis.disclaimer)}</p></div>`;
    }

    analysisResults.innerHTML = html;
  }

  function displayError(message) {
    analysisResults.innerHTML = `<div class="error"><strong>Error:</strong> ${escapeHtml(message)}</div>`;
    document.getElementById('probabilitiesSection').style.display = 'none';
    document.getElementById('predictionSection').style.display = 'none';
    resultsSection.style.display = 'block';
  }
});
