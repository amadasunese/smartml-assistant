// Global variables
let currentDataset = null;
let currentModel = null;

// Tab functionality
document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tabs li');
  const tabContents = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      
      // Update active tab
      tabs.forEach(t => t.classList.remove('is-active'));
      tab.classList.add('is-active');
      
      // Show active content
      tabContents.forEach(content => {
        content.classList.add('is-hidden');
        if (content.id === target) {
          content.classList.remove('is-hidden');
        }
      });
    });
  });

  // File input name display
  const fileInput = document.querySelector('#fileInput');
  if (fileInput) {
    fileInput.onchange = () => {
      if (fileInput.files.length > 0) {
        const fileName = document.querySelector('#fileName');
        fileName.textContent = fileInput.files[0].name;
      }
    };
  }

  // Hide progress bar
  document.getElementById('hideProgress').addEventListener('click', () => {
    document.getElementById('progressContainer').style.display = 'none';
  });

  // Mobile menu toggle
  const mobileMenuButton = document.getElementById('mobileMenuButton');
  const sidebar = document.getElementById('sidebar');
  
  if (mobileMenuButton) {
    mobileMenuButton.addEventListener('click', () => {
      if (sidebar.style.display === 'none' || sidebar.style.display === '') {
        sidebar.style.display = 'block';
        sidebar.style.width = '250px';
      } else {
        sidebar.style.display = 'none';
        sidebar.style.width = '0';
      }
    });
  }

  // Sidebar navigation
  const sidebarLinks = document.querySelectorAll('.sidebar-menu a');
  sidebarLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const targetId = link.getAttribute('href').substring(1);
      
      // Remove active class from all links
      sidebarLinks.forEach(l => l.classList.remove('active'));
      
      // Add active class to clicked link
      link.classList.add('active');
      
      // Scroll to target section
      if (targetId) {
        const targetSection = document.getElementById(targetId);
        if (targetSection) {
          targetSection.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });
  });

  // Check if mobile view
  function checkMobileView() {
    if (window.innerWidth <= 768) {
      mobileMenuButton.style.display = 'block';
      sidebar.style.display = 'none';
      sidebar.style.width = '0';
    } else {
      mobileMenuButton.style.display = 'none';
      sidebar.style.display = 'block';
      sidebar.style.width = '250px';
    }
  }
  
  // Initial check
  checkMobileView();
  
  // Listen for resize events
  window.addEventListener('resize', checkMobileView);
  
  // Initialize all dropdowns
  refreshAllDropdowns();

  // Attach event listeners after DOM is loaded
  const uploadForm = document.getElementById("uploadForm");
  if (uploadForm) {
    uploadForm.onsubmit = async (e) => {
      e.preventDefault();
      showProgress();
      let formData = new FormData(e.target);
      
      // Add custom dataset name if provided
      const datasetName = document.getElementById('datasetName').value;
      if (datasetName) {
        formData.append('name', datasetName);
      }
      
      let res = await fetch("/upload", { method: "POST", body: formData });
      let data = await res.json();
      document.getElementById("uploadResult").innerText = JSON.stringify(data, null, 2);
      hideProgress();
      refreshAllDropdowns();
      refreshStats();
    };
  }

  const preprocessButton = document.getElementById("preprocessButton");
  if (preprocessButton) {
    preprocessButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();
      
      const datasetSelector = document.getElementById("preprocessDatasetSelector");
      const dataset = datasetSelector.value;
      
      if (!dataset) {
        alert("Please select a dataset first");
        hideProgress();
        return;
      }
      
      const formData = {
        missing_strategy: document.querySelector('select[name="missing_strategy"]').value,
        scaler: document.querySelector('select[name="scaler"]').value,
        outlier_method: document.querySelector('select[name="outlier_method"]').value,
        encoding_method: document.querySelector('select[name="encoding_method"]').value
      };
      
      let res = await fetch(`/preprocess/${dataset}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(formData)
      });
      let data = await res.json();
      document.getElementById("preprocessResult").innerText = JSON.stringify(data, null, 2);
      hideProgress();
      refreshAllDropdowns();
      refreshStats();
    });
  }

  const featureEngineeringButton = document.getElementById("featureEngineeringButton");
  if (featureEngineeringButton) {
    featureEngineeringButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();
      
      const datasetSelector = document.getElementById("featureDatasetSelector");
      const dataset = datasetSelector.value;
      
      if (!dataset) {
        alert("Please select a dataset first");
        hideProgress();
        return;
      }
      
      const formData = {
        feature_generation: document.querySelector('select[name="feature_generation"]').value,
        feature_selection: document.querySelector('select[name="feature_selection"]').value,
        dimensionality_reduction: document.querySelector('select[name="dimensionality_reduction"]').value
      };
      
      let res = await fetch(`/feature-engineering/${dataset}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(formData)
      });
      let data = await res.json();
      document.getElementById("featureEngineeringResult").innerText = JSON.stringify(data, null, 2);
      hideProgress();
      refreshAllDropdowns();
      refreshStats();
    });
  }


  // Populate target column options when dataset changes
async function updateTargetColumnOptions() {
  const datasetSelector = document.getElementById("trainDatasetSelector");
  const dataset = datasetSelector.value;
  const targetColumnSelector = document.getElementById("targetColumnSelector");

  // Reset dropdown
  targetColumnSelector.innerHTML = '<option value="">Select target column</option>';

  if (!dataset) return;

  try {
    let res = await fetch(`/dataset-columns/${dataset}`);

    if (!res.ok) throw new Error("Failed to fetch columns");

    let columns = await res.json();

    columns.forEach(col => {
      let option = document.createElement("option");
      option.value = col;
      option.textContent = col;
      targetColumnSelector.appendChild(option);
    });
  } catch (err) {
    console.error("Error loading target columns:", err);
  }
}

// Handle training
const trainButton = document.getElementById("trainButton");
if (trainButton) {
  trainButton.addEventListener("click", async (e) => {
    e.preventDefault();
    showProgress();

    const datasetSelector = document.getElementById("trainDatasetSelector");
    const dataset = datasetSelector.value;
    const targetColumn = document.getElementById("targetColumnSelector").value;

    if (!dataset || !targetColumn) {
      alert("Please select a dataset and target column first");
      hideProgress();
      return;
    }

    const formData = {
      target: targetColumn,
      algorithm: document.querySelector('select[name="algorithm"]').value,
      test_size: document.querySelector('select[name="test_size"]').value,
      cross_validation: document.querySelector('select[name="cross_validation"]').value,
      hyperparameter_tuning: document.querySelector('select[name="hyperparameter_tuning"]').value
    };

    try {
      let res = await fetch(`/train/${dataset}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(formData)
      });

      let data = await res.json();
      document.getElementById("trainResult").innerText = JSON.stringify(data, null, 2);
    } catch (err) {
      console.error("Error training model:", err);
      document.getElementById("trainResult").innerText = "Training failed.";
    } finally {
      hideProgress();
      refreshAllDropdowns();
      refreshStats();
    }
  });
}


  document.getElementById("evaluationDatasetSelector").addEventListener("change", async (e) => {
    const dataset = e.target.value;
    const targetColumnSelector = document.getElementById("evaluationTargetColumnSelector");

    // Clear previous options
    targetColumnSelector.innerHTML = '<option value="">Select target column</option>';

    if (!dataset) {
        return; // Exit if no dataset is selected
    }

    try {
        // Fetch the column names from your backend API
        const res = await fetch(`/get-columns/${dataset}`);
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const columns = await res.json();

        // Populate the dropdown with the received columns
        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            targetColumnSelector.appendChild(option);
        });
    } catch (error) {
        console.error("Failed to fetch columns:", error);
        alert("An error occurred while fetching columns for the selected dataset.");
    }
});
  
  

  const evaluateButton = document.getElementById("evaluateButton");
  if (evaluateButton) {
    evaluateButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();

      const modelSelector = document.getElementById("modelSelector");
      const model = modelSelector.value;
      const evaluationDataset = document.getElementById("evaluationDatasetSelector").value;

      // ✅ only grab targetColumn if the selector exists
      const targetSelector = document.getElementById("evaluationTargetColumnSelector");
      const targetColumn = targetSelector ? targetSelector.value : null;

      if (!model || !evaluationDataset || !targetColumn) {
        alert("Please select a model, evaluation dataset, and target column first");
        hideProgress();
        return;
      }

      const metricsSelect = document.querySelector('select[name="evaluation_metrics"]');
      const selectedMetrics = Array.from(metricsSelect.selectedOptions).map(option => option.value);

      const formData = {
        dataset: evaluationDataset,
        target: targetColumn,  // ✅ send target column
        metrics: selectedMetrics,
        visualization: document.querySelector('select[name="visualization"]').value
      };

      try {
        console.log("📤 Sending evaluation request:", formData);
        let res = await fetch(`/evaluate/${model}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData)
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        let data = await res.json();

        document.getElementById("evaluateResult").innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        console.error("Evaluation error:", err);
        alert("An error occurred during evaluation. Check console for details.");
      } finally {
        hideProgress();
      }
    });
  }


  // async function updateEvaluationTargetColumnOptions() {
  //   const datasetSelector = document.getElementById("evaluationDatasetSelector");
  //   const dataset = datasetSelector.value;
  //   const targetColumnSelector = document.getElementById("evaluationTargetColumnSelector");
  
  //   // reset dropdown
  //   targetColumnSelector.innerHTML = '<option value="">Select target column</option>';
  
  //   if (!dataset) return;
  
  //   try {
  //     let res = await fetch(`/dataset/columns/${dataset}`);
  //     if (!res.ok) throw new Error(`Failed to fetch columns for ${dataset}`);
  
  //     let columns = await res.json();
  
  //     columns.forEach(col => {
  //       let option = document.createElement("option");
  //       option.value = col;
  //       option.textContent = col;
  //       targetColumnSelector.appendChild(option);
  //     });
  //   } catch (err) {
  //     console.error("Error fetching dataset columns:", err);
  //     alert("Could not load target columns. Please try again.");
  //   }
  // }
  



  const predictButton = document.getElementById("predictButton");
  if (predictButton) {
    predictButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();
      
      const modelSelector = document.getElementById("predictModelSelector");
      const model = modelSelector.value;
      const inputFormat = document.getElementById("inputFormatSelector").value;
      
      if (!model) {
        alert("Please select a model first");
        hideProgress();
        return;
      }
      
      let inputData;
      
      if (inputFormat === 'json') {
        const inputText = document.querySelector('textarea[name="input_data"]').value;
        try {
          inputData = JSON.parse(inputText);
        } catch (error) {
          alert("Invalid JSON format in input data");
          hideProgress();
          return;
        }
      } else {
        inputData = {};
        const inputs = document.querySelectorAll('#featureInputs input');
        inputs.forEach(input => {
          inputData[input.name] = input.value;
        });
      }
      
      let res = await fetch(`/predict/${model}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({data: inputData})
      });
      let data = await res.json();
      document.getElementById("predictResult").innerText = JSON.stringify(data, null, 2);
      hideProgress();
      refreshStats();
    });
  }


document.addEventListener('DOMContentLoaded', function() {
  const downloadButton = document.getElementById('downloadButton');
  const modelSelector = document.getElementById('downloadModelSelector');
  const downloadResult = document.getElementById('downloadResult');

  /**
   * Fetches the list of available models from the backend API and
   * populates the model selector dropdown.
   */
  async function populateModelSelector() {
      try {
          // Fetch the list of model IDs from the backend's /api/models endpoint
          const response = await fetch('/models');
          if (!response.ok) {
              throw new Error('Failed to fetch models list.');
          }
          const models = await response.json();

          // Clear any existing options
          modelSelector.innerHTML = '<option value="">Select a model</option>';
          
          // Populate the dropdown with the fetched models
          if (models.length > 0) {
              models.forEach(modelId => {
                  const option = document.createElement('option');
                  option.value = modelId;
                  option.textContent = modelId;
                  modelSelector.appendChild(option);
              });
          } else {
              // If no models are found, show a message
              downloadResult.innerHTML = '<p class="has-text-warning">No models available for download.</p>';
          }

      } catch (error) {
          // Handle errors during the fetch request
          console.error('Error fetching models:', error);
          downloadResult.innerHTML = `<p class="has-text-danger">Failed to load models: ${error.message}</p>`;
      }
  }

  // Call the function to populate the selector when the page loads
  populateModelSelector();

  // Event listener for the download button
  downloadButton.addEventListener('click', function() {
      const selectedModelId = modelSelector.value;

      if (selectedModelId) {
          // Display a message while the download is being prepared
          downloadResult.innerHTML = '<p class="has-text-info">Preparing download...</p>';

          // Create a temporary anchor element to trigger the download
          const downloadUrl = `/api/download-model/${selectedModelId}`;
          const link = document.createElement('a');
          link.href = downloadUrl;
          link.download = `${selectedModelId}.pkl`; // Set the suggested filename
          
          // Append the link to the body, click it, and remove it
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          
          // Show a success message
          downloadResult.innerHTML = `<p class="has-text-success">Download started for **${selectedModelId}.pkl**.</p>`;

      } else {
          // Warn the user if no model is selected
          downloadResult.innerHTML = '<p class="has-text-danger">Please select a model to download.</p>';
      }
  });


// Run EDA
async function runEDA() {
  showProgress();
  const datasetSelector = document.getElementById("edaDatasetSelector");
  const dataset = datasetSelector.value;
  const analysisType = document.getElementById("analysisType").value;
  
  if (!dataset) {
    alert("Please select a dataset first");
    hideProgress();
    return;
  }
  
  let res = await fetch(`/eda/${analysisType}/${dataset}`, { method: "GET" });
  let data = await res.json();
  
  // Display EDA results in table format
  displayEDAResults(data, analysisType);
  hideProgress();
}

// Function to display EDA results
function displayEDAResults(data, analysisType) {
    const edaResult = document.getElementById('edaResult');
    
    // Clear any previous results
    edaResult.innerHTML = '';
    
    if (data.error) {
      edaResult.innerHTML = `<div class="notification is-danger">${data.error}</div>`;
      return;
    }
    
    let html = '';
    
    if (analysisType === 'summary') {
      html = `
        <h3 class="section-title">Dataset Overview</h3>
        <div class="columns">
          <div class="column">
            <p><strong>Shape:</strong> ${data.shape[0]} rows × ${data.shape[1]} columns</p>
          </div>
        </div>
  
        <h3 class="section-title">Structural Information (df.info())</h3>
        <pre class="info-pre">${data.info}</pre>
        
        <h3 class="section-title">Columns</h3>
        <div class="tags">
      `;
      
      // Add column tags
      data.columns.forEach(column => {
        html += `<span class="tag is-primary is-light">${column}</span>`;
      });
      
      html += `</div>
        
        <h3 class="section-title">Data Preview (First 5 Rows)</h3>
        <div class="eda-table-container">
          <table class="eda-table">
            <thead>
              <tr>
      `;
      
      // Add table headers
      data.columns.forEach(column => {
        html += `<th>${column}</th>`;
      });
      
      html += `</tr>
            </thead>
            <tbody>`;
      
      // Add table rows
      data.head.forEach(row => {
        html += `<tr>`;
        data.columns.forEach(column => {
          const cellValue = row[column] !== null ? row[column] : '—';
          html += `<td>${cellValue}</td>`;
        });
        html += `</tr>`;
      });
      
      html += `</tbody>
          </table>
        </div>
        
        <h3 class="section-title">Statistical Summary</h3>
        <div class="eda-table-container">
          <table class="stats-table">
            <thead>
              <tr>
                <th>Statistic</th>`;
      
      // Add statistic headers
      data.columns.forEach(column => {
        if (data.describe[column]) {
          html += `<th>${column}</th>`;
        }
      });
      
      html += `</tr>
            </thead>
            <tbody>`;
      
      // Add statistic rows
      const stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
      stats.forEach(stat => {
        html += `<tr><td>${stat}</td>`;
        data.columns.forEach(column => {
          if (data.describe[column] && data.describe[column][stat] !== undefined) {
            const value = data.describe[column][stat];
            html += `<td>${typeof value === 'number' ? 
                    value.toFixed(4) : value}</td>`;
          } else {
            html += `<td>-</td>`;
          }
        });
        html += `</tr>`;
      });
      
      html += `</tbody>
          </table>
        </div>`;
    } else if (analysisType === 'correlation') {
      html = `
        <h3 class="section-title">Correlation Matrix</h3>
        <div class="eda-table-container">
          <table class="stats-table">
            <thead>
              <tr>
                <th>Variable</th>
      `;
      
      // Add headers
      data.columns.forEach(column => {
        html += `<th>${column}</th>`;
      });
      
      html += `</tr>
            </thead>
            <tbody>`;
      
      // Add correlation rows
      data.columns.forEach((row, rowIndex) => {
        html += `<tr><td><strong>${row}</strong></td>`;
        data.columns.forEach((col, colIndex) => {
          const value = data.correlation[rowIndex][colIndex];
          html += `<td>${value.toFixed(4)}</td>`;
        });
        html += `</tr>`;
      });
      
      html += `</tbody>
          </table>
        </div>`;
    } else if (analysisType === 'distribution') {
      html = `
        <h3 class="section-title">Distribution Analysis</h3>
        <div class="eda-table-container">
          <table class="stats-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Skewness</th>
                <th>Kurtosis</th>
                <th>Normal Test (p-value)</th>
              </tr>
            </thead>
            <tbody>`;
      
      // Add distribution stats
      data.columns.forEach(column => {
        if (data.distribution[column]) {
          html += `
            <tr>
              <td><strong>${column}</strong></td>
              <td>${data.distribution[column].skewness.toFixed(4)}</td>
              <td>${data.distribution[column].kurtosis.toFixed(4)}</td>
              <td>${data.distribution[column].normality.toFixed(4)}</td>
            </tr>
          `;
        }
      });
      
      html += `</tbody>
          </table>
        </div>`;
    } else if (analysisType === 'missing') {
      html = `
        <h3 class="section-title">Missing Values Analysis</h3>
        <div class="eda-table-container">
          <table class="stats-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Missing Values</th>
                <th>Percentage</th>
              </tr>
            </thead>
            <tbody>`;
      
      // Add missing values stats
      data.columns.forEach(column => {
        if (data.missing[column]) {
          html += `
            <tr>
              <td><strong>${column}</strong></td>
              <td>${data.missing[column].count}</td>
              <td>${data.missing[column].percentage.toFixed(2)}%</td>
            </tr>
          `;
        }
      });
      
      html += `</tbody>
          </table>
        </div>`;
    } else if (analysisType === 'categorical') {
      html = `
        <h3 class="section-title">Categorical Variable Counts</h3>
        <div class="categorical-results">
      `;
      
      for (const column in data.categorical_counts) {
        const counts = data.categorical_counts[column];
        let innerHtml = '';
        for (const category in counts) {
          innerHtml += `<p><strong>${category}:</strong> ${counts[category]} occurrences</p>`;
        }
        
        html += `
          <div class="card my-4">
            <div class="card-header">
              <h4 class="card-header-title">${column}</h4>
            </div>
            <div class="card-content">
              ${innerHtml}
            </div>
          </div>
        `;
      }
      
      html += `</div>`;
    } else if (analysisType === 'outliers') {
      html = `
        <h3 class="section-title">Outlier Detection (IQR Method)</h3>
        <div class="eda-table-container">
          <table class="stats-table">
            <thead>
              <tr>
                <th>Column</th>
                <th>Outlier Count</th>
                <th>Outlier Percentage</th>
              </tr>
            </thead>
            <tbody>`;
            
      for (const column in data.outliers) {
        const outlier = data.outliers[column];
        html += `
          <tr>
            <td><strong>${column}</strong></td>
            <td>${outlier.count}</td>
            <td>${outlier.percentage.toFixed(2)}%</td>
          </tr>
        `;
      }
      
      html += `</tbody></table></div>`;
    } else if (analysisType === 'pairplot') {
      html = `
        <h3 class="section-title">Bivariate Plot (Pairplot)</h3>
        <p>A pairplot of the first few numerical features.</p>
        <div class="pairplot-container">
          <img src="${data.image}" alt="Pairplot" class="responsive-image">
        </div>
      `;
    }
    
    edaResult.innerHTML = html;
  }

// Refresh all dropdowns
async function refreshAllDropdowns() {
  await refreshDatasets();
  await refreshModels();
}

// Refresh datasets dropdown
async function refreshDatasets() {
  showProgress();
  let res = await fetch('/datasets', { method: "GET" });
  let data = await res.json();
  
  // Update all dataset selectors
  const selectors = [
    'edaDatasetSelector',
    'preprocessDatasetSelector',
    'featureDatasetSelector',
    'trainDatasetSelector',
    'evaluationDatasetSelector'
  ];
  
  selectors.forEach(selectorId => {
    let selector = document.getElementById(selectorId);
    if (selector) {
      selector.innerHTML = '<option value="">Select a dataset</option>';
      
      if (data.datasets) {
        data.datasets.forEach(dataset => {
          let option = document.createElement('option');
          option.value = dataset;
          option.textContent = dataset;
          selector.appendChild(option);
        });
      }
    }
  });
  
  hideProgress();
}

// Refresh models dropdown
async function refreshModels() {
  showProgress();
  let res = await fetch('/models', { method: "GET" });
  let data = await res.json();
  
  // Update all model selectors
  const selectors = [
    'modelSelector',
    'predictModelSelector',
    'deployModelSelector'
  ];
  
  selectors.forEach(selectorId => {
    let selector = document.getElementById(selectorId);
    if (selector) {
      selector.innerHTML = '<option value="">Select a model</option>';
      
      if (data.models) {
        data.models.forEach(model => {
          let option = document.createElement('option');
          option.value = model;
          option.textContent = model;
          selector.appendChild(option);
        });
      }
    }
  });
  
  hideProgress();
}

// Update target column options based on selected dataset
async function updateTargetColumnOptions() {
  const datasetSelector = document.getElementById("trainDatasetSelector");
  const dataset = datasetSelector.value;
  
  if (!dataset) {
    return;
  }
  
  showProgress();
  let res = await fetch(`/dataset/columns/${dataset}`, { method: "GET" });
  let data = await res.json();
  
  const targetColumnSelector = document.getElementById("targetColumnSelector");
  targetColumnSelector.innerHTML = '<option value="">Select target column</option>';
  
  if (data.columns) {
    data.columns.forEach(column => {
      let option = document.createElement('option');
      option.value = column;
      option.textContent = column;
      targetColumnSelector.appendChild(option);
    });
  }
  
  hideProgress();
}

// Toggle input format for prediction
function toggleInputFormat() {
  const inputFormat = document.getElementById("inputFormatSelector").value;
  const jsonInput = document.getElementById("jsonInput");
  const formInput = document.getElementById("formInput");
  
  if (inputFormat === 'json') {
    jsonInput.style.display = 'block';
    formInput.style.display = 'none';
  } else {
    jsonInput.style.display = 'none';
    formInput.style.display = 'block';
    
    // If form inputs haven't been generated yet, generate them
    if (document.getElementById("featureInputs").children.length === 1) {
      generateFeatureInputs();
    }
  }
}

// Generate feature inputs for prediction form
async function generateFeatureInputs() {
  const modelSelector = document.getElementById("predictModelSelector");
  const model = modelSelector.value;
  
  if (!model) {
    return;
  }
  
  showProgress();
  let res = await fetch(`/model/features/${model}`, { method: "GET" });
  let data = await res.json();
  
  const featureInputs = document.getElementById("featureInputs");
  featureInputs.innerHTML = '';
  
  if (data.features) {
    data.features.forEach(feature => {
      const inputGroup = document.createElement('div');
      inputGroup.className = 'field';
      inputGroup.innerHTML = `
        <label class="label">${feature}</label>
        <div class="control">
          <input class="input" type="text" name="${feature}" placeholder="Enter value for ${feature}">
        </div>
      `;
      featureInputs.appendChild(inputGroup);
    });
  }
  
  hideProgress();
}

// Update EDA dataset selector
function updateEDADataset() {
  const datasetSelector = document.getElementById("edaDatasetSelector");
  currentDataset = datasetSelector.value;
}

// Progress indicator functions
function showProgress() {
  document.getElementById('progressContainer').style.display = 'block';
}

function hideProgress() {
  document.getElementById('progressContainer').style.display = 'none';
}

// Refresh stats
async function refreshStats() {
  // In a real application, you would fetch these values from the server
  // For now, we'll use mock data
  document.getElementById('dataset-count').textContent = Math.floor(Math.random() * 10) + 1;
  document.getElementById('model-count').textContent = Math.floor(Math.random() * 5) + 1;
  document.getElementById('process-count').textContent = Math.floor(Math.random() * 8) + 1;
  document.getElementById('prediction-count').textContent = Math.floor(Math.random() * 20) + 1;
}
});

// Reset module functionality
function resetModule(moduleId) {
    const module = document.getElementById(moduleId);
    if (!module) {
        console.error(`Module with ID '${moduleId}' not found.`);
        return;
    }

    // Reset form inputs
    const form = module.querySelector('form');
    if (form) {
        form.reset();
    }

    // Clear and hide result containers
    const resultContainer = module.querySelector('.result-container');
    if (resultContainer) {
        resultContainer.innerHTML = '';
    }

    // Module-specific reset logic
    if (moduleId === 'upload') {
        const fileName = document.getElementById('fileName');
        if (fileName) {
            fileName.textContent = 'No file selected';
        }
    } else if (moduleId === 'eda') {
        const edaResult = document.getElementById('edaResult');
        if (edaResult) {
            edaResult.innerHTML = '';
        }
    } else if (moduleId === 'predict') {
        const featureInputs = document.getElementById('featureInputs');
        if (featureInputs) {
            featureInputs.innerHTML = '<p class="help">Select a model first to see feature inputs</p>';
        }
    }

    // Optional: reload the page to go back to the original state
    window.location.reload();
}

// Call refresh on page load
window.onload = function() {
  refreshAllDropdowns();
  refreshStats();
}
});