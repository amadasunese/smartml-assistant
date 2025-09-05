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
});

// Upload dataset
document.getElementById("uploadForm").onsubmit = async (e) => {
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

// Function to display EDA results in table format
function displayEDAResults(data, analysisType) {
    const edaResult = document.getElementById('edaResult');

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
                html += `<td>${row[column]}</td>`;
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
                    html += `<td>${typeof data.describe[column][stat] === 'number' ?
                            data.describe[column][stat].toFixed(4) : data.describe[column][stat]}</td>`;
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
    }

    edaResult.innerHTML = html;
}

// Preprocess data
document.getElementById("preprocessButton").addEventListener("click", async (e) => {
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    });
    let data = await res.json();
    document.getElementById("preprocessResult").innerText = JSON.stringify(data, null, 2);
    hideProgress();
    refreshAllDropdowns();
    refreshStats();
});

// Feature Engineering
document.getElementById("featureEngineeringButton").addEventListener("click", async (e) => {
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    });
    let data = await res.json();
    document.getElementById("featureEngineeringResult").innerText = JSON.stringify(data, null, 2);
    hideProgress();
    refreshAllDropdowns();
    refreshStats();
});

// Train model
document.getElementById("trainButton").addEventListener("click", async (e) => {
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

    let res = await fetch(`/train/${dataset}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    });
    let data = await res.json();
    document.getElementById("trainResult").innerText = JSON.stringify(data, null, 2);
    hideProgress();
    refreshAllDropdowns();
    refreshStats();
});

// Evaluate model
document.getElementById("evaluateButton").addEventListener("click", async (e) => {
    e.preventDefault();
    showProgress();

    const modelSelector = document.getElementById("modelSelector");
    const model = modelSelector.value;
    const evaluationDataset = document.getElementById("evaluationDatasetSelector").value;

    if (!model || !evaluationDataset) {
        alert("Please select a model and evaluation dataset first");
        hideProgress();
        return;
    }

    // Get selected metrics (multiple select)
    const metricsSelect = document.querySelector('select[name="evaluation_metrics"]');
    const selectedMetrics = Array.from(metricsSelect.selectedOptions).map(option => option.value);

    const formData = {
        dataset: evaluationDataset,
        metrics: selectedMetrics,
        visualization: document.querySelector('select[name="visualization"]').value
    };

    let res = await fetch(`/evaluate/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    });
    let data = await res.json();
    document.getElementById("evaluateResult").innerText = JSON.stringify(data, null, 2);
    hideProgress();
});

// Predict
document.getElementById("predictButton").addEventListener("click", async (e) => {
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
        // Form input format
        inputData = {};
        const inputs = document.querySelectorAll('#featureInputs input');
        inputs.forEach(input => {
            inputData[input.name] = input.value;
        });
    }

    let res = await fetch(`/predict/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: inputData })
    });
    let data = await res.json();
    document.getElementById("predictResult").innerText = JSON.stringify(data, null, 2);
    hideProgress();
    refreshStats();
});

// Deploy model
document.getElementById("deployButton").addEventListener("click", async (e) => {
    e.preventDefault();
    showProgress();

    const modelSelector = document.getElementById("deployModelSelector");
    const model = modelSelector.value;

    if (!model) {
        alert("Please select a model first");
        hideProgress();
        return;
    }

    const formData = {
        platform: document.querySelector('select[name="deployment_platform"]').value,
        authentication: document.querySelector('select[name="api_authentication"]').value,
        endpoint_name: document.querySelector('input[name="endpoint_name"]').value
    };

    let res = await fetch(`/deploy/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    });
    let data = await res.json();
    document.getElementById("deployResult").innerText = JSON.stringify(data, null, 2);
    hideProgress();
});

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

// Call refresh on page load
window.onload = function() {
    refreshAllDropdowns();
    refreshStats();
};