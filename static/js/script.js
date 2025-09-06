/*************************************************
 * SmartML Assistant - Frontend Script (Clean)
 * ------------------------------------------------
 * - Single DOMContentLoaded initializer
 * - All helpers are hoisted (function declarations)
 * - Robust fetch + DOM guards
 * - Unified endpoints & dropdown refreshers
 **************************************************/

// =============================
// Global State
// =============================
let currentDataset = null;
let currentModel = null;

// =============================
// Progress Indicator
// =============================
function showProgress() {
  const el = document.getElementById("progressContainer");
  if (el) el.style.display = "block";
}

function hideProgress() {
  const el = document.getElementById("progressContainer");
  if (el) el.style.display = "none";
}

// =============================
// Utilities: JSON pretty-print
// =============================
function setJSON(el, data) {
  if (!el) return;
  try {
    el.innerText = JSON.stringify(data, null, 2);
  } catch {
    el.innerText = String(data);
  }
}


// =============================
// Dropdown Refreshers
// =============================
async function refreshAllDropdowns() {
  await Promise.all([refreshDatasets(), refreshModels()]);

  // If preprocess dataset is already selected, load its columns
  const dsSel = document.getElementById("preprocessDatasetSelector");
  if (dsSel && dsSel.value) {
    await loadDatasetColumns(dsSel.value);
  }
}

/**
 * Expected server responses:
 * /datasets -> either { datasets: ["ds1.csv", ...] } OR ["ds1.csv", ...]
 */
async function refreshDatasets() {
  try {
    const res = await fetch("/datasets", { method: "GET" });
    const raw = await res.json();
    const datasets = Array.isArray(raw) ? raw : raw?.datasets || [];

    const datasetSelectors = [
      "edaDatasetSelector",
      "preprocessDatasetSelector",
      "featureDatasetSelector",
      "trainDatasetSelector",
      "evaluationDatasetSelector",
    ];

    datasetSelectors.forEach((id) => {
      const sel = document.getElementById(id);
      if (!sel) return;
      sel.innerHTML = '<option value="">Select a dataset</option>';
      datasets.forEach((ds) => {
        const opt = document.createElement("option");
        opt.value = ds;
        opt.textContent = ds;
        sel.appendChild(opt);
      });

      // Attach onchange only for preprocess selector to load columns
      if (id === "preprocessDatasetSelector") {
        sel.addEventListener("change", async (e) => {
          const dataset = e.target.value;
          if (dataset) {
            await loadDatasetColumns(dataset);
          } else {
            const container = document.getElementById("dropColumnsContainer");
            if (container) container.innerHTML = "<p class='has-text-grey'>Select a dataset to load columns...</p>";
          }

        });
      }
    });

    if (datasets.length && !currentDataset) currentDataset = datasets[0];
  } catch (err) {
    console.error("Error fetching datasets:", err);
  }
}

// Collect excluded columns into an array
// function getExcludedColumns() {
//   const tags = document.querySelectorAll("#dropColumnsContainer .tag.is-excluded");
//   return Array.from(tags).map((t) => t.dataset.column);
// }
function getExcludedColumns() {
  const checkboxes = document.querySelectorAll('#dropColumnsContainer input[name="drop_columns"]:checked');
  return Array.from(checkboxes).map(cb => cb.value);
}

/**
 * Expected server responses:
 * /models -> can be:
 *   - ["id1","id2"] OR
 *   - { models: ["id1","id2"] } OR
 *   - [{id:"id1", name:"Model A"}, ...] OR
 *   - { models: [{id:"id1", name:"Model A"}, ...] }
 */
async function refreshModels() {
  try {
    const res = await fetch("/models", { method: "GET" });
    const raw = await res.json();
    let models = Array.isArray(raw) ? raw : raw?.models || [];

    // Normalize to objects {id, label}
    models = models.map((m) => {
      if (typeof m === "string") return { id: m, label: m };
      const id = m.id ?? m.model_id ?? m.name ?? "";
      const label = m.name ?? m.label ?? m.id ?? id ?? "model";
      return { id, label };
    });

    const modelSelectors = [
      "modelSelector",          // evaluation model selector
      "predictModelSelector",   // prediction model selector
      "deployModelSelector",    // optional deploy selector
      "downloadModelSelector",  // download selector
    ];

    modelSelectors.forEach((id) => {
      const sel = document.getElementById(id);
      if (!sel) return;
      sel.innerHTML = '<option value="">Select a model</option>';
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label;
        sel.appendChild(opt);
      });
    });

    if (models.length && !currentModel) currentModel = models[0].id;
  } catch (err) {
    console.error("Error fetching models:", err);
  }
}

/**
 * Fetch dataset columns for drop-column selection
 */
async function loadDatasetColumns(dataset) {
  try {
    const res = await fetch(`/get-columns/${encodeURIComponent(dataset)}`);
    const columns = await res.json();

    const container = document.getElementById("dropColumnsContainer");
    if (!container) return;

    container.innerHTML = "";
    if (Array.isArray(columns)) {
      columns.forEach((col) => {
        const label = document.createElement("label");
        label.classList.add("checkbox", "mr-3");

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.name = "drop_columns";
        checkbox.value = col;

        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(" " + col));
        container.appendChild(label);
      });
    } else {
      container.innerHTML = `<p class="has-text-danger">Error loading columns</p>`;
    }
  } catch (err) {
    console.error("Error loading columns:", err);
  }
}


// =============================
// Columns Helper
// =============================
async function populateColumnOptions(dataset, selectEl) {
  if (!dataset || !selectEl) return;
  // Unify on /get-columns/:dataset (fallback to /dataset/columns/:dataset)
  try {
    let res = await fetch(`/get-columns/${encodeURIComponent(dataset)}`);
    if (!res.ok) {
      res = await fetch(`/dataset/columns/${encodeURIComponent(dataset)}`);
    }
    if (!res.ok) throw new Error(`Column fetch failed (${res.status})`);
    const data = await res.json();

    const cols = Array.isArray(data) ? data : data.columns || data || [];
    selectEl.innerHTML = '<option value="">Select target column</option>';
    cols.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      selectEl.appendChild(opt);
    });
  } catch (err) {
    console.error("populateColumnOptions error:", err);
    alert("Could not load columns for the selected dataset.");
  }
}

async function updateTrainTargetColumnOptions() {
  const dsSel = document.getElementById("trainDatasetSelector");
  const targetSel = document.getElementById("targetColumnSelector");
  if (!dsSel || !targetSel) return;
  const ds = dsSel.value;
  if (!ds) {
    targetSel.innerHTML = '<option value="">Select target column</option>';
    return;
  }
  await populateColumnOptions(ds, targetSel);
}

async function updateEvaluationTargetColumnOptions() {
  const dsSel = document.getElementById("evaluationDatasetSelector");
  const targetSel = document.getElementById("evaluationTargetColumnSelector");
  if (!dsSel || !targetSel) return;
  const ds = dsSel.value;
  if (!ds) {
    targetSel.innerHTML = '<option value="">Select target column</option>';
    return;
  }
  await populateColumnOptions(ds, targetSel);
}

// =============================
// Prediction Helpers
// =============================
async function generateFeatureInputs() {
  const modelSel = document.getElementById("predictModelSelector");
  const container = document.getElementById("featureInputs");
  if (!modelSel || !container) return;

  const model = modelSel.value;
  if (!model) {
    container.innerHTML =
      '<p class="help">Select a model first to see feature inputs</p>';
    return;
  }

  try {
    showProgress();
    const res = await fetch(`/model/features/${encodeURIComponent(model)}`);
    const data = await res.json();

    container.innerHTML = "";
    const features = data?.features || [];
    features.forEach((f) => {
      const group = document.createElement("div");
      group.className = "field";
      group.innerHTML = `
        <label class="label">${f}</label>
        <div class="control">
          <input class="input" type="text" name="${f}" placeholder="Enter value for ${f}">
        </div>`;
      container.appendChild(group);
    });

    if (!features.length) {
      container.innerHTML =
        '<p class="help">No feature metadata returned for this model.</p>';
    }
  } catch (err) {
    console.error("generateFeatureInputs error:", err);
    alert("Could not load model features.");
  } finally {
    hideProgress();
  }
}

function toggleInputFormat() {
  const formatSel = document.getElementById("inputFormatSelector");
  const jsonInput = document.getElementById("jsonInput");
  const formInput = document.getElementById("formInput");
  if (!formatSel || !jsonInput || !formInput) return;

  if (formatSel.value === "json") {
    jsonInput.style.display = "block";
    formInput.style.display = "none";
  } else {
    jsonInput.style.display = "none";
    formInput.style.display = "block";
    const featureInputs = document.getElementById("featureInputs");
    if (featureInputs && featureInputs.children.length === 0) {
      generateFeatureInputs();
    }
  }
}

// =============================
// EDA
// =============================
async function runEDA() {
  const dsSel = document.getElementById("edaDatasetSelector");
  const analysisSel = document.getElementById("analysisType");
  const edaOut = document.getElementById("edaResult");
  if (!dsSel || !analysisSel || !edaOut) return;

  const dataset = dsSel.value;
  const analysisType = analysisSel.value;
  if (!dataset) {
    alert("Please select a dataset first");
    return;
  }

  try {
    showProgress();
    const res = await fetch(
      `/eda/${encodeURIComponent(analysisType)}/${encodeURIComponent(dataset)}`,
      { method: "GET" }
    );
    const data = await res.json();
    displayEDAResults(data, analysisType);
  } catch (err) {
    console.error("runEDA error:", err);
    edaOut.innerHTML =
      '<div class="notification is-danger">Failed to run EDA.</div>';
  } finally {
    hideProgress();
  }
}

function displayEDAResults(data, analysisType) {
  const edaResult = document.getElementById("edaResult");
  if (!edaResult) return;
  edaResult.innerHTML = "";

  if (data?.error) {
    edaResult.innerHTML = `<div class="notification is-danger">${data.error}</div>`;
    return;
  }

  let html = "";

  if (analysisType === "summary") {
    html += `
      <h3 class="section-title">Dataset Overview</h3>
      <div class="columns">
        <div class="column">
          <p><strong>Shape:</strong> ${data.shape?.[0] ?? "-"} rows × ${
            data.shape?.[1] ?? "-"
          } columns</p>
        </div>
      </div>

      <h3 class="section-title">Structural Information (df.info())</h3>
      <pre class="info-pre">${data.info ?? ""}</pre>

      <h3 class="section-title">Columns</h3>
      <div class="tags">
        ${(data.columns || [])
          .map((c) => `<span class="tag is-primary is-light">${c}</span>`)
          .join("")}
      </div>

      <h3 class="section-title">Data Preview (First 5 Rows)</h3>
      <div class="eda-table-container">
        <table class="eda-table">
          <thead>
            <tr>${(data.columns || [])
              .map((c) => `<th>${c}</th>`)
              .join("")}</tr>
          </thead>
          <tbody>
            ${(data.head || [])
              .map(
                (row) =>
                  `<tr>${(data.columns || [])
                    .map((c) => `<td>${row[c] ?? "—"}</td>`)
                    .join("")}</tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>

      <h3 class="section-title">Statistical Summary</h3>
      <div class="eda-table-container">
        <table class="stats-table">
          <thead>
            <tr>
              <th>Statistic</th>
              ${(data.columns || [])
                .filter((c) => data.describe?.[c])
                .map((c) => `<th>${c}</th>`)
                .join("")}
            </tr>
          </thead>
          <tbody>
            ${["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
              .map((stat) => {
                const cols = (data.columns || []).filter(
                  (c) => data.describe?.[c]
                );
                return `<tr><td>${stat}</td>${cols
                  .map((c) => {
                    const v = data.describe[c]?.[stat];
                    return `<td>${
                      typeof v === "number" ? v.toFixed(4) : v ?? "-"
                    }</td>`;
                  })
                  .join("")}</tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`;
  } else if (analysisType === "correlation") {
    html += `
      <h3 class="section-title">Correlation Matrix</h3>
      <div class="eda-table-container">
        <table class="stats-table">
          <thead>
            <tr><th>Variable</th>${(data.columns || [])
              .map((c) => `<th>${c}</th>`)
              .join("")}</tr>
          </thead>
          <tbody>
            ${(data.columns || [])
              .map((row, ri) => {
                const cells = (data.columns || [])
                  .map((col, ci) => {
                    const val = data.correlation?.[ri]?.[ci];
                    return `<td>${
                      typeof val === "number" ? val.toFixed(4) : "-"
                    }</td>`;
                  })
                  .join("");
                return `<tr><td><strong>${row}</strong></td>${cells}</tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`;
  } else if (analysisType === "distribution") {
    html += `
      <h3 class="section-title">Distribution Analysis</h3>
      <div class="eda-table-container">
        <table class="stats-table">
          <thead>
            <tr>
              <th>Column</th><th>Skewness</th><th>Kurtosis</th><th>Normal Test (p-value)</th>
            </tr>
          </thead>
          <tbody>
            ${(data.columns || [])
              .map((c) => {
                const d = data.distribution?.[c];
                if (!d) return "";
                return `<tr>
                  <td><strong>${c}</strong></td>
                  <td>${Number(d.skewness).toFixed(4)}</td>
                  <td>${Number(d.kurtosis).toFixed(4)}</td>
                  <td>${Number(d.normality).toFixed(4)}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`;
  } else if (analysisType === "missing") {
    html += `
      <h3 class="section-title">Missing Values Analysis</h3>
      <div class="eda-table-container">
        <table class="stats-table">
          <thead>
            <tr><th>Column</th><th>Missing Values</th><th>Percentage</th></tr>
          </thead>
          <tbody>
            ${(data.columns || [])
              .map((c) => {
                const m = data.missing?.[c];
                if (!m) return "";
                return `<tr>
                  <td><strong>${c}</strong></td>
                  <td>${m.count}</td>
                  <td>${Number(m.percentage).toFixed(2)}%</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`;
  } else if (analysisType === "categorical") {
    html += `<h3 class="section-title">Categorical Variable Counts</h3><div class="categorical-results">`;
    const cc = data.categorical_counts || {};
    Object.keys(cc).forEach((col) => {
      const counts = cc[col] || {};
      html += `
        <div class="card my-4">
          <div class="card-header"><h4 class="card-header-title">${col}</h4></div>
          <div class="card-content">
            ${Object.keys(counts)
              .map(
                (k) => `<p><strong>${k}:</strong> ${counts[k]} occurrences</p>`
              )
              .join("")}
          </div>
        </div>`;
    });
    html += `</div>`;
  } else if (analysisType === "outliers") {
    html += `
      <h3 class="section-title">Outlier Detection (IQR Method)</h3>
      <div class="eda-table-container">
        <table class="stats-table">
          <thead>
            <tr><th>Column</th><th>Outlier Count</th><th>Outlier Percentage</th></tr>
          </thead>
          <tbody>
            ${Object.keys(data.outliers || {})
              .map((c) => {
                const o = data.outliers[c];
                return `<tr>
                  <td><strong>${c}</strong></td>
                  <td>${o.count}</td>
                  <td>${Number(o.percentage).toFixed(2)}%</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>`;
  } else if (analysisType === "pairplot") {
    html += `
      <h3 class="section-title">Bivariate Plot (Pairplot)</h3>
      <p>A pairplot of the first few numerical features.</p>
      <div class="pairplot-container">
        <img src="${data.image}" alt="Pairplot" class="responsive-image">
      </div>`;
  
  } else if (analysisType === "duplicates") {
    html += `
      <h3 class="section-title">Duplicate Rows Analysis</h3>
      <p><strong>Total Rows:</strong> ${data.total_rows}</p>
      <p><strong>Duplicate Rows:</strong> ${data.duplicates}</p>
      <p><strong>Percentage:</strong> ${Number(data.percentage).toFixed(2)}%</p>
    `;
  } else if (analysisType === "imbalance") {
    html += `
      <h3 class="section-title">Class Imbalance Check</h3>
      <p><strong>Target Column:</strong> ${data.target_column}</p>
      <div class="eda-table-container">
        <table class="stats-table">
          <thead>
            <tr><th>Class</th><th>Count</th></tr>
          </thead>
          <tbody>
            ${Object.keys(data.class_distribution || {})
              .map(
                (cls) => `<tr>
                  <td><strong>${cls}</strong></td>
                  <td>${data.class_distribution[cls]}</td>
                </tr>`
              )
              .join("")}
          </tbody>
        </table>
      </div>
    `;
  } else if (analysisType === "skewness") {
    html += `
      <h3 class="section-title">Skewness / Box-Cox Check</h3>
      <div class="eda-table-container">
        <table class="stats-table">
          <thead>
            <tr><th>Column</th><th>Skewness</th><th>Box-Cox Applicable?</th></tr>
          </thead>
          <tbody>
            ${(data.columns || [])
              .map((c) => {
                const s = data.skewness?.[c];
                if (!s) return "";
                return `<tr>
                  <td><strong>${c}</strong></td>
                  <td>${Number(s.skewness).toFixed(4)}</td>
                  <td>${s.boxcox_applicable ? "Yes" : "No"}</td>
                </tr>`;
              })
              .join("")}
          </tbody>
        </table>
      </div>
    `;
    }      

  edaResult.innerHTML = html;
}

// =============================
// Dashboard Stats (mock fallback)
// =============================
// async function refreshStats() {
//   try {
//     const res = await fetch("/stats", { method: "GET" });
//     if (res.ok) {
//       const s = await res.json();
//       if (s) {
//         const ids = [
//           ["dataset-count", s.datasets],
//           ["model-count", s.models],
//           ["process-count", s.processes],
//           ["prediction-count", s.predictions],
//         ];
//         ids.forEach(([id, val]) => {
//           const el = document.getElementById(id);
//           if (el && typeof val !== "undefined") {
//             el.textContent = val;
//           }
//         });
//         return;
//       }
//     }
//   } catch (err) {
//     console.error("Error fetching stats:", err);
//   }

fetch("/stats")
  .then(response => response.json())
  .then(data => {
    // Sidebar
    document.getElementById("sidebar-datasets-count").textContent = `(${data.datasets || 0})`;
    document.getElementById("sidebar-models-count").textContent = `(${data.models || 0})`;

    // Stat cards
    document.getElementById("card-datasets-count").textContent = data.datasets || 0;
    document.getElementById("card-models-count").textContent = data.models || 0;
    document.getElementById("card-process-count").textContent = data.processes || 0;
    document.getElementById("card-prediction-count").textContent = data.predictions || 0;
  });


  // fallback values
  // ["dataset-count", "model-count", "process-count", "prediction-count"].forEach(id => {
  //   const el = document.getElementById(id);
  //   if (el) el.textContent = "0";
  // });
// }

// async function refreshStats() {
//   try {
//     const res = await fetch("/stats", { method: "GET" });
//     if (res.ok) {
//       const s = await res.json();
//       document.getElementById("dataset-count")?.append?.();
//       if (s) {
//         const ids = [
//           ["dataset-count", s.datasets],
//           ["model-count", s.models],
//           ["process-count", s.processes],
//           ["prediction-count", s.predictions],
//         ];
//         ids.forEach(([id, val]) => {
//           const el = document.getElementById(id);
//           if (el && typeof val !== "undefined") el.textContent = val;
//         });
//         return;
//       }
//     }
//   } catch {
//     // fall through to mock
//   }
//   // Mock values
//   const setRand = (id, max) => {
//     const el = document.getElementById(id);
//     if (el) el.textContent = Math.floor(Math.random() * max) + 0;
//   };
//   setRand("dataset-count", 0);
//   setRand("model-count", 0);
//   setRand("process-count", 0);
//   setRand("prediction-count", 0);
// }

// =============================
// Module Reset
// =============================
function resetModule(moduleId) {
  const module = document.getElementById(moduleId);
  if (!module) {
    console.error(`Module with ID '${moduleId}' not found.`);
    return;
  }

  // Reset form inputs
  const form = module.querySelector("form");
  if (form) form.reset();

  // Clear & hide result containers
  const resultContainer = module.querySelector(".result-container");
  if (resultContainer) resultContainer.innerHTML = "";

  // Module-specific resets
  if (moduleId === "upload") {
    const fileName = document.getElementById("fileName");
    if (fileName) fileName.textContent = "No file selected";
  } else if (moduleId === "eda") {
    const edaResult = document.getElementById("edaResult");
    if (edaResult) edaResult.innerHTML = "";
  } else if (moduleId === "predict") {
    const featureInputs = document.getElementById("featureInputs");
    if (featureInputs) {
      featureInputs.innerHTML =
        '<p class="help">Select a model first to see feature inputs</p>';
    }
  }

  // Optional: reload page
  // window.location.reload();
}

// =============================
// DOMContentLoaded - Wire Up UI
// =============================
// // --- Tab Switching (Upload Module)
// document.querySelectorAll('#upload .tabs ul li').forEach(tab => {
//   tab.addEventListener('click', () => {
//     // remove is-active from all tabs
//     document.querySelectorAll('#upload .tabs ul li').forEach(t => t.classList.remove('is-active'));
//     // hide all tab-content
//     document.querySelectorAll('#upload .tab-content').forEach(c => c.classList.add('is-hidden'));

//     // activate clicked tab
//     tab.classList.add('is-active');
//     const target = tab.getAttribute('data-tab');
//     document.getElementById(`tab-${target}`).classList.remove('is-hidden');
//   });
// });


document.addEventListener("DOMContentLoaded", () => {
  // Tabs (data-tab -> content id)
  const tabs = document.querySelectorAll(".tabs li");
  const tabContents = document.querySelectorAll(".tab-content");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;

      tabs.forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");

      tabContents.forEach((content) => {
        content.classList.add("is-hidden");
        if (content.id === target) content.classList.remove("is-hidden");
      });
    });
  });
  

  // File input name display
  const fileInput = document.querySelector("#fileInput");
  if (fileInput) {
    fileInput.onchange = () => {
      if (fileInput.files.length > 0) {
        const fileName = document.querySelector("#fileName");
        if (fileName) fileName.textContent = fileInput.files[0].name;
      }
    };
  }

  // Hide progress via button
  const hideBtn = document.getElementById("hideProgress");
  if (hideBtn) {
    hideBtn.addEventListener("click", () => {
      const cont = document.getElementById("progressContainer");
      if (cont) cont.style.display = "none";
    });
  }

  // Mobile menu & sidebar
  const mobileMenuButton = document.getElementById("mobileMenuButton");
  const sidebar = document.getElementById("sidebar");
  if (mobileMenuButton && sidebar) {
    mobileMenuButton.addEventListener("click", () => {
      const visible =
        sidebar.style.display !== "none" && sidebar.style.display !== "";
      if (visible) {
        sidebar.style.display = "none";
        sidebar.style.width = "0";
      } else {
        sidebar.style.display = "block";
        sidebar.style.width = "250px";
      }
    });

    function checkMobileView() {
      if (window.innerWidth <= 768) {
        mobileMenuButton.style.display = "block";
        sidebar.style.display = "none";
        sidebar.style.width = "0";
      } else {
        mobileMenuButton.style.display = "none";
        sidebar.style.display = "block";
        sidebar.style.width = "250px";
      }
    }
    checkMobileView();
    window.addEventListener("resize", checkMobileView);
  }

  // Sidebar smooth-scroll
  const sidebarLinks = document.querySelectorAll(".sidebar-menu a");
  sidebarLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href")?.substring(1);
      sidebarLinks.forEach((l) => l.classList.remove("active"));
      link.classList.add("active");
      if (targetId) {
        const targetSection = document.getElementById(targetId);
        if (targetSection) targetSection.scrollIntoView({ behavior: "smooth" });
      }
    });
  });

//   // --- Upload Dataset

// --- Upload Handler
const uploadButton = document.getElementById("uploadButton");
if (uploadButton) {
  uploadButton.addEventListener("click", async () => {
    const activeTab = document.querySelector("#upload .tabs li.is-active").dataset.tab;
    const datasetName = document.getElementById("datasetName").value;
    let formData = new FormData();

    if (activeTab === "local") {
      const fileInput = document.getElementById("fileInput");
      if (!fileInput.files.length) {
        alert("Please select a file");
        return;
      }
      formData.append("file", fileInput.files[0]);
      if (datasetName) formData.append("name", datasetName);

      await sendUpload("/upload", formData);

    } else if (activeTab === "url") {
      const url = document.getElementById("datasetUrl").value;
      if (!url) {
        alert("Please enter a dataset URL");
        return;
      }
      await sendUpload("/upload/url", JSON.stringify({ url, name: datasetName }), "json");

    } else if (activeTab === "cloud") {
      // Example: Choose Google Drive for now
      const source = document.querySelector("#cloudSource").value; // dropdown select
      const url = document.querySelector("#cloudUrl").value;
      const datasetName = document.getElementById("datasetName").value;

      const res = await fetch("/upload/cloud", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, url, name: datasetName }),
      });
      data = await res.json();
    }
  });
}

// helper for uploads
async function sendUpload(endpoint, data, type = "form") {
  showProgress();
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      body: type === "form" ? data : data,
      headers: type === "json" ? { "Content-Type": "application/json" } : undefined
    });

    const result = await res.json();
    setJSON(document.getElementById("uploadResult"), result);
    await refreshAllDropdowns();
    // refreshStats();
  } catch (err) {
    console.error("Upload failed", err);
    setJSON(document.getElementById("uploadResult"), { error: "Upload failed" });
  } finally {
    hideProgress();
  }
}



//   const uploadForm = document.getElementById("uploadForm");
//   if (uploadForm) {
//     uploadForm.onsubmit = async (e) => {
//       e.preventDefault();
//       showProgress();
//       try {
//         const formData = new FormData(uploadForm);

//         // add custom dataset name if provided
//         const datasetNameEl = document.getElementById("datasetName");
//         if (datasetNameEl && datasetNameEl.value) {
//           formData.append("name", datasetNameEl.value);
//         }

//         const res = await fetch("/upload", { method: "POST", body: formData });
//         const data = await res.json();
//         setJSON(document.getElementById("uploadResult"), data);
//       } catch (err) {
//         console.error("Upload error:", err);
//         setJSON(document.getElementById("uploadResult"), {
//           error: "Upload failed.",
//         });
//       } finally {
//         hideProgress();
//         await refreshAllDropdowns();
//         refreshStats();
//       }
//     };
//   }

//   // --- Upload Dataset from URL
// const urlUploadForm = document.getElementById("urlUploadForm");
// if (urlUploadForm) {
//   urlUploadForm.onsubmit = async (e) => {
//     e.preventDefault();
//     showProgress();
//     try {
//       const url = document.getElementById("datasetUrl").value;
//       const res = await fetch("/upload/url", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ url }),
//       });

//       const data = await res.json();
//       setJSON(document.getElementById("uploadResult"), data);
//     } catch (err) {
//       console.error("URL upload error:", err);
//       setJSON(document.getElementById("uploadResult"), {
//         error: "URL upload failed.",
//       });
//     } finally {
//       hideProgress();
//       await refreshAllDropdowns();
//       refreshStats();
//     }
//   };
// }

// // --- Cloud Upload (Google Drive, Dropbox, S3)
// async function uploadFromCloud(source, url) {
//   showProgress();
//   try {
//     const res = await fetch("/upload/cloud", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ source, url }),
//     });

//     const data = await res.json();
//     setJSON(document.getElementById("uploadResult"), data);
//   } catch (err) {
//     console.error("Cloud upload error:", err);
//     setJSON(document.getElementById("uploadResult"), {
//       error: `Failed to import from ${source}.`,
//     });
//   } finally {
//     hideProgress();
//     await refreshAllDropdowns();
//     refreshStats();
//   }
// }

// // Example triggers (assuming you let users paste the file link)
// document.getElementById("btn-gdrive")?.addEventListener("click", () => {
//   const url = prompt("Paste Google Drive file link:");
//   if (url) uploadFromCloud("gdrive", url);
// });

// document.getElementById("btn-dropbox")?.addEventListener("click", () => {
//   const url = prompt("Paste Dropbox file link:");
//   if (url) uploadFromCloud("dropbox", url);
// });

// document.getElementById("btn-s3")?.addEventListener("click", () => {
//   const url = prompt("Paste S3 file link:");
//   if (url) uploadFromCloud("s3", url);
// });


// --- Preprocess
const preprocessButton = document.getElementById("preprocessButton");
if (preprocessButton) {
  preprocessButton.addEventListener("click", async (e) => {
    e.preventDefault();
    showProgress();
    try {
      const dsSel = document.getElementById("preprocessDatasetSelector");
      const dataset = dsSel ? dsSel.value : "";
      if (!dataset) {
        alert("Please select a dataset first");
        return;
      }

      // Collect preprocessing options
      const body = {
        // Duplicates
        duplicate_handling: document.querySelector('select[name="duplicate_handling"]')?.value || "none",

        drop_columns: getExcludedColumns(),

        // Missing values
        missing_strategy: document.querySelector('select[name="missing_strategy"]')?.value || "none",
        missing_columns: Array.from(document.querySelectorAll('input[name="missing_columns"]:checked')).map(el => el.value),
        missing_indicator: document.querySelector('input[name="missing_indicator"]')?.checked || false,

        // Outliers
        outlier_method: document.querySelector('select[name="outlier_method"]')?.value || "none",
        outlier_columns: Array.from(document.querySelectorAll('input[name="outlier_columns"]:checked')).map(el => el.value),
        transform_method: document.querySelector('select[name="transform_method"]')?.value || "none",

        // Skewness
        skewness_method: document.querySelector('select[name="skewness_method"]')?.value || "none",
        skewness_columns: Array.from(document.querySelectorAll('input[name="skewness_columns"]:checked')).map(el => el.value),

        // Encoding
        encoding_method: document.querySelector('select[name="encoding_method"]')?.value || "none",
        encoding_columns: Array.from(document.querySelectorAll('input[name="encoding_columns"]:checked')).map(el => el.value),

        // Scaling
        scaler: document.querySelector('select[name="scaler"]')?.value || "none",
        scaling_columns: Array.from(document.querySelectorAll('input[name="scaling_columns"]:checked')).map(el => el.value),

        // Target + imbalance
        target_column: document.querySelector('select[name="target_column"]')?.value || null,
        imbalance_method: document.querySelector('select[name="imbalance_method"]')?.value || "none",
        random_state: 42
      };


      // Send to backend
      const res = await fetch(`/preprocess/${encodeURIComponent(dataset)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      setJSON(document.getElementById("preprocessResult"), data);

    } catch (err) {
      console.error("Preprocess error:", err);
      setJSON(document.getElementById("preprocessResult"), {
        error: "Preprocessing failed.",
      });
    } finally {
      hideProgress();
      await refreshAllDropdowns();
      refreshStats();
    }
  });
}



  // --- Feature Engineering
  const featureEngineeringButton = document.getElementById(
    "featureEngineeringButton"
  );
  if (featureEngineeringButton) {
    featureEngineeringButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();
      try {
        const dsSel = document.getElementById("featureDatasetSelector");
        const dataset = dsSel ? dsSel.value : "";
        if (!dataset) {
          alert("Please select a dataset first");
          return;
        }
        const body = {
          feature_generation:
            document.querySelector('select[name="feature_generation"]')
              ?.value || "",
          feature_selection:
            document.querySelector('select[name="feature_selection"]')
              ?.value || "",
          dimensionality_reduction:
            document.querySelector('select[name="dimensionality_reduction"]')
              ?.value || "",
        };
        const res = await fetch(
          `/feature-engineering/${encodeURIComponent(dataset)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }
        );
        const data = await res.json();
        setJSON(document.getElementById("featureEngineeringResult"), data);
      } catch (err) {
        console.error("Feature engineering error:", err);
        setJSON(document.getElementById("featureEngineeringResult"), {
          error: "Feature engineering failed.",
        });
      } finally {
        hideProgress();
        await refreshAllDropdowns();
        refreshStats();
      }
    });
  }

  // --- Training
  const trainButton = document.getElementById("trainButton");
  if (trainButton) {
    trainButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();
      const dsSel = document.getElementById("trainDatasetSelector");
      const dataset = dsSel ? dsSel.value : "";
      const target = document.getElementById("targetColumnSelector")?.value;
      if (!dataset || !target) {
        alert("Please select a dataset and target column first");
        hideProgress();
        return;
      }
      const body = {
        target,
        algorithm: document.querySelector('select[name="algorithm"]')?.value,
        test_size: document.querySelector('select[name="test_size"]')?.value,
        cross_validation:
          document.querySelector('select[name="cross_validation"]')?.value,
        hyperparameter_tuning:
          document.querySelector('select[name="hyperparameter_tuning"]')?.value,
      };
      try {
        const res = await fetch(`/train/${encodeURIComponent(dataset)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        setJSON(document.getElementById("trainResult"), data);
      } catch (err) {
        console.error("Training error:", err);
        setJSON(document.getElementById("trainResult"), {
          error: "Training failed.",
        });
      } finally {
        hideProgress();
        await refreshAllDropdowns();
        refreshStats();
      }
    });
  }

  // Update train target columns when dataset changes
  document
    .getElementById("trainDatasetSelector")
    ?.addEventListener("change", updateTrainTargetColumnOptions);

  // --- Evaluation
  document
    .getElementById("evaluationDatasetSelector")
    ?.addEventListener("change", updateEvaluationTargetColumnOptions);

  const evaluateButton = document.getElementById("evaluateButton");
  if (evaluateButton) {
    evaluateButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();
      try {
        const model = document.getElementById("modelSelector")?.value;
        const dataset =
          document.getElementById("evaluationDatasetSelector")?.value;
        const target =
          document.getElementById("evaluationTargetColumnSelector")?.value;

        if (!model || !dataset || !target) {
          alert(
            "Please select a model, evaluation dataset, and target column first"
          );
          return;
        }

        const metricsSel = document.querySelector(
          'select[name="evaluation_metrics"]'
        );
        const selectedMetrics = metricsSel
          ? Array.from(metricsSel.selectedOptions).map((o) => o.value)
          : [];

        const body = {
          dataset,
          target,
          metrics: selectedMetrics,
          visualization:
            document.querySelector('select[name="visualization"]')?.value || "",
        };

        const res = await fetch(`/evaluate/${encodeURIComponent(model)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        setJSON(document.getElementById("evaluateResult"), data);
      } catch (err) {
        console.error("Evaluation error:", err);
        alert("An error occurred during evaluation.");
      } finally {
        hideProgress();
      }
    });
  }

  // --- Prediction
  const inputFormatSelector = document.getElementById("inputFormatSelector");
  inputFormatSelector?.addEventListener("change", toggleInputFormat);

  const predictModelSelector = document.getElementById("predictModelSelector");
  predictModelSelector?.addEventListener("change", () => {
    // if form mode, regenerate inputs
    const fmt = document.getElementById("inputFormatSelector")?.value;
    if (fmt === "form") generateFeatureInputs();
  });

  const predictButton = document.getElementById("predictButton");
  if (predictButton) {
    predictButton.addEventListener("click", async (e) => {
      e.preventDefault();
      showProgress();
      try {
        const model = document.getElementById("predictModelSelector")?.value;
        const fmt = document.getElementById("inputFormatSelector")?.value;
        if (!model) {
          alert("Please select a model first");
          return;
        }
        let inputData;
        if (fmt === "json") {
          const raw = document.querySelector('textarea[name="input_data"]')
            ?.value;
          try {
            inputData = JSON.parse(raw || "{}");
          } catch {
            alert("Invalid JSON format in input data");
            return;
          }
        } else {
          inputData = {};
          const inputs = document.querySelectorAll("#featureInputs input");
          inputs.forEach((inp) => (inputData[inp.name] = inp.value));
        }

        const res = await fetch(`/predict/${encodeURIComponent(model)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data: inputData }),
        });
        const data = await res.json();
        setJSON(document.getElementById("predictResult"), data);
      } catch (err) {
        console.error("Prediction error:", err);
        setJSON(document.getElementById("predictResult"), {
          error: "Prediction failed.",
        });
      } finally {
        hideProgress();
        refreshStats();
      }
    });
  }

  // --- Download Model
  const downloadButton = document.getElementById("downloadButton");
  const downloadModelSelector = document.getElementById("downloadModelSelector");
  const downloadResult = document.getElementById("downloadResult");

  downloadButton?.addEventListener("click", () => {
    const selectedModelId = downloadModelSelector?.value;
    if (selectedModelId) {
      if (downloadResult)
        downloadResult.innerHTML =
          '<p class="has-text-info">Preparing download...</p>';
      const link = document.createElement("a");
      link.href = `/api/download-model/${encodeURIComponent(selectedModelId)}`;
      link.download = `${selectedModelId}.pkl`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      if (downloadResult)
        downloadResult.innerHTML = `<p class="has-text-success">Download started for <strong>${selectedModelId}.pkl</strong>.</p>`;
    } else {
      if (downloadResult)
        downloadResult.innerHTML =
          '<p class="has-text-danger">Please select a model to download.</p>';
    }
  });

  // --- EDA Button (if present)
  document.getElementById("runEDAButton")?.addEventListener("click", runEDA);

  // --- EDA dataset change sets currentDataset
  document.getElementById("edaDatasetSelector")?.addEventListener("change", () => {
    const ds = document.getElementById("edaDatasetSelector")?.value;
    currentDataset = ds || currentDataset;
  });

  // Initial load
  (async () => {
    await refreshAllDropdowns();
    // refreshStats();
    toggleInputFormat(); // set initial visibility if present
  })();
});

// =============================
// Expose selective helpers (if HTML uses inline handlers)
// =============================
window.refreshAllDropdowns = refreshAllDropdowns;
window.updateTargetColumnOptions = updateTrainTargetColumnOptions; // for inline onchange on train dataset
window.updateEvaluationTargetColumnOptions = updateEvaluationTargetColumnOptions;
window.toggleInputFormat = toggleInputFormat;
window.generateFeatureInputs = generateFeatureInputs;
window.runEDA = runEDA;
window.resetModule = resetModule;
