import { fetchDatasets, fetchModels, fetchColumns } from "./api.js";

export async function refreshDatasets() {
  const datasets = await fetchDatasets();
  const selectors = [
    "edaDatasetSelector",
    "preprocessDatasetSelector",
    "featureDatasetSelector",
    "trainDatasetSelector",
    "evaluationDatasetSelector",
  ];

  selectors.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = '<option value="">Select a dataset</option>';
    datasets.forEach(ds => {
      const opt = document.createElement("option");
      opt.value = ds;
      opt.textContent = ds;
      sel.appendChild(opt);
    });

    // Hook column loader for preprocessing
    if (id === "preprocessDatasetSelector") {
      sel.onchange = async (e) => {
        const dataset = e.target.value;
        const container = document.getElementById("dropColumnsContainer");
        if (!dataset) {
          container.innerHTML = "<p class='has-text-grey'>Select a dataset...</p>";
          return;
        }
        const cols = await fetchColumns(dataset);
        container.innerHTML = cols.map(c =>
          `<label class="checkbox mr-3">
             <input type="checkbox" name="drop_columns" value="${c}"> ${c}
           </label>`
        ).join("");
      };
    }
  });
}

export async function refreshModels() {
  const models = await fetchModels();
  const selectors = [
    "modelSelector", "predictModelSelector", 
    "deployModelSelector", "downloadModelSelector"
  ];

  selectors.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = '<option value="">Select a model</option>';
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      sel.appendChild(opt);
    });
  });
}

export async function refreshAllDropdowns() {
  await Promise.all([refreshDatasets(), refreshModels()]);
}
