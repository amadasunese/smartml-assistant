import { showProgress, hideProgress, setJSON } from "./utils.js";
import { refreshAllDropdowns } from "./dropdowns.js";
import { refreshStats } from "./dashboard.js";

export function initPreprocess() {
  const btn = document.getElementById("preprocessButton");
  if (!btn) return;

  btn.addEventListener("click", async (e) => {
    e.preventDefault();
    showProgress();
    try {
      const dsSel = document.getElementById("preprocessDatasetSelector");
      const dataset = dsSel?.value;
      if (!dataset) {
        alert("Please select a dataset first");
        return;
      }

      const body = {
        drop_columns: Array.from(document.querySelectorAll('input[name="drop_columns"]:checked')).map(el => el.value),
        missing_strategy: document.querySelector('select[name="missing_strategy"]')?.value || "none",
        missing_indicator: document.querySelector('input[name="missing_indicator"]')?.checked || false,
        outlier_method: document.querySelector('select[name="outlier_method"]')?.value || "none",
        outlier_detection: document.querySelector('select[name="outlier_detection"]')?.value || "none",
        outlier_transform_method: document.querySelector('select[name="outlier_transform_method"]')?.value || "none",
        encoding_method: document.querySelector('select[name="encoding_method"]')?.value || "none",
        scaler: document.querySelector('select[name="scaler"]')?.value || "none",
        imbalance_method: document.querySelector('select[name="imbalance_method"]')?.value || "none",
        random_state: 42,
      };

      const res = await fetch(`/preprocess/${encodeURIComponent(dataset)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      setJSON(document.getElementById("preprocessResult"), data);
    } catch (err) {
      setJSON(document.getElementById("preprocessResult"), { error: "Preprocessing failed" });
    } finally {
      hideProgress();
      await refreshAllDropdowns();
      refreshStats();
    }
  });
}
