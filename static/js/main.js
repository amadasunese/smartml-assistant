import { refreshAllDropdowns } from "./dropdowns.js";
import { refreshStats } from "./dashboard.js";
import { initPreprocess } from "./preprocess.js";
import { initUI } from "./ui.js"; 
import { initEDA } from "./eda.js"; 
// ... other initializers

document.addEventListener("DOMContentLoaded", async () => {
  initUI();
  initPreprocess();
  initEDA();
  // ... initFeatureEngineering(), initTraining(), etc.

  await refreshAllDropdowns();
  await refreshStats();
});




import { refreshAllDropdowns } from "./dropdowns.js";
import { refreshStats } from "./dashboard.js";
import { initPreprocess } from "./preprocess.js";
import { initFolders } from "./folders.js";
// import other modules as you build them

document.addEventListener("DOMContentLoaded", async () => {
  initPreprocess();
  initFolders();
  // initEDA(); initFeature(); initTraining(); initEvaluation(); initPrediction();

  await refreshAllDropdowns();
  await refreshStats();
});
