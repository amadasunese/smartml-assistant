// Show/Hide loading spinner
export function showProgress() {
    const el = document.getElementById("progressContainer");
    if (el) el.style.display = "block";
  }
  
  export function hideProgress() {
    const el = document.getElementById("progressContainer");
    if (el) el.style.display = "none";
  }
  
  // Pretty-print JSON
  export function setJSON(el, data) {
    if (!el) return;
    try {
      el.innerText = JSON.stringify(data, null, 2);
    } catch {
      el.innerText = String(data);
    }
  }
  