// Dataset fetch
export async function fetchDatasets() {
    const res = await fetch("/datasets");
    const raw = await res.json();
    return Array.isArray(raw) ? raw : raw.datasets || [];
  }
  
  // Model fetch
  export async function fetchModels() {
    const res = await fetch("/models");
    const raw = await res.json();
    let models = Array.isArray(raw) ? raw : raw.models || [];
    return models.map(m => {
      if (typeof m === "string") return { id: m, label: m };
      const id = m.id ?? m.model_id ?? m.name ?? "";
      const label = m.name ?? m.label ?? m.id ?? id ?? "model";
      return { id, label };
    });
  }
  
  // Columns fetch
  export async function fetchColumns(dataset) {
    const res = await fetch(`/get-columns/${encodeURIComponent(dataset)}`);
    if (!res.ok) throw new Error("Could not fetch columns");
    const raw = await res.json();
    return Array.isArray(raw) ? raw : raw.columns || [];
  }
  
  // Folder contents
  export async function fetchFolder(path) {
    const res = await fetch(`/list-folder/${path}`);
    if (!res.ok) return [];
    const raw = await res.json();
    return raw.files || [];
  }
  