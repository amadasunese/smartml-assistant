import { fetchFolder } from "./api.js";

export function initFolders() {
  const datasetLink = document.querySelector('a[href="#uploaded_data"]');
  const modelLink = document.querySelector('a[href="#saved_models"]');

  if (datasetLink) datasetLink.addEventListener("click", () => openFolder("uploaded_data", "dataset-count", "datasetFiles"));
  if (modelLink) modelLink.addEventListener("click", () => openFolder("saved_models", "model-count", "modelFiles"));
}

async function openFolder(path, badgeId, containerId) {
  const files = await fetchFolder(path);
  document.getElementById(badgeId).innerText = files.length;

  const container = document.getElementById(containerId);
  container.innerHTML = `
    <div>
      <button class="button is-small is-link" onclick="document.getElementById('${path}Upload').click()">Upload File</button>
      <input type="file" id="${path}Upload" style="display:none;" onchange="uploadFile('${path}', this.files[0])" />
    </div>
    <ul>
      ${files.map(f => `<li><a href="/download/${path}/${f}">${f}</a></li>`).join("")}
    </ul>
  `;
}
