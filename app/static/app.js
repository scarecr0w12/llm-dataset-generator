const state = {
  datasets: [],
  selectedDataset: null,
  examples: [],
  selectedExample: null,
  activeWorkspaceTab: "review",
  exampleFilters: {
    query: "",
    status: "all",
  },
};

const elements = {
  datasetList: document.querySelector("#datasetList"),
  datasetForm: document.querySelector("#datasetForm"),
  refreshDatasets: document.querySelector("#refreshDatasets"),
  datasetUpdateButton: document.querySelector("#datasetUpdateButton"),
  datasetDeleteButton: document.querySelector("#datasetDeleteButton"),
  datasetTitle: document.querySelector("#datasetTitle"),
  datasetDescription: document.querySelector("#datasetDescription"),
  activeDatasetMeta: document.querySelector("#activeDatasetMeta"),
  datasetSchemaBadge: document.querySelector("#datasetSchemaBadge"),
  statExamples: document.querySelector("#statExamples"),
  statTokens: document.querySelector("#statTokens"),
  statSchema: document.querySelector("#statSchema"),
  manualExampleForm: document.querySelector("#manualExampleForm"),
  uploadForm: document.querySelector("#uploadForm"),
  syntheticForm: document.querySelector("#syntheticForm"),
  searxngForm: document.querySelector("#searxngForm"),
  webImportForm: document.querySelector("#webImportForm"),
  githubForm: document.querySelector("#githubForm"),
  integrationResult: document.querySelector("#integrationResult"),
  curationForm: document.querySelector("#curationForm"),
  curationResult: document.querySelector("#curationResult"),
  exampleTable: document.querySelector("#exampleTable"),
  exampleSearch: document.querySelector("#exampleSearch"),
  exampleStatusFilter: document.querySelector("#exampleStatusFilter"),
  workspaceTabs: Array.from(document.querySelectorAll("[data-tab-target]")),
  workspacePanels: Array.from(document.querySelectorAll("[data-tab-panel]")),
  annotationForm: document.querySelector("#annotationForm"),
  deleteExampleButton: document.querySelector("#deleteExampleButton"),
  exportForm: document.querySelector("#exportForm"),
};

function field(form, name) {
  return form.elements.namedItem(name);
}

function ensureDataset() {
  if (!state.selectedDataset) {
    alert("Select a dataset first.");
    return false;
  }
  return true;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail || "Request failed");
  }
  const disposition = response.headers.get("content-disposition") || "";
  if (disposition.includes("attachment")) {
    return { response, blob: await response.blob() };
  }
  return response.json();
}

function formToJson(form) {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function parseLabels(value) {
  return (value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function setActiveWorkspaceTab(tabName) {
  state.activeWorkspaceTab = tabName;
  elements.workspaceTabs.forEach((button) => {
    const isActive = button.dataset.tabTarget === tabName;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
  elements.workspacePanels.forEach((panel) => {
    panel.hidden = panel.dataset.tabPanel !== tabName;
  });
}

function filterExamples() {
  const query = state.exampleFilters.query.trim().toLowerCase();
  const status = state.exampleFilters.status;

  return state.examples.filter((example) => {
    const matchesStatus = status === "all" || example.status === status;
    if (!matchesStatus) {
      return false;
    }
    if (!query) {
      return true;
    }
    const haystack = [
      example.instruction,
      example.input_text,
      example.output_text,
      example.system_prompt,
      ...(example.labels || []),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
}

function updateActionState() {
  const hasDataset = Boolean(state.selectedDataset);
  const hasExample = Boolean(state.selectedExample);

  elements.datasetUpdateButton.disabled = !hasDataset;
  elements.datasetDeleteButton.disabled = !hasDataset;
  elements.deleteExampleButton.disabled = !hasExample;
  Array.from(elements.manualExampleForm.elements).forEach((control) => {
    if (control.type !== "submit") {
      control.disabled = !hasDataset;
    }
  });
  Array.from(elements.uploadForm.elements).forEach((control) => {
    control.disabled = !hasDataset;
  });
  Array.from(elements.syntheticForm.elements).forEach((control) => {
    control.disabled = !hasDataset;
  });
  Array.from(elements.searxngForm.elements).forEach((control) => {
    control.disabled = !hasDataset;
  });
  Array.from(elements.webImportForm.elements).forEach((control) => {
    control.disabled = !hasDataset;
  });
  Array.from(elements.githubForm.elements).forEach((control) => {
    control.disabled = !hasDataset;
  });
  Array.from(elements.curationForm.elements).forEach((control) => {
    control.disabled = !hasDataset;
  });
  Array.from(elements.exportForm.elements).forEach((control) => {
    control.disabled = !hasDataset;
  });
  Array.from(elements.annotationForm.elements).forEach((control) => {
    if (control.name !== "id") {
      control.disabled = !hasExample;
    }
  });
}

function setDatasetForm(dataset) {
  field(elements.datasetForm, "name").value = dataset?.name || "";
  field(elements.datasetForm, "description").value = dataset?.description || "";
  field(elements.datasetForm, "schema_name").value = dataset?.schema_name || "alpaca";
}

function renderDatasets() {
  elements.datasetList.innerHTML = "";
  if (state.datasets.length === 0) {
    elements.datasetList.innerHTML = '<div class="empty-state">No datasets yet. Create one to start collecting examples.</div>';
    return;
  }
  for (const dataset of state.datasets) {
    const button = document.createElement("button");
    button.className = `dataset-item ${state.selectedDataset?.id === dataset.id ? "active" : ""}`;
    button.innerHTML = `<span class="dataset-name"></span><span class="dataset-meta"></span>`;
    button.querySelector(".dataset-name").textContent = dataset.name;
    button.querySelector(".dataset-meta").textContent = `${dataset.schema_name} • ${dataset.example_count} examples • ${dataset.token_total} tokens`;
    button.addEventListener("click", () => selectDataset(dataset.id));
    elements.datasetList.appendChild(button);
  }
}

function renderExamples() {
  elements.exampleTable.innerHTML = "";
  const examples = filterExamples();

  if (!state.selectedDataset) {
    elements.exampleTable.innerHTML = '<div class="empty-state">Select a dataset to review examples.</div>';
    return;
  }

  if (examples.length === 0) {
    const message = state.examples.length === 0
      ? "No examples yet. Add one manually or use an import tool above."
      : "No examples match the current filter.";
    elements.exampleTable.innerHTML = `<div class="empty-state">${message}</div>`;
    return;
  }

  const visibleIds = new Set(examples.map((example) => example.id));
  if (state.selectedExample && !visibleIds.has(state.selectedExample.id)) {
    state.selectedExample = examples[0] || null;
  }

  for (const example of examples) {
    const row = document.createElement("button");
    row.className = `example-row ${state.selectedExample?.id === example.id ? "active" : ""}`;
    row.innerHTML = `<strong></strong><span></span>`;
    row.querySelector("strong").textContent = example.instruction || "Untitled example";
    row.querySelector("span").textContent = `${example.status} • ${example.token_count} tokens • ${example.labels.join(", ") || "no labels"}`;
    row.addEventListener("click", () => selectExample(example.id));
    elements.exampleTable.appendChild(row);
  }
}

function renderSummary() {
  const dataset = state.selectedDataset;
  elements.datasetTitle.textContent = dataset ? dataset.name : "Select or create a dataset";
  elements.datasetDescription.textContent = dataset
    ? dataset.description || "No description provided."
    : "Use the controls on the left to create a dataset, import records, curate them, and export for training.";
  elements.activeDatasetMeta.textContent = dataset
    ? `${dataset.schema_name} schema • ${dataset.example_count} examples • ${dataset.token_total} tokens`
    : "Create a dataset or select one to start working.";
  elements.datasetSchemaBadge.textContent = dataset?.schema_name ? `${dataset.schema_name} schema` : "No schema";
  elements.statExamples.textContent = dataset?.example_count ?? 0;
  elements.statTokens.textContent = dataset?.token_total ?? 0;
  elements.statSchema.textContent = dataset?.schema_name ?? "-";
  setDatasetForm(dataset);
  updateActionState();
}

function renderAnnotationForm() {
  const example = state.selectedExample;
  field(elements.annotationForm, "id").value = example?.id || "";
  field(elements.annotationForm, "instruction").value = example?.instruction || "";
  field(elements.annotationForm, "input_text").value = example?.input_text || "";
  field(elements.annotationForm, "output_text").value = example?.output_text || "";
  field(elements.annotationForm, "system_prompt").value = example?.system_prompt || "";
  field(elements.annotationForm, "labels").value = example?.labels?.join(", ") || "";
  field(elements.annotationForm, "status").value = example?.status || "draft";
  updateActionState();
}

async function loadDatasets(preferredId = null) {
  state.datasets = await request("/api/datasets");
  if (state.datasets.length === 0) {
    state.selectedDataset = null;
    state.examples = [];
    state.selectedExample = null;
    renderDatasets();
    renderSummary();
    renderExamples();
    renderAnnotationForm();
    return;
  }
  const requestedId = preferredId || state.selectedDataset?.id;
  const matchedDataset = state.datasets.find((dataset) => dataset.id === requestedId);
  renderDatasets();
  await selectDataset((matchedDataset || state.datasets[0]).id);
}

async function selectDataset(datasetId) {
  state.selectedDataset = await request(`/api/datasets/${datasetId}`);
  renderSummary();
  renderDatasets();
  state.examples = await request(`/api/datasets/${datasetId}/examples`);
  state.selectedExample = state.examples[0] || null;
  renderExamples();
  renderAnnotationForm();
}

function selectExample(exampleId) {
  state.selectedExample = state.examples.find((example) => example.id === exampleId) || null;
  renderExamples();
  renderAnnotationForm();
}

elements.exampleSearch.addEventListener("input", (event) => {
  state.exampleFilters.query = event.target.value;
  renderExamples();
  renderAnnotationForm();
});

elements.exampleStatusFilter.addEventListener("change", (event) => {
  state.exampleFilters.status = event.target.value;
  renderExamples();
  renderAnnotationForm();
});

elements.workspaceTabs.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveWorkspaceTab(button.dataset.tabTarget);
  });
});

async function refreshDatasetState() {
  if (!state.selectedDataset) {
    await loadDatasets();
    return;
  }
  await loadDatasets(state.selectedDataset.id);
}

elements.datasetForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = formToJson(elements.datasetForm);
  try {
    const created = await request("/api/datasets", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    await loadDatasets(created.id);
  } catch (error) {
    alert(error.message);
  }
});

elements.datasetUpdateButton.addEventListener("click", async () => {
  if (!ensureDataset()) {
    return;
  }
  try {
    await request(`/api/datasets/${state.selectedDataset.id}`, {
      method: "PUT",
      body: JSON.stringify(formToJson(elements.datasetForm)),
    });
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.datasetDeleteButton.addEventListener("click", async () => {
  if (!ensureDataset()) {
    return;
  }
  if (!confirm(`Delete dataset ${state.selectedDataset.name}?`)) {
    return;
  }
  try {
    await request(`/api/datasets/${state.selectedDataset.id}`, { method: "DELETE" });
    await loadDatasets();
  } catch (error) {
    alert(error.message);
  }
});

elements.refreshDatasets.addEventListener("click", () => loadDatasets(state.selectedDataset?.id));

elements.manualExampleForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  const payload = formToJson(elements.manualExampleForm);
  payload.labels = parseLabels(payload.labels);
  payload.metadata = {};
  payload.conversation = [];
  try {
    await request(`/api/datasets/${state.selectedDataset.id}/examples`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.manualExampleForm.reset();
    setActiveWorkspaceTab("review");
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  const formData = new FormData(elements.uploadForm);
  try {
    const result = await request(`/api/datasets/${state.selectedDataset.id}/import`, {
      method: "POST",
      body: formData,
    });
    elements.curationResult.textContent = `Imported ${result.imported} examples.`;
    elements.uploadForm.reset();
    setActiveWorkspaceTab("review");
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.syntheticForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  const payload = formToJson(elements.syntheticForm);
  payload.count = Number(payload.count || 1);
  payload.temperature = Number(payload.temperature || 0.7);
  try {
    const result = await request(`/api/datasets/${state.selectedDataset.id}/synthetic`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.curationResult.textContent = `Generated ${result.generated} examples.`;
    setActiveWorkspaceTab("review");
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.curationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  const payload = formToJson(elements.curationForm);
  payload.drop_empty = field(elements.curationForm, "drop_empty").checked;
  payload.deduplicate = field(elements.curationForm, "deduplicate").checked;
  try {
    const result = await request(`/api/datasets/${state.selectedDataset.id}/curate`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.curationResult.textContent = `Updated ${result.updated} rows, removed ${result.removed}, total ${result.example_count} examples / ${result.token_total} tokens.`;
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.annotationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedExample) {
    alert("Select an example first.");
    return;
  }
  const payload = formToJson(elements.annotationForm);
  payload.labels = parseLabels(payload.labels);
  payload.metadata = state.selectedExample.metadata || {};
  payload.conversation = state.selectedExample.conversation || [];
  try {
    await request(`/api/examples/${state.selectedExample.id}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.deleteExampleButton.addEventListener("click", async () => {
  if (!state.selectedExample) {
    return;
  }
  if (!confirm("Delete the selected example?")) {
    return;
  }
  try {
    await request(`/api/examples/${state.selectedExample.id}`, { method: "DELETE" });
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.exportForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  try {
    const format = field(elements.exportForm, "export_format").value;
    const result = await request(`/api/datasets/${state.selectedDataset.id}/export?export_format=${encodeURIComponent(format)}`);
    const url = URL.createObjectURL(result.blob);
    const link = document.createElement("a");
    const disposition = result.response.headers.get("content-disposition") || "attachment; filename=dataset.json";
    const match = disposition.match(/filename="?([^\"]+)"?/);
    link.href = url;
    link.download = match ? match[1] : `dataset.${format}`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    alert(error.message);
  }
});

elements.searxngForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  const payload = formToJson(elements.searxngForm);
  payload.limit = Number(payload.limit || 5);
  payload.safesearch = 1;
  payload.crawl_pages = field(elements.searxngForm, "crawl_pages").checked;
  payload.verify_ssl = field(elements.searxngForm, "verify_ssl").checked;
  payload.labels = parseLabels(payload.labels);
  try {
    const result = await request(`/api/datasets/${state.selectedDataset.id}/sources/searxng`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.integrationResult.textContent = `Imported ${result.imported} records from SearxNG.`;
    setActiveWorkspaceTab("review");
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.webImportForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  const payload = formToJson(elements.webImportForm);
  payload.max_pages = Number(payload.max_pages || 5);
  payload.max_depth = Number(payload.max_depth || 1);
  payload.same_domain_only = field(elements.webImportForm, "same_domain_only").checked;
  payload.verify_ssl = field(elements.webImportForm, "verify_ssl").checked;
  payload.labels = parseLabels(payload.labels);
  try {
    const result = await request(`/api/datasets/${state.selectedDataset.id}/sources/web`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.integrationResult.textContent = `Imported ${result.imported} records from the web crawler.`;
    setActiveWorkspaceTab("review");
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

elements.githubForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!ensureDataset()) {
    return;
  }
  const payload = formToJson(elements.githubForm);
  payload.limit = Number(payload.limit || 5);
  payload.include_readme = field(elements.githubForm, "include_readme").checked;
  payload.verify_ssl = field(elements.githubForm, "verify_ssl").checked;
  payload.labels = parseLabels(payload.labels);
  if (!payload.token) {
    delete payload.token;
  }
  try {
    const result = await request(`/api/datasets/${state.selectedDataset.id}/sources/github`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    elements.integrationResult.textContent = `Imported ${result.imported} records from GitHub.`;
    setActiveWorkspaceTab("review");
    await refreshDatasetState();
  } catch (error) {
    alert(error.message);
  }
});

setActiveWorkspaceTab(state.activeWorkspaceTab);
loadDatasets();
