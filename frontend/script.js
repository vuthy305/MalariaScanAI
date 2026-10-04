// If the page is served by FastAPI, call the same server.
// If index.html is opened directly from disk, call the local backend.
const API = location.protocol === "file:" ? "http://127.0.0.1:8000" : location.origin;

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const hint = document.getElementById("dropzone-hint");
const button = document.getElementById("recognize");
const message = document.getElementById("message");
const word = document.getElementById("word");
const confidence = document.getElementById("confidence");
const top3 = document.getElementById("top3");
const explain = document.getElementById("explain");
const heatmap = document.getElementById("heatmap");
const meaning = document.getElementById("meaning");

// General information shown with each result. It is fixed text, not generated per image.
const MEANING = {
  PARASITIZED: "The model found a pattern that matches a malaria parasite. After staining, a parasite " +
    "usually shows as a dark purple dot or ring inside the cell. Next step: a trained health worker " +
    "should confirm with microscopy or a rapid diagnostic test.",
  UNINFECTED: "The model found no pattern that matches a parasite in this cell. One clear cell does not " +
    "rule out malaria: many cells must be checked, and a person with symptoms still needs a proper test.",
};

let selectedFile = null;

const percent = (value) => (value * 100).toFixed(1) + "%";
const NAMES = { PARASITIZED: "Parasitized", UNINFECTED: "Uninfected" };
const pretty = (label) => NAMES[label] || label.replaceAll("_", " ");

function showError(text) {
  message.textContent = text;
  message.hidden = false;
}

function selectFile(file) {
  message.hidden = true;
  if (!file || !file.type.startsWith("image/")) {
    showError("That file is not an image. Choose a JPG or PNG.");
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  hint.hidden = true;
  button.disabled = false;
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragging");
  selectFile(e.dataTransfer.files[0]);
});

button.addEventListener("click", async () => {
  if (!selectedFile) return;
  message.hidden = true;
  button.disabled = true;
  button.textContent = "Checking…";

  const body = new FormData();
  body.append("file", selectedFile);

  try {
    const response = await fetch(API + "/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The server returned an error.");

    // Below 80% confidence the model is close to guessing, so do not show a class.
    const unsure = data.confidence < 0.8;
    word.textContent = unsure ? "Not sure" : pretty(data.prediction);
    word.classList.toggle("alert", !unsure && data.prediction === "PARASITIZED");
    confidence.innerHTML = "Confidence: <strong>" + percent(data.confidence) + "</strong>" +
      (unsure ? "<br>Too low to trust. Please upload a clear image of one stained blood cell." : "");
        // Heatmap and explanation are only shown when the model is confident enough.
    explain.hidden = unsure || !data.heatmap;
    if (!explain.hidden) {
      heatmap.src = data.heatmap;
      meaning.textContent = MEANING[data.prediction] || "";
    }
      top3.innerHTML = "";
    data.top3.forEach((item, i) => {
      const li = document.createElement("li");
      li.innerHTML =
        '<div class="row"><span>' + (i + 1) + ". " + pretty(item.label) + "</span><span>" +
        percent(item.confidence) + '</span></div><div class="bar"><div class="fill" style="width:' +
        item.confidence * 100 + '%"></div></div>';
      top3.appendChild(li);
    });
  } catch (error) {
    showError(error instanceof TypeError
      ? "Cannot reach the backend. Start it with: uvicorn backend.main:app --reload"
      : error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Check Cell";
  }
});