import maplibregl from "maplibre-gl";
import { ZarrLayer } from "@carbonplan/zarr-layer";
import { createNlcdLayer, NLCD_CLASSES } from "./nlcd";
import { NUM_BANDS, readPatch, readPoint } from "./embeddings";
import { trainLogistic, type Example, type Model } from "./classifier";
import {
  nlcdClassAt,
  sampleBackground,
  type BackgroundSample,
} from "./background";
import {
  deleteSaved,
  downloadTraining,
  listSaved,
  loadCurrent,
  loadSaved,
  readTrainingFile,
  saveCurrent,
  saveTraining,
  type Training,
} from "./trainings";

const SOURCE = "https://data.source.coop/tge-labs/aef-mosaic";
const YEAR_ORIGIN = 2017;
// The latest AEF year, matching the NLCD 2025 land cover shown beneath it.
const YEAR = 2025;
// Full-resolution only (no pyramid): each 256px region is 4 MB of int8 bands
// and ~2.3 MB to download, and a zoom-12 viewport is ~50 regions.
const MIN_ZOOM = 12;
// Band indices for the false-color view, following Google's AEF examples.
const RGB_BANDS = [1, 16, 9];
const BACKGROUND_COUNT = 200;
// Background pixels are unlabeled guesses from NLCD, so they count for less
// than a pixel the user marked.
const BACKGROUND_WEIGHT = 0.3;

const BAND_INDICES = Array.from({ length: NUM_BANDS }, (_, i) => i);
const queryUniform = (i: number) => `u_q${i}`;
const MODES = { rgb: 0, similarity: 1, classify: 2 } as const;
type Mode = keyof typeof MODES;

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const modeSelect = $<HTMLSelectElement>("mode");
const hint = $<HTMLDivElement>("hint");
const threshold = $<HTMLInputElement>("threshold");
const thresholdLabel = $<HTMLSpanElement>("threshold-label");
const thresholdValue = $<HTMLOutputElement>("threshold-value");
const opacity = $<HTMLInputElement>("opacity");
const opacityValue = $<HTMLOutputElement>("opacity-value");
const samplesLabel = $<HTMLSpanElement>("samples");
const clearButton = $<HTMLButtonElement>("clear");
const status = $<HTMLParagraphElement>("status");
const classifyControls = $<HTMLFieldSetElement>("classify-controls");
const className = $<HTMLInputElement>("class-name");
const backgroundClass = $<HTMLSelectElement>("background-class");
const backgroundCount = $<HTMLSpanElement>("background-count");
const resampleButton = $<HTMLButtonElement>("resample");
const showBackground = $<HTMLInputElement>("show-background");
const backgroundSummary = $<HTMLElement>("background-summary");
const modelStats = $<HTMLParagraphElement>("model-stats");
const saveButton = $<HTMLButtonElement>("save");
const exportButton = $<HTMLButtonElement>("export");
const importButton = $<HTMLButtonElement>("import");
const importFile = $<HTMLInputElement>("import-file");
const savedSelect = $<HTMLSelectElement>("saved");
const loadButton = $<HTMLButtonElement>("load");
const deleteButton = $<HTMLButtonElement>("delete");

for (const { code, name } of NLCD_CLASSES) {
  backgroundClass.add(new Option(`${code} ${name}`, String(code)));
}
backgroundClass.value = "42";

// Restore the in-progress training before the layer and map read the year
// and view from the page.
const restored = loadCurrent();
if (restored) {
  className.value = restored.name;
  backgroundClass.value = String(restored.backgroundClass);
}

// Raw int8 values arrive as floats (fill already NaN). Dequantize with
// (x / 127.5)^2 * sign(x); embeddings are unit length, so a dot product
// against a unit-length query is the cosine similarity.
const dequant = (band: string) => `(${band} / 127.5) * abs(${band} / 127.5)`;

const dotTerms = BAND_INDICES.map(
  (i) => `  linear += ${dequant(`band_${i}`)} * ${queryUniform(i)};`,
).join("\n");

const [r, g, b] = RGB_BANDS.map((i) => dequant(`band_${i}`));

// The finding modes draw nothing until there is a query, so the imagery shows
// through. Similarity mode scores the dot product with a query embedding; classify
// mode scores sigmoid(dot + bias) with the trained weights in the same
// uniforms, so both highlight pixels whose score clears the threshold.
const customFrag = `
  uniform float u_mode;
  uniform float u_threshold;
  uniform float u_hasQuery;
  uniform float u_bias;

  if (isnan(band_0)) {
    discard;
  }

  vec3 rgb = clamp((vec3(${r}, ${g}, ${b}) + 0.3) / 0.6, 0.0, 1.0);

  if (u_mode < 0.5) {
    fragColor = vec4(rgb * opacity, opacity);
  } else if (u_hasQuery < 0.5) {
    discard;
  } else {
    float linear = 0.0;
${dotTerms}
    float score = u_mode > 1.5 ? 1.0 / (1.0 + exp(-(linear + u_bias))) : linear;
    if (score < u_threshold) {
      discard;
    }
    float t = (score - u_threshold) / max(1.0 - u_threshold, 1e-3);
    vec4 c = texture(colormap, vec2(clamp(t, 0.0, 1.0), 0.5));
    fragColor = vec4(c.rgb * opacity, opacity);
  }
`;

type Sample = { embedding: number[]; marker: maplibregl.Marker };
const samples: Sample[] = [];

type Label = {
  lng: number;
  lat: number;
  positive: boolean;
  embeddings: number[][];
  marker: maplibregl.Marker;
};
const labels: Label[] = [];
let background: BackgroundSample[] = [];
let model: Model | null = null;

// Each mode keeps its own threshold: a cosine similarity and a probability
// live on different scales.
const thresholds: Record<Mode, number> = {
  rgb: 0.8,
  similarity: 0.8,
  classify: 0.5,
};

const mode = () => modeSelect.value as Mode;
threshold.value = String(thresholds[mode()]);

function uniforms(): Record<string, number> {
  const current = mode();
  const vector = current === "classify" ? model?.weights : meanQuery();
  const values: Record<string, number> = {
    u_mode: MODES[current],
    u_threshold: Number(threshold.value),
    u_hasQuery: vector ? 1 : 0,
    u_bias: current === "classify" ? (model?.bias ?? 0) : 0,
  };
  BAND_INDICES.forEach((i) => {
    values[queryUniform(i)] = vector?.[i] ?? 0;
  });
  return values;
}

function meanQuery(): number[] | null {
  if (samples.length === 0) return null;
  const mean = new Array(NUM_BANDS).fill(0);
  for (const { embedding } of samples) {
    embedding.forEach((v, i) => (mean[i] += v));
  }
  const norm = Math.hypot(...mean) || 1;
  return mean.map((v) => v / norm);
}

const selector = (yearIdx: number) => ({
  time: { selected: yearIdx, type: "index" as const },
  band: { selected: BAND_INDICES, type: "index" as const },
});

const layer = new ZarrLayer({
  id: "aef",
  source: SOURCE,
  variable: "embeddings",
  selector: selector(YEAR - YEAR_ORIGIN),
  minzoom: MIN_ZOOM,
  clim: [0, 1],
  colormap: ["#fde725", "#f89540", "#e1325a", "#9c179e"],
  opacity: Number(opacity.value),
  customFrag,
  uniforms: uniforms(),
  onLoadingStateChange: ({ loading, error }) => {
    if (error) status.textContent = `Error: ${error.message}`;
    else status.textContent = loading ? "Loading embeddings…" : zoomHint();
  },
});

const map = new maplibregl.Map({
  container: "map",
  center: restored?.view.center ?? [-121.75, 45.33],
  zoom: restored?.view.zoom ?? 14,
  maxZoom: 18,
  // Shift-click adds samples; box zoom would swallow it.
  boxZoom: false,
  style: {
    version: 8,
    sources: {
      imagery: {
        type: "raster",
        tiles: [
          "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        maxzoom: 16,
        attribution: "Imagery: USGS The National Map",
      },
    },
    layers: [{ id: "imagery", type: "raster", source: "imagery" }],
  },
});

const nlcdVisible = $<HTMLInputElement>("nlcd-visible");
const nlcdOpacity = $<HTMLInputElement>("nlcd-opacity");
const nlcdOpacityValue = $<HTMLOutputElement>("nlcd-opacity-value");
nlcdOpacityValue.textContent = nlcdOpacity.value;
const nlcdLayer = createNlcdLayer(Number(nlcdOpacity.value));

function zoomHint(): string {
  return map.getZoom() < MIN_ZOOM
    ? "Zoom in closer, until you can see individual fields and streets."
    : "";
}

function refreshUniforms() {
  layer.setUniforms(uniforms());
}

function addMarker(lngLat: maplibregl.LngLat, kind?: "positive" | "negative") {
  const el = document.createElement("div");
  el.className = kind ? `sample-marker ${kind}` : "sample-marker";
  return new maplibregl.Marker({ element: el }).setLngLat(lngLat).addTo(map);
}

const HINTS: Record<Mode, { intro: string; steps: string[] }> = {
  classify: {
    intro:
      "Teach the map a new land cover class by example. Everything runs in the browser.",
    steps: [
      "Zoom in until you can see individual fields and streets, and name your class.",
      "Click a few examples of it (green).",
      "Shift-click things that aren't it (red), especially wrong highlights.",
      "Raise the match threshold for a stricter map.",
    ],
  },
  similarity: {
    intro: "Highlight everything that looks like a spot you pick.",
    steps: [
      "Zoom in until you can see individual fields and streets, and click a spot.",
      "Shift-click more spots to search for their average.",
      "Adjust the match threshold to widen or narrow the match.",
    ],
  },
  rgb: {
    intro:
      "Three of the 64 embedding dimensions shown as red, green and blue. Similar colors mean similar embeddings.",
    steps: [],
  },
};

function renderHint(current: Mode) {
  const { intro, steps } = HINTS[current];
  const p = document.createElement("p");
  p.textContent = intro;
  const ol = document.createElement("ol");
  for (const step of steps) {
    const li = document.createElement("li");
    li.textContent = step;
    ol.append(li);
  }
  hint.replaceChildren(p, ...(steps.length ? [ol] : []));
}

function updatePanel() {
  const current = mode();
  classifyControls.hidden = current !== "classify";
  thresholdLabel.textContent =
    current === "classify" ? "Match threshold (probability)" : "Match threshold";
  renderHint(current);

  if (current === "classify") {
    const pos = labels.filter((l) => l.positive);
    const neg = labels.filter((l) => !l.positive);
    const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
    samplesLabel.textContent = `${plural(pos.length, "example")} · ${neg.length} not`;
    backgroundCount.textContent = `${background.length} random spots used`;
    const chosen =
      NLCD_CLASSES.find((c) => c.code === Number(backgroundClass.value))?.name ??
      "";
    backgroundSummary.textContent = background.length
      ? `Comparing against: ${chosen} · change`
      : pos.length
        ? `Comparing against: ${chosen} (none in view) · change`
        : "Comparing against: set by your first example";
    modelStats.textContent = model
      ? `Learning from ${plural(pos.length, "example")}, ${plural(neg.length, "counter-example")} and ${background.length} random spots.`
      : "Click an example to start.";
  } else {
    samplesLabel.textContent =
      samples.length === 0
        ? "No samples"
        : `${samples.length} sample${samples.length > 1 ? "s" : ""}`;
  }
}

function retrain() {
  const examples: Example[] = [
    ...labels.flatMap((l) =>
      l.embeddings.map((x): Example => ({
        x,
        y: l.positive ? 1 : 0,
        weight: 1,
      })),
    ),
    ...background.map(
      (b): Example => ({ x: b.embedding, y: 0, weight: BACKGROUND_WEIGHT }),
    ),
  ];
  model = trainLogistic(examples);
  refreshUniforms();
  updatePanel();
  drawBackground();
  saveCurrent(currentTraining());
}

function currentTraining(): Training {
  const center = map.getCenter();
  return {
    name: className.value || "Untitled",
    year: YEAR,
    backgroundClass: Number(backgroundClass.value),
    view: { center: [center.lng, center.lat], zoom: map.getZoom() },
    labels: labels.map(({ lng, lat, positive, embeddings }) => ({
      lng,
      lat,
      positive,
      embeddings,
    })),
    background,
    model,
    savedAt: new Date().toISOString(),
  };
}

function applyTraining(training: Training) {
  clearLabels();
  for (const { lng, lat, positive, embeddings } of training.labels) {
    labels.push({
      lng,
      lat,
      positive,
      embeddings,
      marker: addMarker(
        new maplibregl.LngLat(lng, lat),
        positive ? "positive" : "negative",
      ),
    });
  }
  background = training.background;
  className.value = training.name;
  backgroundClass.value = String(training.backgroundClass);
  map.jumpTo({ center: training.view.center, zoom: training.view.zoom });
  retrain();
}

const BACKGROUND_SOURCE = "background-spots";

function drawBackground() {
  const source = map.getSource(BACKGROUND_SOURCE) as
    | maplibregl.GeoJSONSource
    | undefined;
  source?.setData({
    type: "FeatureCollection",
    features: background.map(({ lng, lat }) => ({
      type: "Feature",
      properties: {},
      geometry: { type: "Point", coordinates: [lng, lat] },
    })),
  });
}

function setBackgroundVisibility() {
  if (!map.getLayer(BACKGROUND_SOURCE)) return;
  const visible = showBackground.checked && mode() === "classify";
  map.setLayoutProperty(BACKGROUND_SOURCE, "visibility", visible ? "visible" : "none");
}

function refreshSaved(selected?: string) {
  savedSelect.replaceChildren(
    ...listSaved().map((name) => new Option(name, name)),
  );
  if (selected) savedSelect.value = selected;
  loadButton.disabled = deleteButton.disabled = savedSelect.options.length === 0;
}

async function resampleBackground() {
  status.textContent = "Sampling NLCD background…";
  try {
    background = await sampleBackground(
      map,
      layer,
      await nlcdLayer,
      Number(backgroundClass.value),
      BACKGROUND_COUNT,
    );
    const chosen = backgroundClass.selectedOptions[0]?.text ?? "that class";
    status.textContent = background.length
      ? zoomHint()
      : `No "${chosen}" in view to compare against. Pick another land cover above.`;
  } catch (err) {
    status.textContent = `Background sampling failed: ${(err as Error).message}`;
    console.error(err);
  }
  retrain();
}

async function onSimilarityClick(e: maplibregl.MapMouseEvent) {
  const embedding = await readPoint(layer, e.lngLat.lng, e.lngLat.lat);
  if (!embedding) {
    status.textContent = "No embedding at that point";
    return;
  }
  if (!e.originalEvent.shiftKey) clearSamples();
  samples.push({ embedding, marker: addMarker(e.lngLat) });
  refreshUniforms();
  updatePanel();
}

async function onClassifyClick(e: maplibregl.MapMouseEvent) {
  const embeddings = await readPatch(layer, e.lngLat.lng, e.lngLat.lat);
  if (embeddings.length === 0) {
    status.textContent = "No embedding at that point";
    return;
  }
  const positive = !e.originalEvent.shiftKey;
  const firstExample = positive && !labels.some((l) => l.positive);
  if (firstExample && background.length === 0) {
    const code = await nlcdClassAt(await nlcdLayer, e.lngLat.lng, e.lngLat.lat);
    if (NLCD_CLASSES.some((c) => c.code === code)) {
      backgroundClass.value = String(code);
    }
  }
  labels.push({
    lng: e.lngLat.lng,
    lat: e.lngLat.lat,
    positive,
    embeddings,
    marker: addMarker(e.lngLat, positive ? "positive" : "negative"),
  });
  if (background.length === 0) await resampleBackground();
  else retrain();
}

map.on("click", async (e) => {
  if (map.getZoom() < MIN_ZOOM || mode() === "rgb") return;
  status.textContent = "Reading embedding…";
  try {
    if (mode() === "classify") await onClassifyClick(e);
    else await onSimilarityClick(e);
    if (status.textContent === "Reading embedding…") {
      status.textContent = zoomHint();
    }
  } catch (err) {
    status.textContent = `Query failed: ${(err as Error).message}`;
    console.error(err);
  }
});

function clearSamples() {
  for (const { marker } of samples) marker.remove();
  samples.length = 0;
}

function clearLabels() {
  for (const { marker } of labels) marker.remove();
  labels.length = 0;
  background = [];
  model = null;
}

clearButton.addEventListener("click", () => {
  if (mode() === "classify") {
    clearLabels();
    saveCurrent(currentTraining());
  } else {
    clearSamples();
  }
  refreshUniforms();
  updatePanel();
});

saveButton.addEventListener("click", () => {
  const training = currentTraining();
  try {
    saveTraining(training);
    refreshSaved(training.name);
    status.textContent = `Saved "${training.name}"`;
  } catch (err) {
    status.textContent = (err as Error).message;
  }
});
exportButton.addEventListener("click", () => downloadTraining(currentTraining()));
importButton.addEventListener("click", () => importFile.click());
importFile.addEventListener("change", async () => {
  const file = importFile.files?.[0];
  importFile.value = "";
  if (!file) return;
  try {
    applyTraining(await readTrainingFile(file));
    status.textContent = `Imported "${file.name}"`;
  } catch (err) {
    status.textContent = `Import failed: ${(err as Error).message}`;
  }
});
loadButton.addEventListener("click", () => {
  const training = loadSaved(savedSelect.value);
  if (training) applyTraining(training);
});
deleteButton.addEventListener("click", () => {
  deleteSaved(savedSelect.value);
  refreshSaved();
});

resampleButton.addEventListener("click", resampleBackground);
showBackground.addEventListener("change", setBackgroundVisibility);
backgroundClass.addEventListener("change", resampleBackground);
className.addEventListener("input", () => {
  updatePanel();
  saveCurrent(currentTraining());
});


let previousMode = mode();
modeSelect.addEventListener("change", () => {
  thresholds[previousMode] = Number(threshold.value);
  previousMode = mode();
  threshold.value = String(thresholds[previousMode]);
  thresholdValue.textContent = Number(threshold.value).toFixed(3);
  for (const s of samples) s.marker.getElement().hidden = mode() !== "similarity";
  for (const l of labels) l.marker.getElement().hidden = mode() !== "classify";
  setBackgroundVisibility();
  refreshUniforms();
  updatePanel();
});

threshold.addEventListener("input", () => {
  thresholdValue.textContent = Number(threshold.value).toFixed(3);
  refreshUniforms();
});
thresholdValue.textContent = Number(threshold.value).toFixed(3);

opacity.addEventListener("input", () => {
  opacityValue.textContent = opacity.value;
  layer.setOpacity(Number(opacity.value));
});
opacityValue.textContent = opacity.value;

map.on("zoomend", () => {
  status.textContent = zoomHint();
});

nlcdVisible.addEventListener("change", () => {
  map.setLayoutProperty("nlcd", "visibility", nlcdVisible.checked ? "visible" : "none");
});
nlcdOpacity.addEventListener("input", async () => {
  nlcdOpacityValue.textContent = nlcdOpacity.value;
  (await nlcdLayer).setOpacity(Number(nlcdOpacity.value));
});

map.on("load", async () => {
  map.addLayer(layer);
  map.addSource(BACKGROUND_SOURCE, {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  map.addLayer({
    id: BACKGROUND_SOURCE,
    type: "circle",
    source: BACKGROUND_SOURCE,
    paint: {
      "circle-radius": 3,
      "circle-color": "#ffffff",
      "circle-stroke-color": "#1f2937",
      "circle-stroke-width": 1,
      "circle-opacity": 0.85,
    },
  });
  drawBackground();
  setBackgroundVisibility();
  try {
    map.addLayer(await nlcdLayer, "aef");
    if (!nlcdVisible.checked) map.setLayoutProperty("nlcd", "visibility", "none");
  } catch (err) {
    status.textContent = `NLCD failed: ${(err as Error).message}`;
    console.error(err);
  }
});

if (restored) applyTraining(restored);
refreshSaved();
updatePanel();

Object.assign(window, { map, layer, nlcdLayer });
