import type { Model } from "./classifier";
import type { BackgroundSample } from "./background";

const STORAGE_KEY = "live-classify:trainings";
const CURRENT_KEY = "live-classify:current";

export interface TrainingLabel {
  lng: number;
  lat: number;
  positive: boolean;
  embeddings: number[][];
}

export interface Training {
  name: string;
  /** Calendar year whose embeddings the labels were read from. */
  year: number;
  backgroundClass: number;
  view: { center: [number, number]; zoom: number };
  labels: TrainingLabel[];
  background: BackgroundSample[];
  model: Model | null;
  savedAt: string;
}

/**
 * On disk, embeddings go back to the int8 values AEF stores, which the
 * signed-square dequantization maps to and from exactly.
 */
interface StoredTraining extends Training {
  version: 1;
}

const quantize = (v: number) =>
  Math.round(Math.sign(v) * Math.sqrt(Math.abs(v)) * 127.5);
const dequantize = (q: number) => Math.sign(q) * (q / 127.5) ** 2;
const mapAll = (rows: number[][], f: (v: number) => number) =>
  rows.map((row) => row.map(f));

export function toStored(training: Training): StoredTraining {
  return {
    ...training,
    version: 1,
    labels: training.labels.map((l) => ({
      ...l,
      embeddings: mapAll(l.embeddings, quantize),
    })),
    background: training.background.map((b) => ({
      ...b,
      embedding: b.embedding.map(quantize),
    })),
  };
}

export function fromStored(stored: StoredTraining): Training {
  if (stored.version !== 1) {
    throw new Error(`Unsupported training version ${stored.version}`);
  }
  const { version: _version, ...rest } = stored;
  return {
    ...rest,
    labels: stored.labels.map((l) => ({
      ...l,
      embeddings: mapAll(l.embeddings, dequantize),
    })),
    background: stored.background.map((b) => ({
      ...b,
      embedding: b.embedding.map(dequantize),
    })),
  };
}

function readAll(): StoredTraining[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

function writeAll(trainings: StoredTraining[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trainings));
  } catch (err) {
    throw new Error(`Browser storage is unavailable or full: ${(err as Error).message}`);
  }
}

export function listSaved(): string[] {
  return readAll().map((t) => t.name);
}

export function loadSaved(name: string): Training | null {
  const stored = readAll().find((t) => t.name === name);
  return stored ? fromStored(stored) : null;
}

/** Saves under the training's name, replacing any training of that name. */
export function saveTraining(training: Training) {
  const others = readAll().filter((t) => t.name !== training.name);
  writeAll([...others, toStored(training)]);
}

/** The in-progress training, kept so a page reload does not lose it. */
export function saveCurrent(training: Training) {
  try {
    localStorage.setItem(CURRENT_KEY, JSON.stringify(toStored(training)));
  } catch (err) {
    console.warn("Could not autosave training", err);
  }
}

export function loadCurrent(): Training | null {
  try {
    const raw = localStorage.getItem(CURRENT_KEY);
    return raw ? fromStored(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

export function deleteSaved(name: string) {
  writeAll(readAll().filter((t) => t.name !== name));
}

export function downloadTraining(training: Training) {
  const blob = new Blob([JSON.stringify(toStored(training))], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${training.name.replace(/[^\w-]+/g, "_") || "training"}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function readTrainingFile(file: File): Promise<Training> {
  return fromStored(JSON.parse(await file.text()));
}
