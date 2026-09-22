import type maplibregl from "maplibre-gl";
import type { ZarrLayer } from "@carbonplan/zarr-layer";
import { readPoint } from "./embeddings";

const MAX_ROUNDS = 4;

export interface BackgroundSample {
  lng: number;
  lat: number;
  embedding: number[];
}

/** The NLCD class code at a point, or undefined outside NLCD's coverage. */
export async function nlcdClassAt(
  nlcd: ZarrLayer,
  lng: number,
  lat: number,
): Promise<number | undefined> {
  try {
    const result = await nlcd.queryData(
      { type: "Point", coordinates: [lng, lat] },
      undefined,
      { includeSpatialCoordinates: false },
    );
    return (result as { land_cover?: number[] }).land_cover?.[0];
  } catch {
    return undefined;
  }
}

/**
 * Embeddings of random pixels in view that NLCD assigns to `nlcdClass`.
 * These stand in for "ordinary" examples of the parent class, so a new
 * subclass is learned against what it is usually confused with.
 */
export async function sampleBackground(
  map: maplibregl.Map,
  aef: ZarrLayer,
  nlcd: ZarrLayer,
  nlcdClass: number,
  count: number,
): Promise<BackgroundSample[]> {
  const bounds = map.getBounds();
  const [west, south, east, north] = [
    bounds.getWest(),
    bounds.getSouth(),
    bounds.getEast(),
    bounds.getNorth(),
  ];
  const kept: Array<[number, number]> = [];

  for (let round = 0; round < MAX_ROUNDS && kept.length < count; round++) {
    const candidates = Array.from(
      { length: (count - kept.length) * 3 },
      (): [number, number] => [
        west + Math.random() * (east - west),
        south + Math.random() * (north - south),
      ],
    );
    const classes = await Promise.all(
      candidates.map(([lng, lat]) => nlcdClassAt(nlcd, lng, lat)),
    );
    candidates.forEach((c, i) => {
      if (classes[i] === nlcdClass && kept.length < count) kept.push(c);
    });
  }

  const samples = await Promise.all(
    kept.map(async ([lng, lat]) => {
      const embedding = await readPoint(aef, lng, lat).catch(() => null);
      return embedding ? { lng, lat, embedding } : null;
    }),
  );
  return samples.filter((s): s is BackgroundSample => s !== null);
}
