import { IcechunkStore } from "icechunk-js";
import { ZarrLayer, codecRegistry } from "@carbonplan/zarr-layer";

// VirtualTIFF records the COG's deflate tiles under the numcodecs name.
const zlib = codecRegistry.get("zlib");
if (zlib) codecRegistry.set("numcodecs.zlib", zlib);

// Virtual references into the Annual NLCD COG, both hosted on R2 (see
// scripts/virtualize_nlcd.py).
export const NLCD_REPO =
  "https://pub-a9c45d804ff74bd19e8f9b7a52d9a90b.r2.dev/nlcd/nlcd.icechunk";

export const NLCD_CLASSES: Array<{ code: number; name: string; color: string }> =
  [
    { code: 11, name: "Open Water", color: "#466b9f" },
    { code: 12, name: "Perennial Ice/Snow", color: "#d1def8" },
    { code: 21, name: "Developed, Open Space", color: "#dec5c5" },
    { code: 22, name: "Developed, Low Intensity", color: "#d99282" },
    { code: 23, name: "Developed, Medium Intensity", color: "#eb0000" },
    { code: 24, name: "Developed, High Intensity", color: "#ab0000" },
    { code: 31, name: "Barren Land", color: "#b3ac9f" },
    { code: 41, name: "Deciduous Forest", color: "#68ab5f" },
    { code: 42, name: "Evergreen Forest", color: "#1c5f2c" },
    { code: 43, name: "Mixed Forest", color: "#b5c58f" },
    { code: 52, name: "Shrub/Scrub", color: "#ccb879" },
    { code: 71, name: "Grassland/Herbaceous", color: "#dfdfc2" },
    { code: 81, name: "Pasture/Hay", color: "#dcd939" },
    { code: 82, name: "Cultivated Crops", color: "#ab6c28" },
    { code: 90, name: "Woody Wetlands", color: "#b8d9eb" },
    { code: 95, name: "Emergent Herbaceous Wetlands", color: "#6c9fb8" },
  ];

const glslColor = (hex: string) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  return `vec3(${r.toFixed(4)}, ${g.toFixed(4)}, ${b.toFixed(4)})`;
};

const classColors = NLCD_CLASSES.map(
  ({ code, color }) =>
    `  else if (land_cover == ${code}.0) c = ${glslColor(color)};`,
).join("\n");

const customFrag = `
  if (isnan(land_cover)) {
    discard;
  }
  vec3 c;
  if (false) {}
${classColors}
  else {
    discard;
  }
  fragColor = vec4(c * opacity, opacity);
`;

export async function createNlcdLayer(opacity: number): Promise<ZarrLayer> {
  const store = await IcechunkStore.open(NLCD_REPO);
  return new ZarrLayer({
    id: "nlcd",
    store,
    variable: "land_cover",
    clim: [0, 1],
    colormap: ["#000000", "#ffffff"],
    opacity,
    customFrag,
  });
}
