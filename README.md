# Live Classify

Teach a map to find something by clicking examples. Live Classify trains a
classifier on Google's [AlphaEarth Foundations satellite embeddings](https://source.coop/tge-labs/aef-mosaic)
entirely in your browser, and redraws the whole view on every click. There is
no server: data streams straight from cloud storage, training takes a few
milliseconds of JavaScript, and every pixel is scored on the GPU.

**Live site:** https://shane98c.github.io/live-classify/

## Using it

1. Zoom in until you can see individual fields and streets, and type what you
   are looking for (golf courses, solar farms, clearcuts…).
2. Click a few examples of it. They show as green markers.
3. Shift-click things that are not it, especially wrong highlights (red).
4. Raise the match threshold for a stricter map.

Work is autosaved in the browser. **Save** keeps named trainings, and
**Export**/**Import** move them as JSON files.

Two other views: **Find similar spots** highlights pixels whose embedding is
close to spots you click, and **Embeddings as color** shows three of the 64
embedding dimensions as RGB.

## How it works

- **Embeddings.** Each 10 m pixel has a 64-number AlphaEarth embedding
  summarizing a year of optical, radar and other observations (2025 here). The
  app reads them from a public Zarr v3 mosaic on Source Cooperative with
  [zarr-layer](https://github.com/carbonplan/zarr-layer), a MapLibre custom
  layer.
- **Training.** Each click reads the embeddings of a 3x3 pixel patch. A
  logistic regression (64 weights and a bias, L2-regularized, class-balanced)
  is retrained in JavaScript after every click.
- **Background from NLCD.** Your clicks alone never show the model what
  ordinary ground looks like. So the app also samples 200 random spots in view
  that [Annual NLCD](https://www.mrlc.gov/data/project/annual-nlcd) assigns to
  a chosen land cover class (by default, the class under your first example)
  and adds their embeddings as down-weighted "not it" examples. NLCD values are
  only used to pick these spots; they are never model inputs. Tick **Show** to
  see where the spots landed.
- **Rendering.** The trained weights are passed to a fragment shader as
  uniforms, which computes `sigmoid(w · embedding + b)` for every pixel on
  screen, so retraining never refetches data.

The embeddings have no zoomed-out pyramid, so they only load when zoomed in.

## Data

| Data | Source | Notes |
| --- | --- | --- |
| AlphaEarth embeddings, 2025 | [tge-labs/aef-mosaic](https://source.coop/tge-labs/aef-mosaic) on Source Cooperative | CC-BY 4.0. Produced by Google and Google DeepMind. |
| NLCD 2025 land cover | USGS Annual NLCD Collection 1 | Public domain. The COG is hosted on Cloudflare R2 and read through a virtual [Icechunk](https://icechunk.io) repo with [icechunk-js](https://github.com/EarthyScience/icechunk-js). |
| Imagery | [USGS The National Map](https://basemap.nationalmap.gov/) | Public domain. |

`scripts/virtualize_nlcd.py` builds the NLCD Icechunk repo: each TIFF IFD
(full resolution plus six overviews) becomes a pyramid level whose chunks are
byte-range references into the COG, with the `proj:`/`spatial:`/`multiscales`
attributes zarr-layer reads. `scripts/r2-cors.json` is the bucket's CORS rule.

## Development

```sh
npm install
npm run dev
```

`vendor/` holds a build of zarr-layer with changes not yet released: custom
shaders read all bands from one texture array (so 64 bands fit), small integer
dtypes upload as integer textures, band arrays are freed after upload, and the
initial fetch respects `minzoom`. It will be replaced by the npm release once
those land upstream.

To rebuild the NLCD repo (Python 3.12+, [uv](https://docs.astral.sh/uv/)):

```sh
uv venv && uv pip install virtualizarr virtual-tiff icechunk xarray zarr obstore
.venv/bin/python scripts/virtualize_nlcd.py --help
```

Pushes to `main` deploy to GitHub Pages via `.github/workflows/deploy.yml`.
