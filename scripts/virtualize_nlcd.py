"""Virtualize the Annual NLCD land cover COG into an Icechunk repo.

Each TIFF IFD (full resolution plus six 2x overviews) becomes a pyramid
level `<level>/land_cover` whose chunks are byte-range references into the
COG's 512x512 tiles. No pixel data is copied.

The root carries the proj/spatial/multiscales conventions zarr-layer reads,
so the repo renders with no layer configuration beyond the variable name.

Chunk references are stamped with the time the repo is written, and readers
reject chunks whose object was modified after that. Run this after the COG
is uploaded to the URL it points at.

Usage (after uploading the COG to the bucket):
    .venv/bin/python scripts/virtualize_nlcd.py \
        --tif ~/Downloads/Annual_NLCD_LndCov_2025_CU_C1V2.tif \
        --url https://pub-a9c45d804ff74bd19e8f9b7a52d9a90b.r2.dev/nlcd/Annual_NLCD_LndCov_2025_CU_C1V2.tif \
        --repo /tmp/nlcd.icechunk
    rclone copy /tmp/nlcd.icechunk r2:live-classify-data/nlcd/nlcd.icechunk
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import icechunk
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtual_tiff import VirtualTIFF
from virtualizarr import open_virtual_dataset

VARIABLE = "land_cover"


def level_transform(base, full_shape, level_shape):
    """Affine for an overview: same extent, coarser pixels (GDAL order)."""
    x_res, _, x0, _, y_res, y0 = base
    return [
        x_res * full_shape[1] / level_shape[1],
        0.0,
        x0,
        0.0,
        y_res * full_shape[0] / level_shape[0],
        y0,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", type=Path, required=True)
    parser.add_argument("--url", required=True, help="URL the chunk refs point at")
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    tif = args.tif.resolve()
    file_prefix = f"file://{tif.parent}"
    registry = ObjectStoreRegistry({file_prefix: LocalStore(str(tif.parent))})
    source_url = f"file://{tif}"

    levels = []
    level = 0
    while True:
        try:
            vds = open_virtual_dataset(
                source_url, registry=registry, parser=VirtualTIFF(ifd=level)
            )
        except (IndexError, ValueError):
            break
        (name,) = vds.data_vars
        vds = vds.rename({name: VARIABLE})
        vds[VARIABLE].attrs = {}
        vds = vds.vz.rename_paths(args.url)
        levels.append(vds)
        level += 1
    print(f"{len(levels)} levels:", [tuple(v[VARIABLE].shape) for v in levels])

    url_prefix = args.url.rsplit("/", 1)[0] + "/"
    config = icechunk.RepositoryConfig.default()
    config.set_virtual_chunk_container(
        icechunk.VirtualChunkContainer(url_prefix, icechunk.http_store())
    )
    if args.repo.exists():
        shutil.rmtree(args.repo)
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(str(args.repo)), config=config
    )
    session = repo.writable_session("main")

    for i, vds in enumerate(levels):
        vds.vz.to_icechunk(session.store, group=str(i))

    full_shape = levels[0][VARIABLE].shape
    x0, y0 = -2415585.0, 3314805.0
    base = [30.0, 0.0, x0, 0.0, -30.0, y0]
    wkt2 = subprocess.run(
        ["gdalsrsinfo", "-o", "wkt2", str(tif)], capture_output=True, text=True
    ).stdout.strip()

    root = zarr.open_group(session.store, mode="a")
    root.attrs.update(
        {
            "proj:wkt2": wkt2,
            "spatial:dimensions": ["y", "x"],
            "spatial:transform": base,
            "spatial:shape": list(full_shape),
            "spatial:bbox": [
                x0,
                y0 - 30.0 * full_shape[0],
                x0 + 30.0 * full_shape[1],
                y0,
            ],
            "spatial:registration": "pixel",
            "multiscales": {
                "layout": [
                    {
                        "asset": str(i),
                        "spatial:shape": list(v[VARIABLE].shape),
                        "spatial:transform": level_transform(
                            base, full_shape, v[VARIABLE].shape
                        ),
                    }
                    for i, v in enumerate(levels)
                ]
            },
        }
    )
    snapshot = session.commit("Virtualize Annual NLCD land cover 2025")
    print("committed", snapshot, "to", args.repo)


if __name__ == "__main__":
    main()
