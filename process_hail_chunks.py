"""Run the existing ECMWF diagnostics in small forecast-time batches."""
import gc
from pathlib import Path
import numpy as np
import process_hail_grib as processor
from ecmwf_hail_data import STEPS

CHUNK_SIZE = 5
FINAL = Path("data/hail_diagnostics.npz")
BASE_OPEN_FIELD = processor.open_field
BASE_OPEN_SURFACE = processor.open_surface_field


def main():
    pieces = []
    for start in range(0, len(STEPS), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(STEPS))
        processor.open_field = lambda name, a=start, b=stop: BASE_OPEN_FIELD(name).isel(step=slice(a, b))
        processor.open_surface_field = lambda name, a=start, b=stop: BASE_OPEN_SURFACE(name).isel(step=slice(a, b))
        processor.OUTPUT_FILE = Path(f"data/hail_diagnostics_chunk_{start:02d}.npz")
        print(f"Processing ECMWF forecast steps {STEPS[start]}–{STEPS[stop-1]} h", flush=True)
        processor.main()
        pieces.append(processor.OUTPUT_FILE)
        gc.collect()

    with np.load(pieces[0]) as first:
        names = first.files
    combined = {}
    for name in names:
        if name in ("latitude", "longitude", "init_time"):
            with np.load(pieces[0]) as first:
                combined[name] = first[name]
        else:
            arrays = []
            for path in pieces:
                with np.load(path) as item:
                    arrays.append(item[name])
            combined[name] = np.concatenate(arrays, axis=0)
    np.savez_compressed(FINAL, **combined)
    for path in pieces:
        path.unlink()
    print(f"Combined {len(STEPS)} forecast steps into {FINAL}", flush=True)


if __name__ == "__main__":
    main()
