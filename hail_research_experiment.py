"""Save uncalibrated hail research diagnostics without changing the V2 map.

WMAXSHEAR here uses ECMWF MUCAPE and the existing 850–300 hPa wind
difference. It is not the effective-shear / mixed-layer index in the
Spanish study and must be validated before operational use.
"""

import argparse
from pathlib import Path

import numpy as np

from hail_index import hail_potential_v2, inverse_scale, scale


def candidate_fields(data):
    base = hail_potential_v2(
        shear06=data["shear_850_300"], wbz_m=data["wbz_m"],
        hgl_depth_m=data["hgl_depth_m"],
        lapse_700_500=data["lapse_700_500"], t500_c=data["t500_c"],
        mixing_ratio=data["moisture_850"], li850=data["li850"],
        omega700=data["omega_700"], omega500=data["omega_500"],
    )
    # sqrt(2 CAPE) is a theoretical maximum updraft speed (m/s), not
    # a model-resolved vertical velocity. Preserve physical units.
    wmaxshear_proxy = np.sqrt(2.0 * np.maximum(data["mucape"], 0.0)) * data["shear_850_300"]
    wbz = np.minimum(scale(data["wbz_m"], 1400, 2200),
                     inverse_scale(data["wbz_m"], 3300, 4400))
    growth = scale(data["hgl_depth_m"], 2400, 5000)
    survival_growth = wbz * growth
    return base, wmaxshear_proxy, survival_growth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/hail_diagnostics.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/hail_research_experiment.npz"))
    args = parser.parse_args()
    with np.load(args.input) as data:
        base, proxy, interaction = candidate_fields(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, latitude=data["latitude"], longitude=data["longitude"],
                            steps=data["steps"], init_time=data["init_time"],
                            v2=base.astype("float32"),
                            wmaxshear_proxy=proxy.astype("float32"),
                            survival_growth=interaction.astype("float32"))
    print(f"Experimental fields saved: {args.output}; V2 weights unchanged")


if __name__ == "__main__":
    main()
