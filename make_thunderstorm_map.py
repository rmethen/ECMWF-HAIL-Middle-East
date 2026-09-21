"""Generate the ECMWF Thunderstorm & Lightning Potential map."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from thunderstorm_index import thunderstorm_potential

DATA_FILE = Path("data/hail_diagnostics.npz")
OUTPUT_FILE = Path("output/ECMWF_THUNDERSTORM_LIGHTNING_POTENTIAL_LATEST.png")


def smooth_field(field):
    """Apply a light 3x3 smoother without hiding regional signals."""
    padded = np.pad(field, 1, mode="edge")
    return sum(
        padded[i:i + field.shape[0], j:j + field.shape[1]]
        for i in range(3) for j in range(3)
    ) / 9.0


def format_time(value):
    return np.datetime_as_string(np.datetime64(value, "m"), unit="m").replace("T", " ") + " UTC"


def main():
    data = np.load(DATA_FILE)
    index = thunderstorm_potential(
        data["li850"], data["omega_700"], data["moisture_850"],
        data["lapse_700_500"], data["shear_850_300"], data["t500_c"],
    )
    field = np.nanmax(index, axis=0) if index.ndim == 3 else index
    field = smooth_field(field)
    step_scores = np.nanmax(index.reshape(index.shape[0], -1), axis=1)
    peak_idx = int(np.nanargmax(step_scores))
    step = data["steps"][peak_idx]
    peak_hour = (
        int(step / np.timedelta64(1, "h"))
        if np.issubdtype(step.dtype, np.timedelta64)
        else int(step)
    )
    init_time = data["init_time"]
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([20, 65, 10, 45], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    plot = ax.contourf(
        data["longitude"], data["latitude"], field,
        levels=np.arange(0, 101, 10), cmap="turbo", extend="max",
        transform=ccrs.PlateCarree(),
    )
    ax.plot(47.98, 29.38, marker="*", color="black", markersize=10,
            transform=ccrs.PlateCarree())
    ax.text(48.3, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False
    cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
    cbar.set_label("Thunderstorm & Lightning Potential (0–100)")
    plt.title(
        "ECMWF Experimental Thunderstorm & Lightning Potential – Middle East\n"
        f"Init: {format_time(init_time)} | Peak: +{peak_hour} h | "
        f"Valid: {format_time(init_time + np.timedelta64(peak_hour, 'h'))}",
        fontsize=14, weight="bold",
    )
    plt.figtext(
        0.5, 0.02,
        "Convective potential: LI • Omega 700 • Moisture • Lapse Rate • Shear – not flash density",
        ha="center", fontsize=9,
    )
    plt.figtext(
        0.94, 0.02, f"Max: {np.nanmax(field):.1f}",
        ha="right", fontsize=10, weight="bold",
    )
    plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    plt.close()
    print("Map saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
