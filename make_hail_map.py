"""Generate the real-data ECMWF Experimental Hail Potential map."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from hail_index import hail_potential_v2

DATA_FILE = Path("data/hail_diagnostics.npz")
OUTPUT_FILE = Path("output/ECMWF_HAIL_INDEX_MIDDLE_EAST_LATEST.png")
WEST, EAST, SOUTH, NORTH = 20, 65, 10, 45


def smooth_field(field):
    """Apply two light passes to reduce pixel noise without erasing signals."""
    result = np.asarray(field, dtype=float)
    for _ in range(2):
        padded = np.pad(result, 1, mode="edge")
        result = sum(
            padded[i:i + result.shape[0], j:j + result.shape[1]]
            for i in range(3) for j in range(3)
        ) / 9.0
    return result


def white_low_cmap():
    colors = plt.get_cmap("turbo", 10)(np.arange(10))
    colors[0] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors)


def format_time(value):
    return np.datetime_as_string(np.datetime64(value, "m"), unit="m").replace("T", " ") + " UTC"


def draw_map(lons, lats, field, peak_hour, init_time):
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([WEST, EAST, SOUTH, NORTH], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    display_field = np.ma.masked_less(field, 20.0)
    plot = ax.contourf(
        lons, lats, display_field, levels=np.arange(0, 101, 10), cmap=white_low_cmap(),
        extend="max", transform=ccrs.PlateCarree(),
    )
    ax.plot(47.98, 29.38, marker="*", color="black", markersize=10,
            transform=ccrs.PlateCarree())
    ax.text(48.3, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False
    cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
    cbar.set_label("Experimental Large Hail Potential (0–100)")
    plt.title(
        "ECMWF Experimental Large Hail Potential Index V3 – Middle East\n"
        f"Init: {format_time(init_time)} | Peak: +{peak_hour} h | "
        f"Valid: {format_time(init_time + np.timedelta64(peak_hour, 'h'))}",
        fontsize=14, weight="bold",
    )
    plt.figtext(
        0.5, 0.035,
        "Large-hail environment: HGL + WBZ + Shear | Support: LI • Lapse • Moisture • Omega 700/500",
        ha="center", fontsize=9,
    )
    plt.figtext(
        0.06, 0.012, "Signal: 60–79 significant | 80–100 very strong (not hail diameter)",
        ha="left", fontsize=9,
    )
    plt.figtext(
        0.94, 0.012, f"Max: {np.nanmax(field):.1f}",
        ha="right", fontsize=10, weight="bold",
    )
    plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    plt.close()
    print("Map saved:", OUTPUT_FILE)


def main():
    data = np.load(DATA_FILE)
    index = hail_potential_v2(
        shear06=data["shear_850_300"], wbz_m=data["wbz_m"],
        hgl_depth_m=data["hgl_depth_m"],
        lapse_700_500=data["lapse_700_500"], t500_c=data["t500_c"],
        mixing_ratio=data["moisture_850"], li850=data["li850"],
        omega700=data["omega_700"], omega500=data["omega_500"],
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
    draw_map(data["longitude"], data["latitude"], field, peak_hour, data["init_time"])


if __name__ == "__main__":
    main()
