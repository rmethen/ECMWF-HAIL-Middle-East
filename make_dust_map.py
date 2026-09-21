"""Generate ECMWF dust-storm and convective wall-dust potential map."""

from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from dust_index import dust_storm_potential
from thunderstorm_index import thunderstorm_potential

DATA_FILE = Path("data/hail_diagnostics.npz")
OUTPUT_FILE = Path("output/ECMWF_DUST_WALL_STORM_POTENTIAL_LATEST.png")


def smooth_field(field):
    result = np.asarray(field, dtype=float)
    for _ in range(2):
        padded = np.pad(result, 1, mode="edge")
        result = sum(
            padded[i:i + result.shape[0], j:j + result.shape[1]]
            for i in range(3) for j in range(3)
        ) / 9.0
    return result


def white_low_cmap():
    colors = plt.get_cmap("YlOrBr", 10)(np.arange(10))
    colors[0] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors)


def format_time(value):
    return np.datetime_as_string(np.datetime64(value, "m"), unit="m").replace("T", " ") + " UTC"


def main():
    data = np.load(DATA_FILE)
    thunder = thunderstorm_potential(
        data["li850"], data["omega_700"], data["omega_500"], data["moisture_850"],
        data["lapse_700_500"], data["shear_850_300"], data["t500_c"],
        data["total_totals"], data["kuwait_total_totals"],
    )
    dust, wall = dust_storm_potential(
        data["wind10"], data["gust10"], data["msl_hpa"],
        data["dewpoint_depression"], thunder, data["omega_700"],
    )
    step_scores = np.nanmax(dust.reshape(dust.shape[0], -1), axis=1)
    peak_idx = int(np.nanargmax(step_scores))
    field = smooth_field(np.nanmax(dust, axis=0))
    wall_field = smooth_field(np.nanmax(wall, axis=0))
    step = data["steps"][peak_idx]
    peak_hour = int(step / np.timedelta64(1, "h")) if np.issubdtype(step.dtype, np.timedelta64) else int(step)
    init_time = data["init_time"]

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([20, 65, 10, 45], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    plot = ax.contourf(
        data["longitude"], data["latitude"], np.ma.masked_less(field, 15.0),
        levels=np.arange(0, 101, 10), cmap=white_low_cmap(), extend="max",
        transform=ccrs.PlateCarree(),
    )
    wall_contours = ax.contour(
        data["longitude"], data["latitude"], wall_field,
        levels=[45, 60, 75], colors=["#8b4513", "#b22222", "#4b0000"],
        linewidths=[1.0, 1.5, 2.0], transform=ccrs.PlateCarree(),
    )
    ax.clabel(wall_contours, inline=True, fontsize=7, fmt={45:"Wall 45",60:"Wall 60",75:"Wall 75"})
    ax.plot(47.98, 29.38, marker="*", color="black", markersize=10, transform=ccrs.PlateCarree())
    ax.text(48.3, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False
    cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
    cbar.set_label("Dust Storm Potential (0–100)")
    plt.title(
        "ECMWF Experimental Dust & Wall Storm Potential – Middle East\n"
        f"Init: {format_time(init_time)} | Peak: +{peak_hour} h | "
        f"Valid: {format_time(init_time + np.timedelta64(peak_hour, 'h'))}",
        fontsize=14, weight="bold",
    )
    plt.figtext(
        0.5, 0.02,
        "Gusts • 10-m wind • MSLP (<1000 hPa support) • Surface dryness • Convective outflow | Contours: wall-dust potential",
        ha="center", fontsize=9,
    )
    plt.figtext(0.94, 0.02, f"Max: {np.nanmax(field):.1f}", ha="right", fontsize=10, weight="bold")
    plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    plt.close()
    print("Map saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
