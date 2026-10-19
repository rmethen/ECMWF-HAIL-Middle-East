"""Plot the second, MUCAPE-aware ECMWF lightning-environment map."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from lightning_environment_index import lightning_environment_potential


DATA_FILE = Path("data/hail_diagnostics.npz")
OUTPUT_FILE = Path("output/ECMWF_LIGHTNING_ENVIRONMENT_EXPERIMENTAL_LATEST.png")


def format_time(value):
    return np.datetime_as_string(np.datetime64(value, "m"), unit="m").replace("T", " ") + " UTC"


def main():
    with np.load(DATA_FILE) as data:
        index = lightning_environment_potential(
            data["mucape"], data["li850"], data["omega_700"],
            data["omega_500"], data["moisture_850"],
            data["lapse_700_500"], data["total_totals"],
            data["kuwait_total_totals"],
        )
        if index.ndim != 3:
            raise ValueError("Expected ECMWF fields with (step, latitude, longitude) dimensions")
        field = np.nanmax(index, axis=0)
        peak_idx = int(np.nanargmax(np.nanmax(index.reshape(index.shape[0], -1), axis=1)))
        step = data["steps"][peak_idx]
        peak_hour = (int(step / np.timedelta64(1, "h"))
                     if np.issubdtype(np.asarray(step).dtype, np.timedelta64)
                     else int(step))
        init_time = data["init_time"]
        lon, lat = data["longitude"], data["latitude"]
        z500 = data["z500_m"][peak_idx]

        OUTPUT_FILE.parent.mkdir(exist_ok=True)
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([20, 65, 10, 45], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="white")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.6)
        plot = ax.contourf(lon, lat, np.ma.masked_less(field, 15.0),
                           levels=np.arange(0, 101, 10), cmap="turbo",
                           extend="max", transform=ccrs.PlateCarree())
        zmin = np.floor(np.nanmin(z500) / 60.0) * 60.0
        zmax = np.ceil(np.nanmax(z500) / 60.0) * 60.0
        if zmax > zmin:
            heights = ax.contour(lon, lat, z500,
                                 levels=np.arange(zmin, zmax + 1.0, 60.0),
                                 colors="#666666", linewidths=0.45,
                                 alpha=0.55, transform=ccrs.PlateCarree())
            ax.clabel(heights, inline=True, fontsize=6, fmt="%d")
        ax.plot(47.98, 29.38, marker="*", color="black", markersize=10,
                transform=ccrs.PlateCarree())
        ax.text(48.3, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
        grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        grid.top_labels = False
        grid.right_labels = False
        bar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
        bar.set_label("Experimental lightning environment (0–100)")
        plt.title("ECMWF Experimental Lightning Environment – Middle East\n"
                  f"Init: {format_time(init_time)} | Peak: +{peak_hour} h | "
                  f"Valid: {format_time(init_time + np.timedelta64(peak_hour, 'h'))}",
                  fontsize=14, weight="bold")
        plt.figtext(0.5, 0.02,
                    "MUCAPE • LI850 • TTI • Omega 700/500 • 850-hPa moisture • lapse rate | Gray: 500-hPa height (m)",
                    ha="center", fontsize=9)
        plt.figtext(0.94, 0.02, f"Max: {np.nanmax(field):.1f}",
                    ha="right", fontsize=10, weight="bold")
        plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
        plt.close(fig)
    print("Map saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
