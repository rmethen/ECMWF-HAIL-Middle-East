"""ECMWF snow-environment map with thickness and upper-air temperature contours."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from process_hail_grib import open_field, open_surface_field, get_level

OUT = Path("output/ECMWF_SNOW_ENVIRONMENT_MIDDLE_EAST_LATEST.png")


def main():
    gh = open_field("gh")["gh"]
    temp = open_field("t")["t"]
    t2m = open_surface_field("2t")
    tp = open_surface_field("tp")
    thickness = (get_level(gh, 500).values - get_level(gh, 1000).values) / 10.0
    t850 = get_level(temp, 850).values - 273.15
    t500 = get_level(temp, 500).values - 273.15
    surface = t2m.values - 273.15
    total_mm = tp.values * 1000.0
    precip6 = np.maximum(np.diff(total_mm, axis=0, prepend=total_mm[:1]), 0)
    # Screening only: the actual snow level also depends on terrain and the full thermal profile.
    supportive = ((precip6 >= 0.5) & (surface <= 2) &
                  (t850 <= -5) & (thickness <= 540) & (t500 <= -25))
    scores = np.count_nonzero(supportive.reshape(supportive.shape[0], -1), axis=1)
    peak = 1 + int(np.argmax(scores[1:])) if len(scores) > 1 else 0
    snow = np.where(supportive[peak], precip6[peak], np.nan)
    lon = gh.longitude.values
    lat = gh.latitude.values
    step = int(gh.step.values[peak] / np.timedelta64(1, "h"))
    init = np.datetime_as_string(gh.time.values, unit="h").replace("T", " ")
    valid = np.datetime_as_string(gh.valid_time.values[peak], unit="h").replace("T", " ")

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    ax = fig.add_axes([0.07, 0.20, 0.80, 0.68], projection=ccrs.PlateCarree())
    ax.set_extent([20, 65, 10, 45], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.55)
    grid = ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, alpha=0.3)
    grid.top_labels = False
    grid.right_labels = False
    colors = ["#e8f4ff", "#a9d7f9", "#65b5ec", "#277bc5", "#17468c"]
    levels = [0.5, 1, 2, 5, 10, 50]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    if np.isfinite(snow).any():
        ax.contourf(lon, lat, snow, levels=levels, cmap=cmap, norm=norm,
                    extend="max", transform=ccrs.PlateCarree())
    fields = [
        (thickness[peak], [528, 534, 540, 546, 552, 558], "#85602a", "solid"),
        (t850[peak], [-12, -8, -5, 0, 4], "#1465bc", "dashed"),
        (t500[peak], [-40, -35, -30, -25, -20, -15], "#913f91", "dotted"),
    ]
    for field, contour_levels, color, style in fields:
        present = [v for v in contour_levels if np.nanmin(field) < v < np.nanmax(field)]
        if present:
            cs = ax.contour(lon, lat, field, levels=present, colors=color,
                            linestyles=style, linewidths=1.05, transform=ccrs.PlateCarree())
            ax.clabel(cs, fmt="%g", fontsize=7, inline=True)
    ax.plot(47.98, 29.38, marker="*", color="black", markersize=9, transform=ccrs.PlateCarree())
    ax.text(48.2, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
    fig.suptitle("ECMWF Experimental Snow Environment - Middle East\n"
                 f"Init: {init} UTC | Forecast: +{step} h | Valid: {valid} UTC",
                 fontsize=16, fontweight="bold", y=0.98)
    cax = fig.add_axes([0.15, 0.12, 0.64, 0.018])
    bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", ticks=levels[:-1], extend="max")
    bar.set_label("Forecast-interval precipitation (6/12 h) in snow-supportive environment (mm)")
    legend = [Line2D([0], [0], color="#85602a", label="1000-500 hPa thickness (dam)"),
              Line2D([0], [0], color="#1465bc", ls="--", label="850 hPa temperature (°C)"),
              Line2D([0], [0], color="#913f91", ls=":", label="500 hPa temperature (°C)")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.47, 0.045), ncol=3, frameon=False)
    fig.text(0.5, 0.017, "Diagnostic only; terrain, surface temperature and full vertical profile govern actual snowfall.",
             ha="center", fontsize=9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}; peak +{step} h, supportive grid cells: {scores[peak]}")


if __name__ == "__main__":
    main()
