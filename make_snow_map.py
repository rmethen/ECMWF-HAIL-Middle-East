"""ECMWF snow-environment map with thickness and upper-air temperature contours."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ecmwf_hail_data import SNOWFALL_FILE
from process_hail_grib import open_field, open_surface_field, get_level

OUT = Path("output/ECMWF_SNOW_ENVIRONMENT_MIDDLE_EAST_LATEST.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-snowfall", action="store_true")
    args = parser.parse_args()
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
    snowfall = None
    if SNOWFALL_FILE.exists():
        try:
            sf = open_surface_field("sf", SNOWFALL_FILE).transpose(*tp.dims)
            for coord in ("time", "step", "latitude", "longitude"):
                if not sf[coord].equals(tp[coord]):
                    raise ValueError(f"Snowfall {coord} does not match precipitation")
            snowfall = np.maximum(np.diff(sf.values * 1000, axis=0,
                                          prepend=sf.values[:1] * 1000), 0)
        except Exception as exc:
            print(f"Snowfall layer unavailable: {exc}")

    if args.require_snowfall and (snowfall is None or not np.isfinite(snowfall).any()):
        raise RuntimeError("No usable ECMWF snowfall data for this cycle")

    regional = ((gh.latitude.values[:, None] >= 10) &
                (gh.latitude.values[:, None] <= 45) &
                (gh.longitude.values[None, :] >= 20) &
                (gh.longitude.values[None, :] <= 65))
    scores = np.count_nonzero(supportive & regional, axis=(1, 2))
    ranking = scores
    if snowfall is not None:
        snow_scores = np.count_nonzero((snowfall >= 0.1) & regional, axis=(1, 2))
        if snow_scores.any():
            ranking = snow_scores
    peak = 1 + int(np.argmax(ranking[1:])) if len(ranking) > 1 else 0
    interval = int((gh.step.values[peak] - gh.step.values[max(0, peak - 1)]) /
                   np.timedelta64(1, "h"))
    lon = gh.longitude.values
    lat = gh.latitude.values
    step = int(gh.step.values[peak] / np.timedelta64(1, "h"))
    init = np.datetime_as_string(gh.time.values, unit="h").replace("T", " ")
    valid = np.datetime_as_string(gh.valid_time.values[peak], unit="h").replace("T", " ")

    fields = [
        (thickness[peak], [528, 534, 540, 546, 552, 558], "#85602a", "solid"),
        (t850[peak], [-12, -8, -5, 0, 4], "#1465bc", "dashed"),
        (t500[peak], [-40, -35, -30, -25, -20, -15], "#913f91", "dotted"),
    ]
    filtered = np.where(supportive[peak], precip6[peak], np.nan)
    layers = [
        ("Raw precipitation", precip6[peak], 0.5),
        ("Precipitation passing snow screen", filtered, 0.5),
        ("ECMWF snowfall (water equivalent)", None if snowfall is None else snowfall[peak], 0.1),
    ]
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(1, 3, figsize=(19, 7),
                             subplot_kw={"projection": projection})
    for index, (ax, (label, values, minimum)) in enumerate(zip(axes, layers)):
        ax.set_extent([20, 65, 10, 45], crs=projection)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.55)
        status = ""
        if values is None or not np.isfinite(values[regional]).any():
            status = "\nData unavailable"
            if index == 1 and np.isfinite(precip6[peak][regional]).any():
                status = "\nNo grid cells passed the screen"
        elif not (values[regional] >= minimum).any():
            status = f"\nBelow {minimum:g} mm display threshold"
        else:
            levels = [minimum, 1, 2, 5, 10, 50]
            cmap = ListedColormap(["#e8f4ff", "#a9d7f9", "#65b5ec", "#277bc5", "#17468c"])
            norm = BoundaryNorm(levels, cmap.N)
            mesh = ax.pcolormesh(lon, lat, np.where(values >= minimum, values, np.nan),
                                 cmap=cmap, norm=norm, transform=projection)
            fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05,
                         ticks=levels[:-1], extend="max", label="mm")
        for field, contour_levels, color, style in fields:
            present = [v for v in contour_levels if np.nanmin(field) < v < np.nanmax(field)]
            if present:
                cs = ax.contour(lon, lat, field, levels=present, colors=color,
                                linestyles=style, linewidths=0.75, transform=projection)
                ax.clabel(cs, fmt="%g", fontsize=6, inline=True)
        ax.plot(37, 40.5, marker="x", color="black", markersize=6, transform=projection)
        ax.set_title(label + status, fontsize=11)

    fig.suptitle("ECMWF Snowfall Diagnostics - Middle East\n"
                 f"Init: {init} UTC | Forecast: +{step} h | Valid: {valid} UTC | Interval: {interval} h",
                 fontsize=15, fontweight="bold")
    legend = [Line2D([0], [0], color="#85602a", label="1000-500 hPa thickness (dam)"),
              Line2D([0], [0], color="#1465bc", ls="--", label="850 hPa temperature (°C)"),
              Line2D([0], [0], color="#913f91", ls=":", label="500 hPa temperature (°C)"),
              Line2D([0], [0], marker="x", color="black", ls="", label="Turkey reference point")]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}; selected +{step} h, supportive regional grid cells: {scores[peak]}")


if __name__ == "__main__":
    main()
