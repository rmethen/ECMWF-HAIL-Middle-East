"""Generate the real-data ECMWF Experimental Hail Potential map."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from hail_index import hail_potential_v2, inverse_scale, scale
from hail_report import (
    CITIES, city_diagnostics, draw_city_table, draw_diagnostics, forecast_hours,
    format_time, peak_fields, related_output, valid_time, write_city_csv,
)

DATA_FILE = Path("data/hail_diagnostics.npz")
OUTPUT_FILE = Path("output/ECMWF_HAIL_INDEX_MIDDLE_EAST_LATEST.png")
MAP_LABEL = "Original WBZ"
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


def white_low_cmap(*, white_below_20=False):
    colors = plt.get_cmap("turbo", 10)(np.arange(10))
    colors[:2 if white_below_20 else 1] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors)


def extreme_ingredient_count(data):
    """Count strong ingredients; experimental diagnostic, not a hail probability."""
    li = inverse_scale(data["li850"], -6.0, 2.0)
    ascent = (
        0.60 * inverse_scale(data["omega_700"], -1.0, 0.2)
        + 0.40 * inverse_scale(data["omega_500"], -0.8, 0.15)
    )
    return sum(
        condition.astype(np.uint8)
        for condition in (
            scale(data["hgl_depth_m"], 2400.0, 5000.0) >= 0.5,
            scale(data["shear_850_300"], 10.0, 28.0) >= 0.5,
            scale(data["lapse_700_500"], 5.5, 8.0) >= 0.5,
            scale(data["moisture_850"], 5.0, 13.0) >= 0.5,
            np.maximum(li, ascent) >= 0.5,
        )
    )


def draw_map(lons, lats, field, peak_hour, init_time, *,
             output_file=OUTPUT_FILE, map_label=MAP_LABEL, extreme_count=None,
             omega_overlay=None, hours=None, city_rows=None, timing=False):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_axes([0.065, 0.26, 0.80, 0.58], projection=ccrs.PlateCarree())
    ax.set_extent([WEST, EAST, SOUTH, NORTH], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    levels = np.arange(0, 101, 10)
    if timing:
        # Mask cells without a meaningful V2 signal; a zero score has no
        # useful peak time. Values are actual forecast leads, not array indices.
        spacing = float(np.min(np.diff(hours))) if len(hours) > 1 else 1.0
        timing_levels = np.r_[hours - spacing / 2, hours[-1] + spacing / 2]
        plot = ax.pcolormesh(
            lons, lats, np.ma.masked_invalid(field), shading="nearest", cmap="viridis",
            norm=BoundaryNorm(timing_levels, ncolors=256), transform=ccrs.PlateCarree(),
        )
    elif omega_overlay is not None:
        # Color every valid grid cell, including isolated maxima next to NaNs.
        plot = ax.pcolormesh(
            lons, lats, field, shading="nearest",
            cmap=white_low_cmap(white_below_20=True),
            norm=BoundaryNorm(levels, ncolors=10, clip=True),
            transform=ccrs.PlateCarree(),
        )
    else:
        display_field = np.ma.masked_less(field, 20.0)
        plot = ax.contourf(
            lons, lats, display_field, levels=levels, cmap=white_low_cmap(),
            extend="max", transform=ccrs.PlateCarree(),
        )
    if extreme_count is not None:
        ax.contourf(
            lons, lats, extreme_count, levels=[3.5, 5.5],
            colors="none", hatches=["////"], transform=ccrs.PlateCarree(),
        )
    if omega_overlay is not None:
        # ECMWF pressure velocity (Pa/s): more negative means stronger
        # model-resolved ascent, not a thunderstorm updraft speed in m/s.
        omega = smooth_field(omega_overlay)
        valid = omega[np.isfinite(omega)]
        if valid.size:
            levels = [level for level in (-0.6, -0.3)
                      if valid.min() < level < valid.max()]
            if levels:
                contours = ax.contour(
                    lons, lats, omega, levels=levels, colors="#7b1fa2",
                    linestyles=["solid" if level == -0.6 else "dashed"
                                for level in levels], linewidths=1.2,
                    transform=ccrs.PlateCarree(),
                )
                ax.clabel(contours, fmt=lambda value: f"{value:.1f}",
                          inline=True, fontsize=8)
    for name, lat, lon, offset in CITIES:
        ax.plot(lon, lat, marker="*" if name == "Kuwait" else "o", color="black",
                markersize=7 if name == "Kuwait" else 3,
                transform=ccrs.PlateCarree())
        ax.annotate(name, xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    xytext=offset, textcoords="offset points", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False
    cax = fig.add_axes([0.89, 0.31, 0.018, 0.48])
    cbar = fig.colorbar(plot, cax=cax,
                        **({"extend": "max"} if omega_overlay is not None else {}))
    cbar.set_label("Local V2 peak lead time (hours from Init)" if timing
                   else "Experimental Large Hail Potential (0–100)")
    if timing:
        cbar.set_ticks(hours[::max(1, int(np.ceil(len(hours) / 10)))])
    fig.suptitle(
        f"ECMWF Experimental Large Hail Potential V2 – {map_label}\n"
        + ("Local peak timing (raw V2 >= 20)" if timing else "Maximum over forecast period"),
        fontsize=15, weight="bold", y=0.97)
    if hours is None:
        hours = np.array([peak_hour], dtype=float)
    fig.text(0.5, 0.907,
             f"Init: {format_time(init_time)} | Forecast: +{hours[0]:g} to +{hours[-1]:g} h\n"
             f"Period (UTC+3): {format_time(valid_time(init_time, hours[0]), local=True)}"
             f" to {format_time(valid_time(init_time, hours[-1]), local=True)}\n"
             f"Domain-wide raw V2 maximum: +{peak_hour:g} h"
             f" ({format_time(valid_time(init_time, peak_hour), local=True)}); city peaks differ",
             ha="center", va="center", fontsize=10)
    if city_rows is not None:
        draw_city_table(fig, city_rows)
    if extreme_count is None:
        footer = "Large-hail environment: HGL + WBZ + Shear | Support: LI • Lapse • Moisture • Omega 700/500"
    else:
        footer = "Hatching: 4–5 of 5 strong ingredients at local peak (HGL, shear, lapse, moisture, LI/ascent); uncalibrated"
        if omega_overlay is not None:
            plt.figtext(
                0.5, 0.058,
                "Purple: ECMWF 700/500 hPa omega at local V2 peak (dashed -0.3, solid -0.6 Pa/s); model-scale ascent, not storm updraft",
                ha="center", fontsize=9,
            )
    if timing:
        footer = "First sampled maximum per grid cell; blank where raw maximum < 20 or missing. No interpolation between forecast times."
    plt.figtext(0.5, 0.035, footer, ha="center", fontsize=9)
    plt.figtext(0.5, 0.225,
                "Table: raw V2 at nearest grid point; map colors are smoothed. Ties use the first forecast time."
                if not timing else "Timing uses raw V2; UTC+3 applies to Saudi Arabia and Kuwait.",
                ha="center", fontsize=9)
    plt.figtext(
        0.06, 0.012, ("Lead time is measured from model initialization" if timing else
                      "Signal: 60–79 significant | 80–100 very strong (not hail diameter)"),
        ha="left", fontsize=9,
    )
    plt.figtext(
        0.94, 0.012, ("Local peak timing" if timing else f"Smoothed map max V2: {np.nanmax(field):.1f}") +
        (f" | Extreme ingredients: {np.nanmax(extreme_count):.0f}/5" if extreme_count is not None else ""),
        ha="right", fontsize=10, weight="bold",
    )
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close()
    print("Map saved:", output_file)


def main(*, wbz_variant="original", output_file=OUTPUT_FILE,
         map_label=MAP_LABEL, show_extreme=False, show_omega=False):
    data = np.load(DATA_FILE)
    index = hail_potential_v2(
        shear06=data["shear_850_300"], wbz_m=data["wbz_m"],
        hgl_depth_m=data["hgl_depth_m"],
        lapse_700_500=data["lapse_700_500"], t500_c=data["t500_c"],
        mixing_ratio=data["moisture_850"], li850=data["li850"],
        omega700=data["omega_700"], omega500=data["omega_500"],
        wbz_variant=wbz_variant,
    )
    index, raw_field, local_peak = peak_fields(index)
    hours = forecast_hours(data["steps"], index.shape[0])
    field = smooth_field(raw_field)
    counts = extreme_ingredient_count(data)
    if counts.ndim == 2:
        counts = counts[None]
    city_rows = city_diagnostics(data, index, hours, counts, field)
    extreme_count = None
    omega_overlay = None
    if show_extreme:
        extreme_count = np.take_along_axis(counts, local_peak[None], axis=0)[0].astype(float)
        extreme_count[~np.isfinite(raw_field)] = np.nan
    if show_omega:
        omega = 0.60 * data["omega_700"] + 0.40 * data["omega_500"]
        if omega.ndim == 2:
            omega = omega[None]
        omega_overlay = np.take_along_axis(omega, local_peak[None], axis=0)[0]
        omega_overlay[~np.isfinite(raw_field)] = np.nan
    safe = np.where(np.isfinite(index), index, -np.inf)
    peak_idx = np.unravel_index(np.argmax(safe), index.shape)[0]
    peak_hour = hours[peak_idx]
    draw_map(data["longitude"], data["latitude"], field, peak_hour, data["init_time"],
             output_file=output_file, map_label=map_label, extreme_count=extreme_count,
             omega_overlay=omega_overlay, hours=hours, city_rows=city_rows)
    timing_field = np.where(np.isfinite(raw_field) & (raw_field >= 20), hours[local_peak], np.nan)
    draw_map(data["longitude"], data["latitude"], timing_field, peak_hour, data["init_time"],
             output_file=related_output(output_file, "PEAK_TIME"), map_label=map_label,
             hours=hours, city_rows=city_rows, timing=True)
    draw_diagnostics(related_output(output_file, "CITY_DIAGNOSTICS"), city_rows,
                     data["init_time"], hours, map_label)
    write_city_csv(related_output(output_file, "CITY_DIAGNOSTICS", "csv"), city_rows,
                   data["init_time"], hours, wbz_variant)
    data.close()


if __name__ == "__main__":
    main()

