"""Generate the real-data ECMWF Experimental Hail Potential map."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from hail_index import hail_potential_v2, inverse_scale, scale

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


def format_time(value):
    return np.datetime_as_string(np.datetime64(value, "m"), unit="m").replace("T", " ") + " UTC"


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
             omega_overlay=None):
    output_file.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([WEST, EAST, SOUTH, NORTH], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    levels = np.arange(0, 101, 10)
    if omega_overlay is not None:
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
    ax.plot(47.98, 29.38, marker="*", color="black", markersize=10,
            transform=ccrs.PlateCarree())
    ax.text(48.3, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False
    cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82,
                        **({"extend": "max"} if omega_overlay is not None else {}))
    cbar.set_label("Experimental Large Hail Potential (0–100)")
    plt.title(
        f"ECMWF Experimental Large Hail Potential V2 – {map_label}\n"
        f"Init: {format_time(init_time)} | Peak: +{peak_hour} h | "
        f"Valid: {format_time(init_time + np.timedelta64(peak_hour, 'h'))}",
        fontsize=14, weight="bold",
    )
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
    plt.figtext(0.5, 0.035, footer, ha="center", fontsize=9)
    plt.figtext(
        0.06, 0.012, "Signal: 60–79 significant | 80–100 very strong (not hail diameter)",
        ha="left", fontsize=9,
    )
    plt.figtext(
        0.94, 0.012, f"Max V2: {np.nanmax(field):.1f}" +
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
    field = np.nanmax(index, axis=0) if index.ndim == 3 else index
    field = smooth_field(field)
    extreme_count = None
    omega_overlay = None
    if show_extreme or show_omega:
        if index.ndim == 3:
            local_peak = np.nanargmax(np.where(np.isfinite(index), index, -np.inf), axis=0)
    if show_extreme:
        counts = extreme_ingredient_count(data)
        if index.ndim == 3:
            extreme_count = np.take_along_axis(counts, local_peak[None], axis=0)[0]
        else:
            extreme_count = counts
    if show_omega:
        omega = 0.60 * data["omega_700"] + 0.40 * data["omega_500"]
        omega_overlay = (
            np.take_along_axis(omega, local_peak[None], axis=0)[0]
            if index.ndim == 3 else omega
        )
    step_scores = np.nanmax(index.reshape(index.shape[0], -1), axis=1)
    peak_idx = int(np.nanargmax(step_scores))
    step = data["steps"][peak_idx]
    peak_hour = (
        int(step / np.timedelta64(1, "h"))
        if np.issubdtype(step.dtype, np.timedelta64)
        else int(step)
    )
    draw_map(data["longitude"], data["latitude"], field, peak_hour, data["init_time"],
             output_file=output_file, map_label=map_label, extreme_count=extreme_count,
             omega_overlay=omega_overlay)


if __name__ == "__main__":
    main()
