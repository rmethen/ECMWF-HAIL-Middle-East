"""Plot ECMWF ENS rainfall agreement and wettest-member maps."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = Path("data/ecmwf_ens_precip_ensemble.npz")
OUT_PROB = Path("output/ECMWF_ENS_PRECIP_AGREEMENT_0_72H_LATEST.png")
OUT_MAX = Path("output/ECMWF_ENS_PRECIP_WETTEST_MEMBER_0_72H_LATEST.png")
EXTENT = [20, 65, 10, 45]


def base_ax(title):
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    grid.top_labels = grid.right_labels = False
    ax.plot(
        47.98, 29.38, marker="*", color="black", markersize=10,
        transform=ccrs.PlateCarree(),
    )
    ax.text(48.3, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
    plt.title(title, fontsize=14, weight="bold")
    return fig, ax


def main():
    data = np.load(DATA_FILE)
    rain = data["precipitation_mm"]
    lats, lons = data["latitude"], data["longitude"]
    members = data["members"]
    init = np.datetime_as_string(
        data["init_time"].astype("datetime64[h]"), unit="h"
    ).replace("T", " ")
    agreement = np.mean(rain >= 25.0, axis=0) * 100.0

    OUT_PROB.parent.mkdir(exist_ok=True)
    fig, ax = base_ax(
        "ECMWF ENS Agreement: 0–72 h Rain ≥25 mm – Middle East\n"
        f"Init: {init} UTC | 51 members"
    )
    shown = np.ma.masked_less(agreement, 10.0)
    plot = ax.contourf(
        lons, lats, shown, levels=np.arange(10, 101, 10),
        cmap="YlOrRd", extend="max", transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
    cbar.set_label("Members reaching 25 mm (%)")
    plt.figtext(
        0.5, 0.02,
        "Agreement map: percentage of ECMWF ENS members reaching the threshold",
        ha="center",
    )
    plt.savefig(OUT_PROB, dpi=160, bbox_inches="tight")
    plt.close(fig)

    member_totals = np.nanmax(rain.reshape(rain.shape[0], -1), axis=1)
    wettest_idx = int(np.nanargmax(member_totals))
    wettest = rain[wettest_idx]
    wettest_name = str(members[wettest_idx])
    fig, ax = base_ax(
        f"ECMWF ENS Wettest Member: {wettest_name} – 0–72 h Precipitation\n"
        f"Init: {init} UTC"
    )
    shown = np.ma.masked_less(wettest, 1.0)
    levels = [1, 5, 10, 25, 50, 75, 100, 150, 200]
    plot = ax.contourf(
        lons, lats, shown, levels=levels, cmap="turbo",
        extend="max", transform=ccrs.PlateCarree(),
    )
    cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
    cbar.set_label("Accumulated precipitation (mm)")
    plt.figtext(
        0.5, 0.02,
        "Extreme ENS scenario, not the ensemble mean or deterministic forecast",
        ha="center",
    )
    plt.savefig(OUT_MAX, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", OUT_PROB, OUT_MAX)


if __name__ == "__main__":
    main()
