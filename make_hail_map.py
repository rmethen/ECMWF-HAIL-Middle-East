"""
ECMWF Experimental Hail Potential Index V2
Middle East Map
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from hail_index import hail_potential_v2


# -------------------------------------------------
# Middle East domain
# -------------------------------------------------

WEST = 20
EAST = 65
SOUTH = 10
NORTH = 45


def draw_hail_map(lons, lats, hail_index, run_text="Experimental"):
    """
    Draw Middle East Hail Potential Index map.
    """

    os.makedirs("output", exist_ok=True)

    fig = plt.figure(figsize=(14, 9))

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    ax.set_extent(
        [WEST, EAST, SOUTH, NORTH],
        crs=ccrs.PlateCarree()
    )

    # Map features
    ax.add_feature(
        cfeature.LAND,
        facecolor="whitesmoke"
    )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white"
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.6
    )

    # Hail Potential
    levels = [
        0, 10, 20, 30, 40,
        50, 60, 70, 80, 90, 100
    ]

    plot = ax.contourf(
        lons,
        lats,
        hail_index,
        levels=levels,
        cmap="turbo",
        extend="max",
        transform=ccrs.PlateCarree()
    )

    # Kuwait marker
    ax.plot(
        47.98,
        29.38,
        marker="*",
        markersize=10,
        transform=ccrs.PlateCarree()
    )

    ax.text(
        48.3,
        29.5,
        "Kuwait",
        fontsize=9,
        transform=ccrs.PlateCarree()
    )

    # Grid
    grid = ax.gridlines(
        draw_labels=True,
        linewidth=0.4,
        alpha=0.5
    )

    grid.top_labels = False
    grid.right_labels = False

    # Color bar
    cbar = plt.colorbar(
        plot,
        ax=ax,
        orientation="vertical",
        pad=0.025,
        shrink=0.82
    )

    cbar.set_label(
        "Experimental Hail Potential Index (0–100)"
    )

    plt.title(
        "ECMWF Experimental Hail Potential Index V2 – Middle East\n"
        + run_text,
        fontsize=14,
        weight="bold"
    )

    plt.figtext(
        0.5,
        0.02,
        "Experimental diagnostic – not an official ECMWF product",
        ha="center",
        fontsize=9
    )

    output_file = (
        "output/"
        "ECMWF_HAIL_INDEX_MIDDLE_EAST_LATEST.png"
    )

    plt.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print("Map saved:", output_file)


if __name__ == "__main__":

    # Temporary test grid.
    # ECMWF forecast fields will replace this in the next stage.

    lons = np.linspace(WEST, EAST, 181)
    lats = np.linspace(SOUTH, NORTH, 141)

    lon2d, lat2d = np.meshgrid(lons, lats)

    # Test signal centered near Kuwait
    hail_test = 85.0 * np.exp(
        -(
            ((lon2d - 47.8) / 4.0) ** 2
            +
            ((lat2d - 29.2) / 3.0) ** 2
        )
    )

    draw_hail_map(
        lon2d,
        lat2d,
        hail_test,
        run_text="TEST MAP – ECMWF fields not connected yet"
    )
