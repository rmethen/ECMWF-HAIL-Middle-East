"""Render the WBZ experiment with a separate extreme-ingredient overlay."""

from pathlib import Path

from make_hail_map import main


if __name__ == "__main__":
    main(
        wbz_variant="1500-3000",
        output_file=Path("output/ECMWF_HAIL_INDEX_WBZ_1500_3000_LATEST.png"),
        map_label="WBZ optimum 1500–3000 m + extreme ingredients",
        show_extreme=True,
        show_omega=True,
    )
