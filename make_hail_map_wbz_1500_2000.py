"""Render the experimental WBZ variant beside the unchanged original map."""
from functools import partial
from pathlib import Path

from hail_index import hail_potential_v2
import make_hail_map as map_module

map_module.hail_potential_v2 = partial(hail_potential_v2, wbz_variant="1500-2000")
map_module.OUTPUT_FILE = Path("output/ECMWF_HAIL_INDEX_WBZ_1500_2000_LATEST.png")
map_module.MAP_LABEL = "WBZ optimum 1500–2000 m"
map_module.main()
