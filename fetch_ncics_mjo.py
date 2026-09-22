"""Mirror the latest NCICS two-day OLR/CFS tropical-wave forecast map."""

from pathlib import Path
import time

import requests


URL = "https://ncics.org/pub/mjo/v2/map/olr.cfs.all.global.2.png"
OUT_FILE = Path("output/NCICS_OLR_CFS_MJO_WAVES_LATEST.png")


def main() -> None:
    for attempt in range(5):
        try:
            response = requests.get(URL, timeout=180)
            response.raise_for_status()
            if response.content.startswith(b"\x89PNG\r\n\x1a\n") and len(response.content) > 50_000:
                OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
                OUT_FILE.write_bytes(response.content)
                print(f"Saved {OUT_FILE} ({len(response.content):,} bytes) from NCICS")
                return
        except requests.RequestException as exc:
            print(f"Attempt {attempt + 1} failed: {exc}")
        time.sleep(2 ** attempt)
    raise RuntimeError("The latest NCICS OLR/CFS tropical-wave map was unavailable")


if __name__ == "__main__":
    main()
