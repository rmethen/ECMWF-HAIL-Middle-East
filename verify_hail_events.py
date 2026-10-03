"""Compare archived research fields with independently documented hail events.

Input CSV columns: event_id,valid_utc,latitude,longitude,hail_mm.
The optional hail_mm=0 rows must be verified storm controls, not arbitrary
clear-weather grid cells. No skill is asserted from one event.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FIELDS = ("v2", "wmaxshear_proxy", "survival_growth")


def hours(steps):
    if np.issubdtype(steps.dtype, np.timedelta64):
        return steps / np.timedelta64(1, "h")
    return steps.astype(float)


def distance_km(lat, lon, lat0, lon0):
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lat0)) * np.sin(dlon / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def evaluate(archive, event, radius_km=75, time_hours=3):
    init = np.datetime64(archive["init_time"], "s")
    valid = init + (hours(archive["steps"]) * 3600).astype("timedelta64[s]")
    target = np.datetime64(datetime.fromisoformat(event["valid_utc"].replace("Z", "+00:00")).astimezone(timezone.utc).replace(tzinfo=None), "s")
    time_ids = np.flatnonzero(np.abs((valid - target) / np.timedelta64(1, "h")) <= time_hours)
    if not len(time_ids):
        raise ValueError(f"{event['event_id']}: no forecast step within ±{time_hours} hours")
    lat, lon = np.meshgrid(archive["latitude"], archive["longitude"], indexing="ij")
    dist = distance_km(lat, lon, float(event["latitude"]), float(event["longitude"]))
    local = dist <= radius_km
    if not np.any(local):
        raise ValueError(f"{event['event_id']}: event outside the forecast grid")
    nearest = np.unravel_index(np.nanargmin(dist), dist.shape)
    result = {"event_id": event["event_id"], "valid_utc": event["valid_utc"],
              "latitude": event["latitude"], "longitude": event["longitude"],
              "hail_mm": event["hail_mm"], "init_utc": str(init),
              "radius_km": radius_km, "time_window_h": time_hours}
    for name in FIELDS:
        field = archive[name][time_ids]
        nearest_value = float(np.nanmax(field[:, nearest[0], nearest[1]]))
        local_values = np.where(local[None, :, :], field, np.nan)
        local_peak = float(np.nanmax(local_values))
        # The local maximum's displacement helps separate a timing/location
        # miss from a weak environmental signal. It is not a hail-swath error.
        peak_at = np.unravel_index(np.nanargmax(local_values), local_values.shape)
        result[f"{name}_nearest"] = round(nearest_value, 3)
        result[f"{name}_nearby_max"] = round(local_peak, 3)
        result[f"{name}_peak_distance_km"] = round(float(dist[peak_at[1:]]), 1)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("observations", type=Path)
    p.add_argument("archives", type=Path, nargs="+")
    p.add_argument("--output", type=Path, default=Path("data/hail_event_comparison.csv"))
    p.add_argument("--radius-km", type=float, default=75)
    p.add_argument("--time-hours", type=float, default=3)
    args = p.parse_args()
    with args.observations.open(newline="") as stream:
        events = list(csv.DictReader(stream))
    if not events:
        raise ValueError("Observation CSV is empty")
    rows = []
    for path in args.archives:
        with np.load(path) as archive:
            for event in events:
                try:
                    rows.append(evaluate(archive, event, args.radius_km, args.time_hours))
                except ValueError as exc:
                    print(f"Skipping {path}: {exc}")
    if not rows:
        raise ValueError("No observation matched a forecast archive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} event/forecast comparisons to {args.output}")


if __name__ == "__main__":
    main()
