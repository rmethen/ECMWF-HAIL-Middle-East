"""Time and nearest-grid city diagnostics for both existing V2 variants."""

import csv
from pathlib import Path

import numpy as np

CITIES = (
    ("Arar", 30.98, 41.04, (7, -12)),
    ("Turaif", 31.68, 38.65, (-48, 10)),
    ("Qurayyat", 31.33, 37.34, (-60, -16)),
    ("Sakaka", 29.97, 40.21, (6, -14)),
    ("Kuwait", 29.38, 47.98, (7, 7)),
)
DIAGNOSTICS = (
    ("hgl_depth_m", "HGL depth (m)", ".0f"),
    ("wbz_m", "Approx. WBZ (m MSL)", ".0f"),
    ("shear_850_300", "850-300 hPa wind difference (m/s)", ".1f"),
    ("lapse_700_500", "700-500 hPa lapse rate (C/km)", ".2f"),
    ("t500_c", "500 hPa temperature (C)", ".1f"),
    ("moisture_850", "850 hPa specific humidity (g/kg)", ".2f"),
    ("li850", "Approx. LI850 (C)", ".2f"),
    ("omega_700", "700 hPa omega (Pa/s)", ".2f"),
    ("omega_500", "500 hPa omega (Pa/s)", ".2f"),
)


def forecast_hours(steps, count):
    steps = np.asarray(steps).reshape(-1)
    hours = (steps / np.timedelta64(1, "h")
             if np.issubdtype(steps.dtype, np.timedelta64)
             else steps.astype(float))
    if (hours.size != count or not np.all(np.isfinite(hours))
            or np.any(np.diff(hours) <= 0)):
        raise ValueError("Forecast steps must match the time axis and increase strictly")
    return hours.astype(float)


def format_time(value, *, local=False):
    value = np.asarray(value).reshape(()).astype("datetime64[m]")[()]
    if np.isnat(value):
        return "N/A"
    if local:
        value += np.timedelta64(3, "h")
    return (np.datetime_as_string(value, unit="m").replace("T", " ")
            + (" UTC+3" if local else " UTC"))


def valid_time(init_time, hour):
    return (np.asarray(init_time).reshape(()).astype("datetime64[m]")[()]
            + np.timedelta64(int(round(hour * 60)), "m"))


def peak_fields(index):
    """First finite maximum at each cell; no-data cells keep a NaN peak."""
    index = np.asarray(index, dtype=float)
    if index.ndim == 2:
        index = index[None]
    if index.ndim != 3 or not index.shape[0]:
        raise ValueError("Expected index shaped (time, latitude, longitude)")
    finite = np.isfinite(index)
    if not finite.any():
        raise ValueError("No finite V2 values in this forecast")
    safe = np.where(finite, index, -np.inf)
    local_peak = np.argmax(safe, axis=0)
    field = np.take_along_axis(safe, local_peak[None], axis=0)[0]
    return index, np.where(finite.any(axis=0), field, np.nan), local_peak


def city_diagnostics(data, index, hours, counts, display_field):
    """Sample the nearest grid cell, with every ingredient at its V2 peak."""
    lats, lons = np.asarray(data["latitude"]), np.asarray(data["longitude"])
    rows = []
    for name, lat, lon, _ in CITIES:
        row = dict(city=name, latitude=lat, longitude=lon)
        if not (lats.min() <= lat <= lats.max() and lons.min() <= lon <= lons.max()):
            row["status"] = "outside grid"
            rows.append(row)
            continue
        y, x = int(np.argmin(abs(lats - lat))), int(np.argmin(abs(lons - lon)))
        row.update(grid_latitude=float(lats[y]), grid_longitude=float(lons[x]))
        series = index[:, y, x]
        if not np.isfinite(series).any():
            row["status"] = "no finite V2"
            rows.append(row)
            continue
        t = int(np.argmax(np.where(np.isfinite(series), series, -np.inf)))
        row.update(status="ok", v2=float(series[t]),
                   map_v2=float(display_field[y, x]), peak_hour=float(hours[t]),
                   peak_utc=format_time(valid_time(data["init_time"], hours[t])),
                   peak_local=format_time(valid_time(data["init_time"], hours[t]), local=True),
                   strong_ingredients=int(counts[t, y, x]))
        for key, _, _ in DIAGNOSTICS:
            values = np.asarray(data[key])
            row[key] = float(values[t, y, x] if values.ndim == 3 else values[y, x])
        rows.append(row)
    return rows


def related_output(output_file, suffix, extension="png"):
    path = Path(output_file)
    stem = path.stem.removesuffix("_LATEST")
    return path.with_name(f"{stem}_{suffix}_LATEST.{extension}")


def write_city_csv(path, rows, init_time, hours, variant):
    metadata = dict(variant=variant, init_utc=format_time(init_time),
                    period_start_utc=format_time(valid_time(init_time, hours[0])),
                    period_end_utc=format_time(valid_time(init_time, hours[-1])))
    columns = list(metadata) + list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(dict(metadata, **row) for row in rows)


def draw_city_table(fig, rows):
    ax = fig.add_axes([0.10, 0.083, 0.78, 0.13])
    ax.axis("off")
    cells = []
    for row in rows:
        if row["status"] == "ok":
            cells.append([row["city"], f'{row["v2"]:.1f}', f'+{row["peak_hour"]:g}',
                          row["peak_local"], f'{row["strong_ingredients"]}/5'])
        else:
            cells.append([row["city"], "N/A", "N/A", row["status"], "N/A"])
    table = ax.table(cellText=cells,
                     colLabels=["Nearest grid cell", "Raw V2", "Lead (h)",
                                "Local V2 peak (UTC+3)", "Strong"],
                     colWidths=[0.18, 0.12, 0.12, 0.44, 0.14],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d4dde7")
        if r == 0:
            cell.set_facecolor("#e7eef5")
            cell.set_text_props(weight="bold")


def draw_diagnostics(path, rows, init_time, hours, map_label):
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["city"] in ("Arar", "Turaif")]
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.axis("off")
    labels = ["Raw V2 (0-100)", "Peak lead (h)", "Peak date/time (UTC+3)",
              "Strong ingredients (/5)"] + [label for _, label, _ in DIAGNOSTICS]
    values = []
    for row in selected:
        if row["status"] != "ok":
            values.append(["N/A"] * len(labels))
            continue
        values.append([f'{row["v2"]:.1f}', f'+{row["peak_hour"]:g}',
                       row["peak_local"], str(row["strong_ingredients"])]
                      + [format(row[key], fmt) if np.isfinite(row[key]) else "N/A"
                         for key, _, fmt in DIAGNOSTICS])
    table = ax.table(cellText=[[label] + [v[i] for v in values]
                               for i, label in enumerate(labels)],
                     colLabels=["At each city's own V2 peak"] + [r["city"] for r in selected],
                     colWidths=[0.48, 0.26, 0.26], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d4dde7")
        if r == 0:
            cell.set_facecolor("#e7eef5")
            cell.set_text_props(weight="bold")
    fig.suptitle(f"ECMWF V2 city diagnostics - {map_label}\nInit: {format_time(init_time)}",
                 fontsize=14, weight="bold", y=0.98)
    fig.text(0.5, 0.075,
             f"Period: {format_time(valid_time(init_time, hours[0]), local=True)}"
             f" to {format_time(valid_time(init_time, hours[-1]), local=True)}",
             ha="center", fontsize=10)
    fig.text(0.5, 0.035,
             "Nearest grid point; raw unsmoothed V2. WBZ is above mean sea level, not ground.\n"
             "Omega is model-scale ascent. V2 is not a probability or hail diameter; strong count is uncalibrated.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
