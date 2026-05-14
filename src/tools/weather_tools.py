"""Multi-corridor, multi-waypoint weather risk.

For each corridor, we evaluate weather independently at every waypoint, score
each waypoint per day against the playbook §6 thresholds, then aggregate:

    waypoint_day_score = sum of triggered flags (0..3)
    corridor_day_score = max(waypoint_day_score for waypoints in corridor)
    corridor_48h_score = max(day0_score, day1_score)

API calls are made concurrently across waypoints; the same lat/lon pair is
cached for the duration of the run so a corridor sharing waypoints with
another corridor (e.g. Newark NJ) does not pay the API cost twice.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import os
import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _fetch_waypoint_forecast(lat: float, lon: float, tz: str) -> Dict[str, Any]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,wind_gusts_10m_max,temperature_2m_min",
        "timezone": tz,
        "forecast_days": 2,
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _score_waypoint_day(
    precip_mm: float,
    wind_kmh: float,
    temp_min_c: float,
    thresholds: Dict[str, float],
) -> Tuple[int, Dict[str, bool]]:
    flags = {
        "heavy_rain_risk": precip_mm >= thresholds["heavy_precipitation_mm_day"],
        "high_wind_risk":  wind_kmh  >= thresholds["high_wind_gust_kmh"],
        "freezing_risk":   temp_min_c <= thresholds["freezing_temp_c"],
    }
    score = int(flags["heavy_rain_risk"]) + int(flags["high_wind_risk"]) + int(flags["freezing_risk"])
    return score, flags


def evaluate_corridor_weather(
    corridors: Dict[str, Any],
    weather_thresholds: Dict[str, float],
    travel_buffer_by_risk: Dict[str, Dict[str, Any]],
    tz: str | None = None,
) -> Dict[str, Any]:
    """Return a structured weather/risk report for every corridor.

    Output shape:
        {
          "<corridor_id>": {
            "waypoints": [{waypoint_id, city, day0:{...}, day1:{...}}, ...],
            "day0":      {"risk_score": int, "worst_waypoint": str, "flags": {...}},
            "day1":      {"risk_score": int, "worst_waypoint": str, "flags": {...}},
            "max_48h_risk_score": int,
            "buffer_pct": int,
            "escalation_required": bool,
          },
          ...
        }
    """
    tz = tz or os.getenv("WEATHER_TZ", "America/New_York")

    # Step 1: gather unique waypoints and fetch in parallel (with intra-run cache)
    unique_points: Dict[Tuple[float, float], None] = {}
    for cfg in corridors.values():
        for wp in cfg["waypoints"]:
            unique_points[(float(wp["lat"]), float(wp["lon"]))] = None

    forecasts: Dict[Tuple[float, float], Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(unique_points))) as ex:
        future_to_pt = {
            ex.submit(_fetch_waypoint_forecast, lat, lon, tz): (lat, lon)
            for (lat, lon) in unique_points
        }
        for fut in as_completed(future_to_pt):
            pt = future_to_pt[fut]
            forecasts[pt] = fut.result()

    # Step 2: score per waypoint per day, then aggregate per corridor
    out: Dict[str, Any] = {}
    for corridor_id, cfg in corridors.items():
        wp_reports: List[Dict[str, Any]] = []
        per_day_scores: Dict[str, List[Tuple[str, int, Dict[str, bool]]]] = {"Day0": [], "Day1": []}

        for wp in cfg["waypoints"]:
            forecast = forecasts[(float(wp["lat"]), float(wp["lon"]))]
            daily = forecast.get("daily", {})
            precip = daily.get("precipitation_sum", []) or [0.0, 0.0]
            gusts  = daily.get("wind_gusts_10m_max", []) or [0.0, 0.0]
            tmins  = daily.get("temperature_2m_min", []) or [10.0, 10.0]

            day_entries = {}
            for day_idx, day_label in enumerate(("Day0", "Day1")):
                p = float(precip[day_idx]) if day_idx < len(precip) else 0.0
                w = float(gusts[day_idx])  if day_idx < len(gusts)  else 0.0
                t = float(tmins[day_idx])  if day_idx < len(tmins)  else 10.0
                score, flags = _score_waypoint_day(p, w, t, weather_thresholds)
                day_entries[day_label] = {
                    "precipitation_mm": p,
                    "wind_gust_kmh": w,
                    "temp_min_c": t,
                    "risk_score": score,
                    "flags": flags,
                }
                per_day_scores[day_label].append((wp["waypoint_id"], score, flags))

            wp_reports.append({
                "waypoint_id": wp["waypoint_id"],
                "city": wp["city"],
                "lat": wp["lat"],
                "lon": wp["lon"],
                **day_entries,
            })

        corridor_day_summary: Dict[str, Any] = {}
        for day_label, entries in per_day_scores.items():
            worst_id, worst_score, worst_flags = max(entries, key=lambda x: x[1])
            corridor_day_summary[day_label] = {
                "risk_score": worst_score,
                "worst_waypoint": worst_id,
                "flags": worst_flags,
            }

        max_48h = max(corridor_day_summary["Day0"]["risk_score"],
                      corridor_day_summary["Day1"]["risk_score"])
        buf = travel_buffer_by_risk[str(max_48h)]

        out[corridor_id] = {
            "corridor_name": cfg["corridor_name"],
            "waypoints": wp_reports,
            "Day0": corridor_day_summary["Day0"],
            "Day1": corridor_day_summary["Day1"],
            "max_48h_risk_score": max_48h,
            "buffer_pct": buf["buffer_pct"],
            "escalation_required": buf["escalation"],
        }

    return out


def compute_sla_violation_from_weather(
    corridor_cfg: Dict[str, Any],
    buffer_pct: float,
) -> Dict[str, Any]:
    """Decide whether weather buffer pushes transit time above the SLA cap.

    adjusted_transit_h = base_transit_h * (1 + buffer_pct/100)
    sla_violation = adjusted_transit_h > max_transit_h
    """
    base = float(corridor_cfg["base_transit_hours"])
    cap  = float(corridor_cfg["max_transit_hours"])
    adjusted = base * (1 + buffer_pct / 100.0)
    return {
        "base_transit_hours": base,
        "adjusted_transit_hours": round(adjusted, 2),
        "max_transit_hours": cap,
        "sla_violation": adjusted > cap,
        "headroom_hours": round(cap - adjusted, 2),
    }
