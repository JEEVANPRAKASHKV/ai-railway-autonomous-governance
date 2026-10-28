"""
tests/test_real_data.py
=======================
The platform runs on the real Indian Railways network, official fares and live weather:
- timetable: real trains, halts and distances (datameet/railways, 2016 snapshot);
- fares: Railway Board TAG 2025 slab values and the standard charges;
- search: any two halts of a train form a journey (Kota → Surat finds the Mumbai Rajdhani);
- weather: Open-Meteo forecast at the station's coordinates, climatology when it is unavailable;
- the simulated outcome datasets are generated on that real network.
"""

from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import core.weather as weather
from backend.main import app
from core.config import settings
from core.fares import base_fare, fare_breakdown, payable_fare
from core.network import network

client = TestClient(app)
SOON = str(date.today() + timedelta(days=3))


# ── Timetable ─────────────────────────────────────────────────────────────────

def test_real_trains_and_halts():
    t = network.train("12951")
    assert t is not None and t["source"] == "BCT" and t["destination"] == "NDLS" and int(t["distance_km"]) == 1384
    halts = network.stops_of("12951")["station_code"].tolist()
    assert halts[0] == "BCT" and halts[-1] == "NDLS" and {"ST", "BRC", "RTM", "KOTA"} <= set(halts)
    kota = network.stop("12951", "KOTA")
    assert 890 <= kota["km_from_origin"] <= 940          # real: 917 km from Mumbai Central
    assert kota["arrival"] == "03:20"


def test_any_two_halts_form_a_journey():
    seg = network.segments("KOTA", "ST")
    row = seg[seg["train_number"] == "12952"].iloc[0]   # Mumbai Rajdhani NDLS → BCT passes Kota, then Surat
    assert row["source"] == "KOTA" and row["destination"] == "ST" and row["origin"] == "NDLS"
    assert 600 <= row["distance_km"] <= 700 and row["duration_hours"] < 12
    assert row["base_fare_3a"] == payable_fare("Rajdhani", "3A", row["distance_km"], "12952")


def test_city_stations_are_searched_together():
    seg = network.segments("NDLS", "CSTM", nearby=True)
    assert len(seg) > 3 and set(seg["destination"]) - {"CSTM"}          # BCT / LTT / BDTS / DR trains too
    assert set(seg["source"]) <= network.city_stations("NDLS")


def test_no_route_is_invented():
    assert network.segments("NDLS", "XYZ").empty
    assert network.segments("RNC", "PUNE").empty and network.connections("RNC", "PUNE")


# ── Official fares ────────────────────────────────────────────────────────────

def test_tag_2025_slab_values():
    assert base_fare("Express", "SL", 500) == 271
    assert base_fare("Express", "2S", 1000) == 273
    assert base_fare("Superfast", "3A", 1000) == 1196
    assert base_fare("Rajdhani", "3A", 1000) == 1373
    assert base_fare("Shatabdi", "CC", 300) == 446
    assert base_fare("Garib Rath", "3A", 1500) == 972
    assert base_fare("Rajdhani", "EC", 500) is None                     # class not offered on this kind of train
    fares = [base_fare("Express", "SL", km) for km in range(50, 2600, 50)]
    assert fares == sorted(fares)                                        # never cheaper for a longer trip


def test_payable_fare_adds_the_standard_charges():
    sl = fare_breakdown("Superfast", "SL", 1000, "12627")
    assert (sl["base_fare"], sl["reservation_charge"], sl["superfast_charge"], sl["gst"], sl["total"]) == (459, 20, 30, 0, 509)
    ac = fare_breakdown("Superfast", "3A", 1000, "12627")
    assert ac["gst"] == round(0.05 * (1196 + 40 + 45)) and ac["total"] == 1345
    tq = fare_breakdown("Superfast", "3A", 1000, "12627", tatkal=True)
    assert 300 <= tq["tatkal_charge"] <= 400
    assert fare_breakdown("Express", "SL", 1000, "11301")["superfast_charge"] == 0


def test_trains_csv_carries_official_fares():
    t = pd.read_csv(f"{settings.data_raw_dir}/trains.csv", dtype={"train_number": str}).set_index("train_number")
    assert t.loc["12951", "base_fare_3a"] == payable_fare("Rajdhani", "3A", 1384, "12951")
    real = pd.read_csv(f"{settings.project_root}/data/real/trains.csv", dtype={"train_number": str})
    assert set(t.index) <= set(real["train_number"])


# ── Services on the real network ──────────────────────────────────────────────

def test_recommender_finds_trains_through_intermediate_stations():
    r = client.post("/api/v1/recommend/trains", json={"source_station": "KOTA", "destination_station": "ST",
                                                       "journey_date": SOON, "preferred_class": "3A"}).json()
    recs = r["recommendations"]
    assert recs and all(x["source"] == "KOTA" and x["destination"] == "ST" for x in recs)
    assert all(network.train(x["train_number"])["source"] != "KOTA" for x in recs)     # none of them starts at Kota
    seg = network.segments("KOTA", "ST").set_index("train_number")
    for x in recs:
        if seg.loc[x["train_number"], "base_fare_3a"] > 0:                              # fare for Kota → Surat only
            assert x["fare_estimate_inr"] == seg.loc[x["train_number"], "base_fare_3a"]
        else:
            assert "No 3A" in x["tags"] and x["fare_class"] != "3A"


def test_delay_uses_the_timetable_distance_and_schedule():
    r = client.post("/api/v1/predict/delay", json={"train_number": "12951", "station_code": "KOTA",
                                                    "journey_date": SOON}).json()
    assert r["distance_source"] == "timetable" and 890 <= r["distance_from_origin_km"] <= 940
    assert r["scheduled_time"] == "03:20"
    assert r["weather_source"].startswith("station climatology")        # live weather is off in tests


def test_fare_is_fixed_on_mail_express_and_flexi_on_rajdhani():
    fixed = client.post("/api/v1/predict/fare", json={"train_number": "12627", "journey_class": "SL",
                                                       "journey_date": str(date.today() + timedelta(days=30))}).json()
    assert fixed["pricing_rule"] == "fixed"
    assert len({p["predicted_fare_inr"] for p in fixed["forecast_curve"]}) == 1
    assert fixed["current_predicted_fare_inr"] == fixed["fare_breakdown"]["total"]
    flexi = client.post("/api/v1/predict/fare", json={"train_number": "12951", "journey_class": "3A",
                                                       "journey_date": str(date.today() + timedelta(days=30))}).json()
    assert flexi["pricing_rule"] == "flexi"
    assert flexi["base_fare_inr"] <= flexi["current_predicted_fare_inr"] <= 1.4 * flexi["base_fare_inr"] + 1


def test_fare_for_part_of_the_route():
    full = client.post("/api/v1/predict/fare", json={"train_number": "12952", "journey_class": "2A",
                                                      "journey_date": SOON}).json()
    part = client.post("/api/v1/predict/fare", json={"train_number": "12952", "journey_class": "2A",
                                                      "journey_date": SOON, "source_station": "KOTA",
                                                      "destination_station": "ST"}).json()
    assert (part["source_station"], part["destination_station"]) == ("KOTA", "ST")
    assert part["base_fare_inr"] < full["base_fare_inr"]
    assert part["fare_breakdown"]["distance_km"] < 700


def test_planner_finds_a_train_between_intermediate_stations():
    r = client.post("/api/v1/agent/plan", json={"query": "Kota to Surat tomorrow in 3A", "user_id": "USR00002"}).json()
    opt = r.get("recommended_itinerary") or {}
    assert r["status"] == "PENDING_APPROVAL" and opt.get("source") == "KOTA" and opt.get("destination") == "ST"


# ── Weather ───────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, j):
        self._j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self._j


def _fog_day(*_a, **_k):
    return _Resp({"daily": {"time": [SOON], "precipitation_sum": [0.0], "weather_code": [45], "wind_speed_10m_max": [8.0]},
                  "hourly": {"visibility": [3000.0] * 6 + [150.0] * 4 + [5000.0] * 14}})


@pytest.fixture
def live_weather(monkeypatch):
    weather._cache.clear()
    monkeypatch.setattr(settings, "weather_live", True)
    yield monkeypatch
    weather._cache.clear()


def test_weather_index_scale():
    assert weather.weather_index(0) == 0.0
    assert weather.weather_index(0, 150) == 6.0                        # dense fog
    assert 4.5 < weather.weather_index(45) < 5.0                       # heavy rain
    assert weather.weather_index(200, 100, True, 80) == 10.0           # capped


def test_live_forecast_is_used_for_the_station(live_weather):
    calls = []
    live_weather.setattr("requests.get", lambda url, params=None, timeout=None: calls.append(params) or _fog_day())
    lat, lon = network.coords("NDLS")
    w = weather.weather_for(lat, lon, date.fromisoformat(SOON))
    assert w["source"] == "Open-Meteo forecast" and w["conditions"]["min_visibility_m"] == 150 and w["index"] == 6.0
    assert abs(calls[0]["latitude"] - 28.642) < 0.01 and calls[0]["start_date"] == SOON
    weather.weather_for(lat, lon, date.fromisoformat(SOON))
    assert len(calls) == 1                                              # cached


def test_delay_forecast_with_live_fog(live_weather):
    live_weather.setattr("requests.get", lambda *a, **k: _fog_day())
    r = client.post("/api/v1/predict/delay", json={"train_number": "12301", "station_code": "NDLS",
                                                    "journey_date": SOON}).json()
    assert r["weather_source"] == "Open-Meteo forecast" and r["weather_index_used"] == 6.0
    assert any("fog" in a["title"].lower() for a in r["alerts"])


def test_weather_falls_back_to_climatology(live_weather):
    def down(*_a, **_k):
        raise ConnectionError("offline")
    live_weather.setattr("requests.get", down)
    w = weather.weather_for(28.6, 77.2, date.fromisoformat(SOON))
    assert w["source"].startswith("station climatology") and 0 <= w["index"] <= 10
    far = weather.weather_for(28.6, 77.2, date.today() + timedelta(days=45))
    assert "outside the forecast window" in far["source"]


def test_climatology_matches_indian_seasons():
    delhi_jan = weather.climatology_index(28.6, 77.2, date(2026, 1, 10))
    delhi_apr = weather.climatology_index(28.6, 77.2, date(2026, 4, 10))
    mumbai_jul = weather.climatology_index(19.0, 72.8, date(2026, 7, 15))
    mumbai_feb = weather.climatology_index(19.0, 72.8, date(2026, 2, 15))
    assert delhi_jan > 3 * delhi_apr                                    # winter fog over the plains
    assert mumbai_jul > 5 * mumbai_feb                                  # south-west monsoon


# ── Simulated outcomes sit on the real network ────────────────────────────────

def test_delay_history_uses_real_halts():
    d = pd.read_csv(f"{settings.data_raw_dir}/delay_logs.csv", nrows=3000, dtype={"train_number": str})
    for r in d.sample(40, random_state=1).itertuples():
        stop = network.stop(r.train_number, r.station_code)
        assert stop is not None and abs(stop["km_from_origin"] - r.distance_from_origin_km) < 0.2


# ── Place names → stations (any real station, common names, small typos) ─────

@pytest.mark.parametrize("text,code", [
    ("kasaragod", "KGQ"), ("Kasargod", "KGQ"), ("kasaragod jn", "KGQ"), ("calicut", "CLT"), ("trichy", "TPJ"),
    ("sealdah", "SDAH"), ("bandra", "BDTS"), ("mumbai central", "BCT"), ("bhopal jn", "BPL"), ("delhi", "NDLS"),
    ("NDLS", "NDLS"),
])
def test_station_names_resolve(text, code):
    assert network.resolve(text) == code


def test_unknown_place_is_not_guessed_but_suggested():
    assert network.resolve("xyzabc") is None
    assert "Kasaragod (KGQ)" in network.suggest("kasaragodu")


def test_planner_understands_any_real_station():
    r = client.post("/api/v1/agent/plan", json={"query": "I want to travel from kasaragod to Delhi tomorrow under ₹1500 "
                                                         "and reach before 10:00 AM.", "user_id": "USR00002"}).json()
    opt = r.get("recommended_itinerary") or {}
    assert r["status"] == "PENDING_APPROVAL" and opt.get("source") == "KGQ"
    assert opt.get("destination") in network.city_stations("NDLS")


def test_build_fingerprint_tracks_code_changes(tmp_path):
    from core.build_info import fingerprint
    (tmp_path / "core").mkdir()
    f = tmp_path / "core" / "x.py"
    f.write_text("a = 1\n")
    before = fingerprint(str(tmp_path))
    f.write_text("a = 22\n")
    assert fingerprint(str(tmp_path)) != before
    (tmp_path / "models" / "registry").mkdir(parents=True)
    after = fingerprint(str(tmp_path))
    (tmp_path / "models" / "registry" / "lifecycle_state.json").write_text("{}")   # run-time writes don't count
    assert fingerprint(str(tmp_path)) == after
