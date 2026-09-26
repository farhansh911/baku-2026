"""Baku 2026 winner likelihood, following the process in the Baku video.

1. Load the season from FastF1.
2. Build the normalized feature list shown on screen.
3. Train and test an XGBoost regressor on 2026 races only.
4. Run 25,000 simulations and report how often each driver wins.

2024 and 2025 are left out. Those cars are from the previous regulations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
FASTF1_CACHE = CACHE / "fastf1"
WEEKENDS_PATH = CACHE / "weekends_2026.json"
FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(FASTF1_CACHE))
logging.getLogger("fastf1").setLevel(logging.ERROR)

YEAR = 2026
BAKU_ROUND = 15
TEST_FROM_ROUND = 11
SIMULATIONS = 25_000
ALPHA = 0.45
HISTORY = 8

# How much a good grid tends to stick. Baku is a high-speed street circuit:
# the main straight allows passes, so the grid matters less than at Monaco.
GRID_STICKINESS = {
    "Melbourne": 0.62,
    "Shanghai": 0.58,
    "Suzuka": 0.70,
    "Miami Gardens": 0.68,
    "Montréal": 0.55,
    "Monte Carlo": 0.95,
    "Barcelona": 0.78,
    "Spielberg": 0.48,
    "Silverstone": 0.50,
    "Spa-Francorchamps": 0.42,
    "Budapest": 0.84,
    "Zandvoort": 0.72,
    "Monza": 0.38,
    "Madrid": 0.60,
    "Baku": 0.70,
}

FEATURES = [
    "grid_score",
    "quali_pos_norm",
    "quali_gap_sec",
    "practice_best_sec",
    "finish_pos_ew",
    "finish_norm_ew",
    "team_finish_norm_ew",
    "spread_per_race",
    "speed_bias",
    "overtaking_pace",
    "tyre_strategy",
    "grid_track_position",
    "air_temp_c",
    "wet_session",
]
# Higher feature value means a worse predicted finish, except the negative ones.
MONOTONE = (-1, 1, 1, 1, 1, 1, 1, 0, -1, -1, 0, -1, 0, 1)


def ew_values(values, alpha: float = ALPHA) -> float:
    """Recent-weighted mean. Pass only races already completed, oldest first."""
    if not values:
        return np.nan
    estimate = float(values[0])
    for value in list(values)[1:]:
        estimate = alpha * float(value) + (1.0 - alpha) * estimate
    return estimate


def seconds(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except TypeError:
        pass
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return np.nan


def load_session(rnd: int, identifier: str, weather: bool):
    session = fastf1.get_session(YEAR, rnd, identifier)
    session.load(telemetry=False, weather=weather, messages=False)
    return session


def practice_times(rnd: int) -> dict[str, float]:
    best: dict[str, float] = {}
    for identifier in ("FP1", "FP2", "FP3"):
        try:
            session = load_session(rnd, identifier, weather=False)
        except Exception:
            continue
        laps = session.laps
        if laps is None or laps.empty or "LapTime" not in laps.columns:
            continue
        usable = laps.dropna(subset=["LapTime"])
        if "IsAccurate" in usable.columns:
            accurate = usable[usable["IsAccurate"] == True]  # noqa: E712
            if not accurate.empty:
                usable = accurate
        for code, group in usable.groupby("Driver"):
            lap = seconds(group["LapTime"].min())
            if not np.isfinite(lap):
                continue
            best[code] = min(best.get(code, lap), lap)
    return best


def tyre_stops(session) -> dict[str, float]:
    laps = session.laps
    if laps is None or laps.empty or "Stint" not in laps.columns:
        return {}
    counted = laps.dropna(subset=["Stint"]).groupby("Driver")["Stint"].nunique()
    return {code: float(max(stints - 1, 0)) for code, stints in counted.items()}


def session_weather(session) -> tuple[float, float]:
    weather = getattr(session, "weather_data", None)
    if weather is None or len(weather) == 0:
        return np.nan, np.nan
    air = float(pd.to_numeric(weather["AirTemp"], errors="coerce").median())
    if "Rainfall" not in weather.columns:
        return air, np.nan
    wet = float(pd.Series(weather["Rainfall"]).fillna(False).astype(bool).any())
    return air, wet


def extract_round(rnd: int, event_name: str, location: str) -> dict:
    qualifying = load_session(rnd, "Q", weather=True)
    race = None
    if rnd != BAKU_ROUND:
        race = load_session(rnd, "R", weather=False)
    air, wet = session_weather(qualifying)
    practice = practice_times(rnd)
    stops = tyre_stops(race) if race is not None else {}
    race_rows = {}
    if race is not None:
        for _, row in race.results.iterrows():
            race_rows[row["Abbreviation"]] = row

    drivers = []
    for _, row in qualifying.results.iterrows():
        code = row["Abbreviation"]
        quali_laps = [seconds(row.get(name)) for name in ("Q3", "Q2", "Q1")]
        quali_laps = [lap for lap in quali_laps if np.isfinite(lap)]
        race_row = race_rows.get(code)
        finish = np.nan
        grid = np.nan
        status = ""
        if race_row is not None:
            finish = pd.to_numeric(race_row.get("Position"), errors="coerce")
            grid = pd.to_numeric(race_row.get("GridPosition"), errors="coerce")
            status = str(race_row.get("Status") or "")
        drivers.append(
            {
                "code": code,
                "name": f"{row['FirstName']} {row['LastName']}".strip(),
                "team": row["TeamName"],
                "quali_pos": float(row["Position"]),
                "quali_best": min(quali_laps) if quali_laps else np.nan,
                "grid": float(grid) if pd.notna(grid) else np.nan,
                "finish": float(finish) if pd.notna(finish) else np.nan,
                "stops": stops.get(code, np.nan),
                "practice_best": practice.get(code, np.nan),
                "status": status,
            }
        )
    return {
        "round": rnd,
        "event": event_name,
        "location": location,
        "air_temp_c": air,
        "wet_session": wet,
        "drivers": drivers,
    }


def load_weekends() -> list[dict]:
    stored = json.loads(WEEKENDS_PATH.read_text()) if WEEKENDS_PATH.exists() else []
    done = {int(item["round"]) for item in stored}
    schedule = fastf1.get_event_schedule(YEAR)
    schedule = schedule[schedule["RoundNumber"].between(1, BAKU_ROUND)]
    for event in schedule.itertuples(index=False):
        rnd = int(event.RoundNumber)
        if rnd in done:
            continue
        print(f"  FastF1  R{rnd:02d} {event.EventName}", flush=True)
        stored.append(extract_round(rnd, event.EventName, event.Location))
        WEEKENDS_PATH.write_text(json.dumps(stored))
        done.add(rnd)
    return sorted(stored, key=lambda item: item["round"])


def is_retirement(status: str) -> bool:
    text = status.lower()
    if text in {"", "finished"}:
        return False
    if "lap" in text:
        return False
    return True


class History:
    def __init__(self):
        self.finish: dict[str, list] = {}
        self.gain_vs_quali: dict[str, list] = {}
        self.gain_vs_grid: dict[str, list] = {}
        self.stops: dict[str, list] = {}
        self.team_finish: dict[str, list] = {}

    def _push(self, store: dict, key: str, value: float):
        if not np.isfinite(value):
            return
        bucket = store.setdefault(key, [])
        bucket.append(float(value))
        del bucket[:-HISTORY]

    def form(self, code: str, team: str) -> dict:
        finishes = self.finish.get(code, [])
        field = 22.0
        norms = [value / field for value in finishes]
        return {
            "finish_pos_ew": ew_values(finishes),
            "finish_norm_ew": ew_values(norms),
            "team_finish_norm_ew": ew_values(self.team_finish.get(team, [])),
            "spread_per_race": float(np.std(finishes)) if len(finishes) >= 2 else np.nan,
            "speed_bias": ew_values(self.gain_vs_quali.get(code, [])),
            "overtaking_pace": ew_values(self.gain_vs_grid.get(code, [])),
            "tyre_strategy": ew_values(self.stops.get(code, [])),
        }

    def update_driver(self, code: str, finish: float, quali_pos: float, grid: float, stops: float):
        self._push(self.finish, code, finish)
        if np.isfinite(finish) and np.isfinite(quali_pos):
            self._push(self.gain_vs_quali, code, quali_pos - finish)
        if np.isfinite(finish) and np.isfinite(grid) and grid > 0:
            self._push(self.gain_vs_grid, code, grid - finish)
        self._push(self.stops, code, stops)

    def update_team(self, team: str, best_finish_norm: float):
        self._push(self.team_finish, team, best_finish_norm)


def weekend_features(weekend: dict, history: History, for_prediction: bool) -> list[dict]:
    drivers = weekend["drivers"]
    field = max(len(drivers), 1)
    pole = min(
        (driver["quali_best"] for driver in drivers if np.isfinite(driver["quali_best"])),
        default=np.nan,
    )
    practice_pole = min(
        (driver["practice_best"] for driver in drivers if np.isfinite(driver["practice_best"])),
        default=np.nan,
    )
    stickiness = GRID_STICKINESS.get(weekend["location"], 0.65)
    rows = []
    for driver in drivers:
        grid = driver["grid"]
        if for_prediction or not np.isfinite(grid) or grid <= 0:
            grid = driver["quali_pos"]
        grid_score = 1 - (grid - 1) / max(field - 1, 1)
        practice = driver["practice_best"]
        rows.append(
            {
                **history.form(driver["code"], driver["team"]),
                "grid_score": grid_score,
                "quali_pos_norm": driver["quali_pos"] / field,
                "quali_gap_sec": driver["quali_best"] - pole if np.isfinite(driver["quali_best"]) else np.nan,
                "practice_best_sec": practice - practice_pole if np.isfinite(practice) and np.isfinite(practice_pole) else np.nan,
                "grid_track_position": grid_score * stickiness,
                "air_temp_c": weekend["air_temp_c"],
                "wet_session": weekend["wet_session"],
                "round": weekend["round"],
                "event": weekend["event"],
                "code": driver["code"],
                "name": driver["name"],
                "team": driver["team"],
                "quali_pos": driver["quali_pos"],
                "finish": driver["finish"],
                "retired": is_retirement(driver["status"]),
            }
        )
    if not for_prediction:
        team_finishes: dict[str, list] = {}
        for driver in drivers:
            grid = driver["grid"] if np.isfinite(driver["grid"]) and driver["grid"] > 0 else driver["quali_pos"]
            history.update_driver(driver["code"], driver["finish"], driver["quali_pos"], grid, driver["stops"])
            if np.isfinite(driver["finish"]):
                team_finishes.setdefault(driver["team"], []).append(driver["finish"])
        for team, finishes in team_finishes.items():
            history.update_team(team, min(finishes) / field)
    return rows


def build_table(weekends: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, History]:
    history = History()
    past = []
    live = []
    for weekend in weekends:
        predicting = weekend["round"] == BAKU_ROUND
        rows = weekend_features(weekend, history, for_prediction=predicting)
        if predicting:
            live.extend(rows)
        else:
            past.extend(rows)
    return pd.DataFrame(past), pd.DataFrame(live), history


def new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        objective="reg:squarederror",
        monotone_constraints=MONOTONE,
        random_state=42,
        n_jobs=2,
        verbosity=0,
    )


def fit_model(frame: pd.DataFrame) -> XGBRegressor:
    model = new_model()
    learned = frame.dropna(subset=["finish"])
    model.fit(learned[FEATURES], learned["finish"])
    return model


def simulate(predicted: np.ndarray, sigma: float, retire_rate: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=(SIMULATIONS, len(predicted)))
    retired = rng.random((SIMULATIONS, len(predicted))) < retire_rate
    score = predicted[None, :] + noise
    score[retired] = 1e9
    winners = score.argmin(axis=1)
    return np.bincount(winners, minlength=len(predicted))


def race_picks(frame: pd.DataFrame, model: XGBRegressor) -> None:
    print("Test on the last 2026 races before Baku")
    hits = 0
    races = 0
    for rnd, race in frame.groupby("round"):
        race = race.reset_index(drop=True)
        predicted = model.predict(race[FEATURES])
        pick = race.iloc[int(np.argmin(predicted))]
        actual = race.loc[race["finish"].idxmin()]
        correct = pick["code"] == actual["code"]
        hits += int(correct)
        races += 1
        mark = "correct" if correct else "missed"
        print(
            f"  R{int(rnd):>2} {race['event'].iloc[0]:<24} "
            f"model {pick['name']:<22} {mark:<7} winner {actual['name']}"
        )
    print(f"Test: {hits}/{races} winners.")
    print()


def main() -> None:
    print("Loading 2026 from FastF1. Older seasons are not used.")
    weekends = load_weekends()
    past, live, _history = build_table(weekends)
    past.to_csv(ROOT / "training_2026.csv", index=False)
    train = past[past["round"] < TEST_FROM_ROUND]
    test = past[past["round"] >= TEST_FROM_ROUND].copy()
    print(
        f"2026 rows: {len(past)} driver-races, "
        f"train R1–R{TEST_FROM_ROUND - 1}, test R{TEST_FROM_ROUND}–R{BAKU_ROUND - 1}."
    )
    print()

    evaluation = fit_model(train)
    race_picks(test, evaluation)
    test["predicted"] = evaluation.predict(test[FEATURES])
    residual = (test["finish"] - test["predicted"]).dropna()
    sigma = float(residual.std(ddof=1))
    if not np.isfinite(sigma) or sigma < 0.5:
        sigma = 0.5
    retire_rate = float(past["retired"].mean())
    print(f"Position error on the test races: {sigma:.2f} places.")
    print(f"2026 retirement rate used in the simulations: {retire_rate:.1%}.")
    print()

    final = fit_model(past)
    final.save_model(ROOT / "baku_2026_regressor.json")
    live = live.copy()
    live["predicted_finish"] = final.predict(live[FEATURES])
    live["wins"] = simulate(live["predicted_finish"].to_numpy(), sigma, retire_rate)
    live["win_pct"] = 100 * live["wins"] / SIMULATIONS
    live = live.sort_values(["win_pct", "quali_pos"], ascending=[False, True]).reset_index(drop=True)
    live.to_csv(ROOT / "baku_likelihoods.csv", index=False)

    baku = next(item for item in weekends if item["round"] == BAKU_ROUND)
    wet = "wet" if baku["wet_session"] >= 0.5 else "dry"
    print(f"Azerbaijan Grand Prix {YEAR}")
    print(
        f"Qualifying weather: {baku['air_temp_c']:.1f}°C, {wet}. "
        f"{SIMULATIONS:,} simulations."
    )
    print()
    print(f"{'#':>2}  {'Driver':<24} {'Team':<16} {'Quali':>5}  {'Sims won':>8}  {'Likelihood':>10}")
    for rank, row in live.iterrows():
        print(
            f"{rank + 1:>2}  {row['name']:<24} {row['team']:<16} "
            f"P{int(row['quali_pos']):>2}   {int(row['wins']):>8}   {row['win_pct']:>8.1f}%"
        )
    winner = live.iloc[0]
    print()
    print(
        f"Highest likelihood: {winner['name']} ({winner['team']}) — "
        f"{int(winner['wins']):,} of {SIMULATIONS:,} ({winner['win_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()
