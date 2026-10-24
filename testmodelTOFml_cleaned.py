"""Compare an Earth-to-Mars ML surrogate, Lambert solution, and N-body physics.

Expected files in the same directory:
    mars_nbody_ml.py          Physical model and orbital mechanics functions
    mars_nbody_surrogate.joblib  Trained ML model bundle

The ML model predicts departure and arrival delta-v magnitudes. Its predicted
initial velocity direction is taken from the Lambert/boundary guess, then the
spacecraft is propagated with the same N-body integration used by the physical
model. The test criterion uses the STANDARD Mars sphere of influence (SOI),
without the 0.7 training-target safety factor.

Single-date TOF search modes (chosen interactively when no flag is given):
    1 / --physics-tof-search   physics-only coarse/fine TOF grid (slowest, most reliable)
    2 / --ml-tof-search        ML scans every TOF, physics verifies a short list (faster)
    3 / --ml-only-tof          ML scans every TOF and the single lowest prediction is
                               used directly; one physics solve checks it (fastest)

If no mode flag is passed (and --tof-days is not given), the script
asks which method you want. The time spent waiting for your answer is NOT
included in the TOF search wall-clock time.

The wall-clock time of the TOF search is printed after the search and also
appended to the performance CSV.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import mars_nbody_mlv2 as physics


# ---------------------------------------------------------------------------
# 1. Default settings
# ---------------------------------------------------------------------------
MODEL_FILE = "mars_nbody_surrogate.joblib"
DEFAULT_DEPARTURE_DATE = "2030-09-01"
MIN_TOF_DAYS = 150
MAX_TOF_DAYS = 400
COARSE_TOF_STEP_DAYS = 10
FINE_TOF_STEP_DAYS = 2
SAMPLE_COUNT = 600

# A valid physical shooting solution should have a small residual to its
# target point on the Mars SOI. This is NOT the spacecraft's distance from
# Mars's centre; that is calculated separately from the propagated trajectory.
SHOOTING_RESIDUAL_TOLERANCE_KM = 10_000.0


# ---------------------------------------------------------------------------
# 2. Date and general helpers
# ---------------------------------------------------------------------------
def date_to_absolute_days(date_string: str) -> float:
    """Convert YYYY-MM-DD to days since J2000 (2000-01-01 12:00 UTC)."""
    dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / physics.DAY_S


def as_2d_array(values: Any) -> np.ndarray:
    """Convert an array-like object to a finite-looking 2-D NumPy array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector, with a helpful error for a zero vector."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector.")
    return vector / norm


def standard_mars_soi_km() -> float:
    """Calculate the standard Laplace sphere-of-influence radius of Mars.

    This deliberately does not use physics.MARS_SOI_KM, because the physical
    module's value may contain the 0.7 factor used for the training target.
    """
    mars_semimajor_axis_km = 1.523679 * physics.AU_KM
    return mars_semimajor_axis_km * (physics.MU_MARS / physics.MU_SUN) ** (2.0 / 5.0)


def get_bundle_item(bundle: Any, names: Iterable[str], required: bool = True) -> Any:
    """Read the first matching alias from a dict-like model bundle."""
    if isinstance(bundle, dict):
        for name in names:
            if name in bundle and bundle[name] is not None:
                return bundle[name]
    if required:
        raise KeyError(
            "Could not find any of these required model-bundle fields: "
            + ", ".join(names)
        )
    return None


def load_model_bundle(model_file: str) -> Dict[str, Any]:
    """Load the trained model and its feature/target scalers."""
    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Model file not found: {model_file}\n"
            "Place the trained mars_nbody_surrogate.joblib file beside this script "
            "or pass --model-file with its path."
        )

    loaded = joblib.load(model_file)
    if not isinstance(loaded, dict):
        # Also support a bare estimator, if no scalers were used during training.
        return {"model": loaded, "x_scaler": None, "y_scaler": None}

    model = get_bundle_item(
        loaded,
        ("model", "ml_model", "surrogate", "regressor", "estimator", "best_model"),
    )
    x_scaler = get_bundle_item(
        loaded,
        ("x_scaler", "X_scaler", "scaler_X", "scaler_x", "feature_scaler"),
        required=False,
    )
    y_scaler = get_bundle_item(
        loaded,
        ("y_scaler", "Y_scaler", "scaler_y", "scaler_Y", "target_scaler"),
        required=False,
    )
    feature_names = get_bundle_item(loaded, ("feature_names", "features"), required=False)
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": feature_names,
        "raw_bundle": loaded,
    }


# ---------------------------------------------------------------------------
# 3. ML prediction
# ---------------------------------------------------------------------------
def predict_delta_v(
    bundle: Dict[str, Any], departure_days: float, tof_days: float
) -> Tuple[float, float]:
    """Predict departure and arrival delta-v magnitudes in km/s.

    The feature vector is produced by physics.trajectory_features(), so this
    test uses the same feature definition as the training dataset generator.
    """
    feature_values = physics.trajectory_features(departure_days, tof_days)

    if isinstance(feature_values, dict):
        feature_names = bundle.get("feature_names")
        if feature_names is not None:
            features = np.asarray(
                [feature_values[name] for name in feature_names], dtype=float
            )
        else:
            # The shared physics function returns the features in a stable order.
            features = np.asarray(list(feature_values.values()), dtype=float)
    else:
        features = np.asarray(feature_values, dtype=float).reshape(-1)

    x = features.reshape(1, -1)
    x_scaler = bundle.get("x_scaler")
    if x_scaler is not None:
        x = x_scaler.transform(x)

    model = bundle["model"]
    y_scaled = np.asarray(model.predict(x), dtype=float)
    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(1, -1)

    y_scaler = bundle.get("y_scaler")
    y = y_scaler.inverse_transform(y_scaled) if y_scaler is not None else y_scaled
    y = np.asarray(y, dtype=float).reshape(-1)

    if y.size < 2:
        raise ValueError(
            "The trained model must predict two values: departure delta-v and "
            "arrival delta-v. Check the training target and saved model bundle."
        )
    if not np.all(np.isfinite(y[:2])):
        raise ValueError(f"Model returned a non-finite delta-v prediction: {y[:2]}")

    # Delta-v magnitudes cannot be negative. Clipping only prevents a small
    # negative regression prediction from creating a physically invalid burn.
    return max(0.0, float(y[0])), max(0.0, float(y[1]))


# ---------------------------------------------------------------------------
# 4. Construct the ML initial state and propagate its trajectory
# ---------------------------------------------------------------------------
def get_boundary_guess(departure_days: float, tof_days: float) -> Dict[str, np.ndarray]:
    """Get the initial state and Lambert-like direction from the physics module."""
    guess = physics.make_boundary_guess(departure_days, tof_days)

    if isinstance(guess, dict):
        required = ("r0", "v_earth", "v0_guess")
        missing = [key for key in required if key not in guess]
        if missing:
            raise KeyError(
                "make_boundary_guess() did not return required fields: "
                + ", ".join(missing)
            )
        return {
            "r0": np.asarray(guess["r0"], dtype=float),
            "v_earth": np.asarray(guess["v_earth"], dtype=float),
            "v0_guess": np.asarray(guess["v0_guess"], dtype=float),
        }

    # Fallback for a tuple return convention: (r0, v_earth, v0_guess, ...).
    if isinstance(guess, (tuple, list)) and len(guess) >= 3:
        return {
            "r0": np.asarray(guess[0], dtype=float),
            "v_earth": np.asarray(guess[1], dtype=float),
            "v0_guess": np.asarray(guess[2], dtype=float),
        }

    raise TypeError(
        "Unsupported return value from physics.make_boundary_guess(). "
        "Expected a dict with r0, v_earth, and v0_guess."
    )


def make_ml_initial_state(
    departure_days: float, tof_days: float, dv_departure_ml: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the ML initial position and heliocentric velocity.

    The model predicts the departure burn from a circular Earth parking orbit.
    Convert that burn magnitude into hyperbolic excess speed using

        v_inf^2 = (v_circular + delta_v)^2 - 2 * mu / r_parking
                = (v_circular + delta_v)^2 - 2 * v_circular^2.

    (The periapsis speed on the hyperbola is v_circular + delta_v, and
    v_periapsis^2 = v_inf^2 + 2*mu/r.) The direction is the direction of the
    boundary/Lambert estimate. This is intentionally not a shooting
    correction: it tests the ML prediction as-is.
    """
    guess = get_boundary_guess(departure_days, tof_days)
    r0 = guess["r0"]
    v_earth = guess["v_earth"]
    v0_guess = guess["v0_guess"]

    # The nominal transfer direction is the difference between the guessed
    # spacecraft departure velocity and Earth's heliocentric velocity.
    departure_direction = unit(v0_guess - v_earth)

    parking_radius_km = physics.R_EARTH + physics.EARTH_PARKING_ALT_KM
    v_circular_kms = math.sqrt(physics.MU_EARTH / parking_radius_km)
    dv = max(0.0, float(dv_departure_ml))
    # FIXED: previously "- v_circular_kms**2", which overstated v_inf.
    v_inf_sq = (v_circular_kms + dv) ** 2 - 2.0 * v_circular_kms**2
    v_inf_kms = math.sqrt(max(0.0, v_inf_sq))

    v0 = v_earth + departure_direction * v_inf_kms
    return r0, v0


def propagate_ml_trajectory(
    departure_days: float,
    tof_days: float,
    dv_departure_ml: float,
    sample_count: int = SAMPLE_COUNT,
) -> Dict[str, np.ndarray]:
    """Propagate the uncorrected ML departure state using the N-body dynamics."""
    r0, v0 = make_ml_initial_state(departure_days, tof_days, dv_departure_ml)
    positions, velocities, solution = physics.integrate_spacecraft(
        departure_days,
        tof_days,
        r0,
        v0,
        sample_count=sample_count,
    )

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Unexpected positions shape from integrator: {positions.shape}")
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError(f"Unexpected velocities shape from integrator: {velocities.shape}")

    # integrate_spacecraft() returns solution.t in seconds, according to the
    # physical model's interface. Fall back to a uniform grid if unavailable.
    if hasattr(solution, "t") and len(solution.t) == len(positions):
        times_days = np.asarray(solution.t, dtype=float) / physics.DAY_S
    else:
        times_days = np.linspace(0.0, float(tof_days), len(positions))

    return {
        "positions": positions,
        "velocities": velocities,
        "times_days": times_days,
        "r0": np.asarray(r0, dtype=float),
        "v0": np.asarray(v0, dtype=float),
    }


# ---------------------------------------------------------------------------
# 5. Lambert two-body solution and its N-body propagated prediction
# ---------------------------------------------------------------------------
def stumpff_c(z: float) -> float:
    """Stumpff C(z), used by the universal-variable Lambert solver."""
    if z > 1.0e-8:
        root = math.sqrt(z)
        return (1.0 - math.cos(root)) / z
    if z < -1.0e-8:
        root = math.sqrt(-z)
        return (math.cosh(root) - 1.0) / (-z)
    return 0.5


def stumpff_s(z: float) -> float:
    """Stumpff S(z), used by the universal-variable Lambert solver."""
    if z > 1.0e-8:
        root = math.sqrt(z)
        return (root - math.sin(root)) / (root ** 3)
    if z < -1.0e-8:
        root = math.sqrt(-z)
        return (math.sinh(root) - root) / (root ** 3)
    return 1.0 / 6.0


def solve_lambert_universal(
    r1_vec_km: np.ndarray,
    r2_vec_km: np.ndarray,
    tof_seconds: float,
    mu_km3_s2: float,
    long_way: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the zero-revolution, short-way Lambert problem.

    This is a two-body Sun-spacecraft transfer connecting r1 to r2 in the
    specified time. ``long_way=True`` selects the complementary transfer arc.
    Planetary perturbations are NOT included in the Lambert boundary-value
    solution. The resulting initial state can then be propagated with the same
    N-body integrator as the ML trajectory to measure its miss.
    """
    from scipy.optimize import brentq

    r1_vec = np.asarray(r1_vec_km, dtype=float).reshape(3)
    r2_vec = np.asarray(r2_vec_km, dtype=float).reshape(3)
    r1 = float(np.linalg.norm(r1_vec))
    r2 = float(np.linalg.norm(r2_vec))
    dt = float(tof_seconds)
    mu = float(mu_km3_s2)

    if r1 <= 0.0 or r2 <= 0.0 or dt <= 0.0 or mu <= 0.0:
        raise ValueError("Lambert solver requires positive radii, time, and mu.")

    cos_dtheta = float(np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(np.cross(r1_vec, r2_vec)))
    sin_dtheta = cross_norm / (r1 * r2)  # short-way angle in [0, pi]
    denominator = 1.0 - cos_dtheta
    if denominator <= 1.0e-12 or sin_dtheta <= 1.0e-12:
        raise ValueError(
            "Lambert geometry is nearly collinear; the transfer arc is ambiguous."
        )
    if long_way:
        sin_dtheta = -sin_dtheta

    A = sin_dtheta * math.sqrt(r1 * r2 / denominator)
    if abs(A) < 1.0e-12:
        raise ValueError("Lambert geometry produced a near-zero A parameter.")

    sqrt_mu = math.sqrt(mu)

    def time_equation(z: float) -> float:
        C = stumpff_c(z)
        S = stumpff_s(z)
        if not np.isfinite(C) or not np.isfinite(S) or C <= 0.0:
            return float("nan")
        y = r1 + r2 + A * (z * S - 1.0) / math.sqrt(C)
        if not np.isfinite(y) or y <= 0.0:
            return float("nan")
        return (y / C) ** 1.5 * S + A * math.sqrt(y) - sqrt_mu * dt

    # The zero-revolution root lies below 4*pi^2. Scan valid points, then use
    # a bracketed root finder rather than relying on a fragile Newton iteration.
    z_grid = np.linspace(-50.0, (2.0 * math.pi) ** 2 - 1.0e-6, 6000)
    brackets = []
    previous_z = None
    previous_f = None
    for z in z_grid:
        try:
            fz = time_equation(float(z))
        except (OverflowError, ValueError, ZeroDivisionError):
            fz = float("nan")
        if not np.isfinite(fz):
            previous_z = None
            previous_f = None
            continue
        if fz == 0.0:
            brackets.append((float(z), float(z)))
            break
        if previous_f is not None and previous_f * fz < 0.0:
            brackets.append((float(previous_z), float(z)))
        previous_z = float(z)
        previous_f = float(fz)

    if not brackets:
        raise RuntimeError(
            "Could not bracket a zero-revolution Lambert solution for these dates. "
            "Try a different time of flight or check the planetary ephemeris."
        )

    # For the short-way, zero-revolution branch, choose the bracket closest to z=0.
    z_low, z_high = min(brackets, key=lambda b: abs(0.5 * (b[0] + b[1])))
    if z_low == z_high:
        z_solution = z_low
    else:
        z_solution = float(brentq(time_equation, z_low, z_high, xtol=1.0e-10, rtol=1.0e-12))

    C = stumpff_c(z_solution)
    S = stumpff_s(z_solution)
    y = r1 + r2 + A * (z_solution * S - 1.0) / math.sqrt(C)
    f = 1.0 - y / r1
    g = A * math.sqrt(y / mu)
    g_dot = 1.0 - y / r2
    if abs(g) < 1.0e-12:
        raise RuntimeError("Lambert solution produced a near-zero Lagrange g coefficient.")

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        raise RuntimeError("Lambert solver returned non-finite velocity vectors.")
    return v1, v2


def propagate_lambert_trajectory(
    departure_days: float,
    tof_days: float,
    sample_count: int = SAMPLE_COUNT,
) -> Dict[str, Any]:
    """Create two comparable Lambert trajectories from one Lambert solution.

    1. ``positions`` / ``velocities``: Lambert two-body prediction, propagated
       under the Sun's gravity only.
    2. ``nbody_positions`` / ``nbody_velocities``: the exact same Lambert
       initial state propagated with the full physical N-body integrator.

    The second trajectory is not re-shot or corrected. The difference between
    these two paths therefore isolates the effect of using the more detailed
    gravitational model after the Lambert initial state has been selected.
    """
    from scipy.integrate import solve_ivp

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    r_earth, v_earth = physics.planet_state("earth", float(departure_days))
    r_mars, v_mars = physics.planet_state(
        "mars", float(departure_days) + float(tof_days)
    )
    r_earth = np.asarray(r_earth, dtype=float).reshape(3)
    v_earth = np.asarray(v_earth, dtype=float).reshape(3)
    r_mars = np.asarray(r_mars, dtype=float).reshape(3)
    v_mars = np.asarray(v_mars, dtype=float).reshape(3)

    dt_seconds = float(tof_days) * physics.DAY_S
    if dt_seconds <= 0.0:
        raise ValueError("Lambert transfer duration must be positive.")

    # Use the same non-singular Earth-departure position convention as the ML
    # path. Starting at Earth's centre would trip the N-body singularity guard.
    v_center_guess, _ = solve_lambert_universal(
        r_earth, r_mars, dt_seconds, physics.MU_SUN
    )
    boundary = get_boundary_guess(departure_days, tof_days)
    r0 = np.asarray(boundary["r0"], dtype=float).reshape(3)
    v0_guess = np.asarray(boundary["v0_guess"], dtype=float).reshape(3)

    earth_clearance = float(np.linalg.norm(r0 - r_earth))
    earth_soi_km = float(getattr(physics, "EARTH_SOI_KM", 0.0))
    min_start_clearance_km = max(
        10.0 * float(physics.R_EARTH), 0.5 * earth_soi_km
    )
    if earth_clearance < min_start_clearance_km:
        direction_vector = v0_guess - v_earth
        if np.linalg.norm(direction_vector) <= 1.0e-12:
            direction_vector = v_center_guess - v_earth
        direction = unit(direction_vector)
        if earth_soi_km <= 0.0:
            raise ValueError(
                "The boundary position is too close to Earth and "
                "physics.EARTH_SOI_KM is unavailable; cannot safely initialize "
                "Lambert propagation."
            )
        r0 = r_earth + direction * earth_soi_km

    # Solve the Lambert boundary-value problem for this exact starting position
    # and the Mars position at the nominal arrival epoch.
    v_lambert_depart, v_lambert_arrive = solve_lambert_universal(
        r0, r_mars, dt_seconds, physics.MU_SUN
    )

    # A. Propagate Lambert's state using only Sun gravity: the true two-body
    # Lambert trajectory for comparison.
    elapsed_seconds = np.linspace(0.0, dt_seconds, int(sample_count))

    def sun_only_rhs(_time_s: float, state: np.ndarray) -> np.ndarray:
        position = state[:3]
        velocity = state[3:]
        radius = float(np.linalg.norm(position))
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Sun-only propagation reached a singular position.")
        acceleration = -physics.MU_SUN * position / radius**3
        return np.concatenate((velocity, acceleration))

    initial_state = np.concatenate((r0, v_lambert_depart))
    sun_solution = solve_ivp(
        sun_only_rhs,
        (0.0, dt_seconds),
        initial_state,
        method="DOP853",
        t_eval=elapsed_seconds,
        rtol=3.0e-10,
        atol=1.0e-7,
        max_step=3.0 * physics.DAY_S,
    )
    if not sun_solution.success:
        raise RuntimeError(
            "Lambert two-body propagation failed: " + sun_solution.message
        )
    sun_states = np.asarray(sun_solution.y.T, dtype=float)
    lambert_positions = sun_states[:, :3]
    lambert_velocities = sun_states[:, 3:]
    times_days = np.asarray(sun_solution.t, dtype=float) / physics.DAY_S

    # B. Propagate exactly the same Lambert initial position and velocity using
    # the physical model's full N-body acceleration, without shooting correction.
    print(
        "Propagating the same Lambert initial state with physical N-body "
        "dynamics (no shooting correction)..."
    )
    nbody_positions, nbody_velocities, nbody_solution = physics.integrate_spacecraft(
        departure_days,
        tof_days,
        r0,
        v_lambert_depart,
        sample_count=sample_count,
    )
    nbody_positions = np.asarray(nbody_positions, dtype=float)
    nbody_velocities = np.asarray(nbody_velocities, dtype=float)
    if hasattr(nbody_solution, "t") and len(nbody_solution.t) == len(nbody_positions):
        nbody_times_days = np.asarray(nbody_solution.t, dtype=float) / physics.DAY_S
    else:
        nbody_times_days = np.linspace(0.0, float(tof_days), len(nbody_positions))

    # Delta-v convention for comparison: departure injection from a circular
    # Earth parking orbit, plus arrival hyperbolic-excess speed relative to Mars.
    # Arrival v-infinity is NOT a capture burn; a capture burn would require a
    # specified Mars target orbit and periapsis altitude.
    v_inf_depart = float(np.linalg.norm(v_lambert_depart - v_earth))
    v_inf_arrive_lambert = float(np.linalg.norm(lambert_velocities[-1] - v_mars))
    v_inf_arrive_nbody = float(np.linalg.norm(nbody_velocities[-1] - v_mars))

    parking_radius_km = physics.R_EARTH + physics.EARTH_PARKING_ALT_KM
    v_circular_kms = math.sqrt(physics.MU_EARTH / parking_radius_km)
    dv_departure_lambert = (
        math.sqrt(v_inf_depart**2 + 2.0 * physics.MU_EARTH / parking_radius_km)
        - v_circular_kms
    )

    return {
        # Pure two-body Lambert-predicted trajectory.
        "positions": lambert_positions,
        "velocities": lambert_velocities,
        "times_days": times_days,
        # Same initial condition, propagated under full N-body gravity.
        "nbody_positions": nbody_positions,
        "nbody_velocities": nbody_velocities,
        "nbody_times_days": nbody_times_days,
        # Lambert boundary conditions and velocity metrics.
        "r0": r0,
        "v0": v_lambert_depart,
        "v_depart_lambert": v_lambert_depart,
        "v_arrive_lambert": v_lambert_arrive,
        "v_inf_depart_kms": v_inf_depart,
        "v_inf_arrive_kms": v_inf_arrive_lambert,
        "v_inf_arrive_nbody_kms": v_inf_arrive_nbody,
        "dv_departure_kms": float(dv_departure_lambert),
        "dv_arrival_kms": v_inf_arrive_lambert,
        "dv_total_kms": float(dv_departure_lambert + v_inf_arrive_lambert),
        "dv_arrival_nbody_kms": v_inf_arrive_nbody,
        "dv_total_nbody_kms": float(dv_departure_lambert + v_inf_arrive_nbody),
        "initial_earth_clearance_km": float(np.linalg.norm(r0 - r_earth)),
    }


# ---------------------------------------------------------------------------
# 6. Planet positions, encounter distances, and TOF searches
# ---------------------------------------------------------------------------
def get_planet_positions(planet_name: str, absolute_days: np.ndarray) -> np.ndarray:
    """Return heliocentric planet positions (km) at absolute day values.

    The physical module expects lowercase planet names ("earth", "mars"), so
    normalize the name here to make this helper case-insensitive.
    """
    planet_key = str(planet_name).strip().lower()
    values = np.asarray(absolute_days, dtype=float).reshape(-1)
    return np.asarray(
        [physics.planet_state(planet_key, float(t))[0] for t in values], dtype=float
    )


def calculate_mars_distances(
    spacecraft_positions: np.ndarray,
    departure_days: float,
    elapsed_days: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate spacecraft-to-Mars-centre distances and Mars positions."""
    absolute_days = float(departure_days) + np.asarray(elapsed_days, dtype=float)
    mars_positions = get_planet_positions("Mars", absolute_days)
    distances_km = np.linalg.norm(
        np.asarray(spacecraft_positions, dtype=float) - mars_positions, axis=1
    )
    return distances_km, mars_positions


def _candidate_residual_km(candidate: Dict[str, Any]) -> float:
    try:
        value = float(candidate.get("miss_distance_km", np.inf))
        return value if np.isfinite(value) else np.inf
    except (TypeError, ValueError):
        return np.inf


def _candidate_total_dv(candidate: Dict[str, Any]) -> float:
    try:
        value = float(candidate.get("dv_total_kms", np.inf))
        return value if np.isfinite(value) else np.inf
    except (TypeError, ValueError):
        return np.inf


def _evaluate_tof_candidates(
    departure_days: float, tof_candidates: Iterable[float]
) -> list:
    """Run the physical shooting solver for a sequence of flight times."""
    evaluated = []
    seen = set()
    candidates = list(tof_candidates)

    for index, tof_days in enumerate(candidates, start=1):
        tof_days = float(tof_days)
        if round(tof_days, 8) in seen:
            continue
        seen.add(round(tof_days, 8))

        print(f"  Physical TOF search {index}/{len(candidates)}: {tof_days:.1f} days")
        t_solve = time.perf_counter()
        try:
            result = physics.solve_nbody_transfer(
                departure_days, tof_days, keep_path=False
            )
            if not isinstance(result, dict):
                print("    Skipping: solver did not return a result dictionary.")
                continue
            result = dict(result)
            result["tof_days"] = tof_days
            result["_residual_km"] = _candidate_residual_km(result)
            result["_dv_total_kms"] = _candidate_total_dv(result)
            if np.isfinite(result["_dv_total_kms"]):
                evaluated.append(result)
            else:
                print("    Skipping: solver returned an invalid total delta-v.")
        except Exception as exc:
            print(f"    Solver failed at {tof_days:.1f} days: {exc}")
        print(f"    solve time: {time.perf_counter() - t_solve:.2f} s")

    return evaluated


def optimize_transfer(
    departure_days: float,
    min_tof_days: float = MIN_TOF_DAYS,
    max_tof_days: float = MAX_TOF_DAYS,
    coarse_step_days: float = COARSE_TOF_STEP_DAYS,
    fine_step_days: float = FINE_TOF_STEP_DAYS,
) -> Dict[str, Any]:
    """Find a low-delta-v physical transfer using coarse then fine TOF search.

    PHYSICS ONLY (no ML). Slow: ~26 coarse + ~11 fine N-body shooting solves.

    Important: miss_distance_km from solve_nbody_transfer is treated as the
    shooting residual to the target point on the SOI, not the closest distance
    to Mars's centre. Actual closest approach is calculated after propagation.
    """
    if min_tof_days <= 0 or max_tof_days < min_tof_days:
        raise ValueError("Invalid TOF search range.")

    coarse = np.arange(
        min_tof_days, max_tof_days + 0.5 * coarse_step_days, coarse_step_days
    )
    print("\nSearching physical transfer (coarse TOF grid)...")
    coarse_results = _evaluate_tof_candidates(departure_days, coarse)
    if not coarse_results:
        raise RuntimeError(
            "The physical solver did not return any usable transfer candidates."
        )

    converged_coarse = [
        item
        for item in coarse_results
        if item["_residual_km"] <= SHOOTING_RESIDUAL_TOLERANCE_KM
    ]
    if converged_coarse:
        best_coarse = min(converged_coarse, key=lambda item: item["_dv_total_kms"])
    else:
        # Preserve a usable best-effort result when the solver's residual is
        # larger than the threshold, but explicitly warn about it.
        best_coarse = min(coarse_results, key=lambda item: item["_residual_km"])
        print(
            "WARNING: no coarse candidate met the shooting-residual tolerance; "
            "refining around the candidate with the smallest residual."
        )

    fine_low = max(min_tof_days, best_coarse["tof_days"] - coarse_step_days)
    fine_high = min(max_tof_days, best_coarse["tof_days"] + coarse_step_days)
    fine = np.arange(fine_low, fine_high + 0.5 * fine_step_days, fine_step_days)
    print("\nSearching physical transfer (fine TOF grid)...")
    fine_results = _evaluate_tof_candidates(departure_days, fine)
    all_results = coarse_results + fine_results

    valid = [
        item
        for item in all_results
        if item["_residual_km"] <= SHOOTING_RESIDUAL_TOLERANCE_KM
    ]
    if valid:
        best = min(valid, key=lambda item: item["_dv_total_kms"])
    else:
        best = min(all_results, key=lambda item: item["_residual_km"])
        print(
            "WARNING: no tested TOF met the shooting-residual tolerance. "
            "The selected result is only the best-residual candidate."
        )

    print(
        f"Selected physical TOF: {best['tof_days']:.1f} days | "
        f"total delta-v: {best['_dv_total_kms']:.4f} km/s | "
        f"shooting residual: {best['_residual_km']:.1f} km"
    )
    return best


def ml_guided_optimize_transfer(
    bundle: Dict[str, Any],
    departure_days: float,
    min_tof_days: float = MIN_TOF_DAYS,
    max_tof_days: float = MAX_TOF_DAYS,
    scan_step_days: float = 1.0,
    n_candidates: int = 4,
    min_separation_days: float = 15.0,
    refine_halfwidth_days: float = 6.0,
    refine_step_days: float = 2.0,
) -> Dict[str, Any]:
    """Use the ML surrogate to shortlist TOFs, then verify with physics.

    1. ML scans every TOF (1-day steps) in one batched prediction  (fast).
    2. The lowest-predicted TOFs, spaced apart, form a shortlist.
    3. The physical shooting solver verifies each shortlisted TOF.
    4. A small local physics refinement runs around the best verified TOF.

    The ML only chooses where to look; the reported answer always comes from
    the physical solver. The scan is clamped to the model's training TOF range
    because an MLP extrapolates unreliably outside the data it saw.
    """
    train_min = float(getattr(physics, "DEFAULT_TOF_MIN_DAYS", 120.0))
    train_max = float(getattr(physics, "DEFAULT_TOF_MAX_DAYS", 350.0))
    lo = max(float(min_tof_days), train_min)
    hi = min(float(max_tof_days), train_max)
    if hi <= lo:
        raise ValueError("Requested TOF range is outside the ML training range.")
    if lo > min_tof_days or hi < max_tof_days:
        print(
            f"NOTE: ML scan limited to {lo:.0f}-{hi:.0f} d "
            f"(model training range {train_min:.0f}-{train_max:.0f} d). "
            "Retrain with a wider --tof-min/--tof-max to cover more."
        )

    # 1. Fast ML scan over every TOF (one batched forward pass)
    print("\nML scan of TOF range...")
    t_scan = time.perf_counter()
    tofs = np.arange(lo, hi + 0.5 * scan_step_days, scan_step_days)
    predictions = predict_delta_v_batch(bundle, departure_days, tofs)
    totals = predictions.sum(axis=1)
    print(
        f"  Scanned {len(tofs)} TOF values in "
        f"{time.perf_counter() - t_scan:.3f} s "
        f"(ML min predicted total {totals.min():.3f} km/s "
        f"at {tofs[int(np.argmin(totals))]:.0f} d)"
    )

    # 2. Shortlist the lowest predictions, spaced apart
    picks = []
    for i in np.argsort(totals):
        t = float(tofs[i])
        if all(abs(t - p) >= min_separation_days for p in picks):
            picks.append(t)
        if len(picks) >= n_candidates:
            break
    print(f"ML shortlist (TOF days): {[round(p, 1) for p in picks]}")

    # 3. Verify the shortlist with the physical solver
    print("\nVerifying shortlist with physical N-body shooting...")
    t_verify = time.perf_counter()
    results = _evaluate_tof_candidates(departure_days, picks)
    print(f"  Verification stage time: {time.perf_counter() - t_verify:.2f} s")
    valid = [
        r for r in results if r["_residual_km"] <= SHOOTING_RESIDUAL_TOLERANCE_KM
    ]
    if not valid:
        raise RuntimeError(
            "No ML-shortlisted TOF passed the physical solver. "
            "Increase n_candidates or use the physics-only search."
        )
    best = min(valid, key=lambda r: r["_dv_total_kms"])

    # 4. Local physics refinement around the winner
    fine = np.arange(
        max(lo, best["tof_days"] - refine_halfwidth_days),
        min(hi, best["tof_days"] + refine_halfwidth_days) + 0.5 * refine_step_days,
        refine_step_days,
    )
    fine = [t for t in fine if abs(t - best["tof_days"]) > 1.0e-6]
    print("\nLocal physics refinement around best candidate...")
    t_refine = time.perf_counter()
    fine_results = _evaluate_tof_candidates(departure_days, fine)
    print(f"  Refinement stage time: {time.perf_counter() - t_refine:.2f} s")
    for r in fine_results:
        if (
            r["_residual_km"] <= SHOOTING_RESIDUAL_TOLERANCE_KM
            and r["_dv_total_kms"] < best["_dv_total_kms"]
        ):
            best = r

    print(
        f"Selected TOF (ML-guided): {best['tof_days']:.1f} days | "
        f"total delta-v: {best['_dv_total_kms']:.4f} km/s | "
        f"shooting residual: {best['_residual_km']:.1f} km"
    )
    return best


def ml_only_optimize_transfer(
    bundle: Dict[str, Any],
    departure_days: float,
    min_tof_days: float = MIN_TOF_DAYS,
    max_tof_days: float = MAX_TOF_DAYS,
    scan_step_days: float = 1.0,
    edge_margin_days: float = 10.0,
    warn_dv_error_kms: float = 0.05,
) -> Dict[str, Any]:
    """Pick the TOF with the lowest ML-predicted total delta-v, with no search.

    1. ML scans every TOF (1-day steps) in one batched prediction.
    2. The single lowest predicted total delta-v is selected directly.
    3. ONE physical solve at that TOF produces the real trajectory numbers and
       lets the script compare predicted vs actual delta-v.

    There is no shortlist and no local refinement, so nothing guards against a
    surrogate error moving the minimum. The scan is clamped to the model's
    training TOF range and trimmed by ``edge_margin_days`` at each end, where
    MLP predictions are least reliable.
    """
    train_min = float(getattr(physics, "DEFAULT_TOF_MIN_DAYS", 120.0))
    train_max = float(getattr(physics, "DEFAULT_TOF_MAX_DAYS", 350.0))
    lo = max(float(min_tof_days), train_min)
    hi = min(float(max_tof_days), train_max)
    if hi <= lo:
        raise ValueError("Requested TOF range is outside the ML training range.")
    if lo > min_tof_days or hi < max_tof_days:
        print(
            f"NOTE: ML scan limited to {lo:.0f}-{hi:.0f} d "
            f"(model training range {train_min:.0f}-{train_max:.0f} d)."
        )
    # Trim the weakest part of the training range, if the range is wide enough.
    if hi - lo > 4.0 * edge_margin_days:
        if lo <= train_min + 1.0e-9:
            lo += edge_margin_days
        if hi >= train_max - 1.0e-9:
            hi -= edge_margin_days
        print(f"ML scan range after edge trim: {lo:.0f}-{hi:.0f} d")

    # 1. ML scan over every TOF (one batched forward pass)
    print("\nML scan of TOF range...")
    t_scan = time.perf_counter()
    tofs = np.arange(lo, hi + 0.5 * scan_step_days, scan_step_days)
    predictions = predict_delta_v_batch(bundle, departure_days, tofs)
    totals = predictions.sum(axis=1)
    idx = int(np.argmin(totals))
    ml_scan_seconds = time.perf_counter() - t_scan
    best_tof = float(tofs[idx])
    pred_dep, pred_arr = float(predictions[idx, 0]), float(predictions[idx, 1])
    pred_total = float(totals[idx])
    print(
        f"  Scanned {len(tofs)} TOF values in {ml_scan_seconds:.3f} s\n"
        f"  ML-selected TOF: {best_tof:.1f} d "
        f"(predicted total delta-v {pred_total:.4f} km/s)"
    )
    if idx == 0 or idx == len(tofs) - 1:
        print(
            "WARNING: the ML minimum is at the edge of the scan range. The true "
            "minimum may lie outside it, or the prediction there is unreliable."
        )

    # 2. One physical solve at the ML-selected TOF
    print("\nPhysical solve at the ML-selected TOF...")
    t_phys = time.perf_counter()
    results = _evaluate_tof_candidates(departure_days, [best_tof])
    physics_seconds = time.perf_counter() - t_phys
    valid = [
        r for r in results if r["_residual_km"] <= SHOOTING_RESIDUAL_TOLERANCE_KM
    ]
    if not valid:
        raise RuntimeError(
            f"The physical solver did not converge at the ML-selected TOF "
            f"({best_tof:.1f} d). Use --ml-tof-search (shortlist + verify) or "
            "the physics-only search instead."
        )
    best = dict(valid[0])

    # 3. Predicted vs actual
    actual_dep = float(best.get("dv_depart_kms", float("nan")))
    actual_arr = float(best.get("dv_arrive_kms", float("nan")))
    actual_total = float(best["_dv_total_kms"])
    error_total = pred_total - actual_total
    print("\nML prediction check at the selected TOF:")
    print(f"  {'':<12}{'Predicted':>12}{'Actual':>12}{'Error':>12}")
    print(f"  {'Departure':<12}{pred_dep:>12.4f}{actual_dep:>12.4f}{pred_dep - actual_dep:>12.4f}")
    print(f"  {'Arrival':<12}{pred_arr:>12.4f}{actual_arr:>12.4f}{pred_arr - actual_arr:>12.4f}")
    print(f"  {'Total':<12}{pred_total:>12.4f}{actual_total:>12.4f}{error_total:>12.4f}  km/s")
    if abs(error_total) > warn_dv_error_kms:
        print(
            f"WARNING: predicted and actual total delta-v differ by "
            f"{abs(error_total):.3f} km/s (> {warn_dv_error_kms:.2f}). "
            "The ML pick may not be near the true optimum; consider "
            "--ml-tof-search."
        )
    else:
        print("  Prediction agrees with physics within tolerance.")

    best["_ml_predicted_total_kms"] = pred_total
    best["_ml_scan_seconds"] = ml_scan_seconds
    best["_physics_seconds"] = physics_seconds
    print(
        f"Selected TOF (ML-only): {best['tof_days']:.1f} days | "
        f"total delta-v: {best['_dv_total_kms']:.4f} km/s | "
        f"shooting residual: {best['_residual_km']:.1f} km"
    )
    return best


# ---------------------------------------------------------------------------
# 7. Initial-guess performance benchmark helpers
# ---------------------------------------------------------------------------
def make_ml_shooting_initial_velocity(
    departure_days: float, tof_days: float, dv_departure_ml: float,
    boundary: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an ML-derived velocity guess at the same Earth-SOI start point.
    The ML model predicts the parking-orbit departure burn. Convert it to
    hyperbolic excess speed, then to the local speed at the spacecraft's
    Earth-SOI starting position using energy conservation. This is for the
    shooting optimizer; the original uncorrected ML trajectory remains intact.
    """
    if boundary is None:
        boundary = physics.make_boundary_guess(departure_days, tof_days)
    r0 = np.asarray(boundary["r0"], dtype=float).reshape(3)
    r_earth = np.asarray(boundary["r_earth"], dtype=float).reshape(3)
    v_earth = np.asarray(boundary["v_earth"], dtype=float).reshape(3)
    v0_guess = np.asarray(boundary["v0_guess"], dtype=float).reshape(3)
    direction = unit(v0_guess - v_earth)

    parking_radius_km = physics.R_EARTH + physics.EARTH_PARKING_ALT_KM
    v_circular_kms = math.sqrt(physics.MU_EARTH / parking_radius_km)
    dv = max(0.0, float(dv_departure_ml))
    v_inf_sq = max(0.0, (v_circular_kms + dv) ** 2 - 2.0 * physics.MU_EARTH / parking_radius_km)
    earth_distance_km = float(np.linalg.norm(r0 - r_earth))
    local_speed_kms = math.sqrt(v_inf_sq + 2.0 * physics.MU_EARTH / earth_distance_km)
    return r0, v_earth + direction * local_speed_kms


def solve_nbody_transfer_with_seed(
    departure_days: float,
    tof_days: float,
    initial_velocity: np.ndarray,
    *,
    keep_path: bool = True,
    miss_tolerance_km: float = 100.0,
    max_nfev: int = 35,
) -> Dict[str, Any]:
    """Call the seeded solver, with compatibility for an older physics module.

    Some existing local copies of mars_nbody_ml.py define
    solve_nbody_transfer() without the ``initial_velocity`` keyword. If such a
    copy is imported, this function performs the same shooting procedure here
    using its existing boundary-state builder and N-body integrator. That avoids
    a confusing ``unexpected keyword argument`` failure and keeps the old
    physical model usable without requiring other project features to change.
    """
    seed = np.asarray(initial_velocity, dtype=float).reshape(3)
    if not np.all(np.isfinite(seed)):
        raise ValueError("Initial velocity guess must contain three finite values.")

    # Use the physical-module implementation whenever it supports seeded runs.
    try:
        signature = inspect.signature(physics.solve_nbody_transfer)
        supports_seed = "initial_velocity" in signature.parameters
    except (TypeError, ValueError):
        supports_seed = False
    if supports_seed:
        return physics.solve_nbody_transfer(
            departure_days,
            tof_days,
            keep_path=keep_path,
            miss_tolerance_km=miss_tolerance_km,
            max_nfev=max_nfev,
            initial_velocity=seed,
        )

    # Backward-compatible implementation for the older function signature.
    if float(tof_days) <= 0.0:
        raise ValueError("Time of flight must be positive.")
    boundary = physics.make_boundary_guess(float(departure_days), float(tof_days))
    r0 = np.asarray(boundary["r0"], dtype=float).reshape(3)
    target = np.asarray(boundary["r_target"], dtype=float).reshape(3)
    evaluations = {"n": 0}

    def residual(v0: np.ndarray) -> np.ndarray:
        evaluations["n"] += 1
        try:
            positions, _, _ = physics.integrate_spacecraft(
                float(departure_days), float(tof_days), r0, v0
            )
            return (positions[-1] - target) / 100_000.0
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            return np.full(3, 1.0e3)

    correction = least_squares(
        residual,
        seed,
        method="lm",
        diff_step=2.0e-5,
        ftol=2.0e-9,
        xtol=2.0e-9,
        gtol=2.0e-9,
        max_nfev=max_nfev,
    )
    v0 = np.asarray(correction.x, dtype=float)
    positions, velocities, solution = physics.integrate_spacecraft(
        float(departure_days), float(tof_days), r0, v0,
        sample_count=500 if keep_path else 1,
    )
    r_end = np.asarray(positions[-1], dtype=float)
    v_end = np.asarray(velocities[-1], dtype=float)
    miss = float(np.linalg.norm(r_end - target))
    if not np.isfinite(miss) or miss > miss_tolerance_km:
        raise RuntimeError(
            "N-body shooting did not converge: "
            f"miss={miss:.1f} km, evaluations={evaluations['n']}, "
            f"status={correction.message}"
        )

    r_earth = np.asarray(boundary["r_earth"], dtype=float)
    v_earth = np.asarray(boundary["v_earth"], dtype=float)
    r_mars = np.asarray(boundary["r_mars"], dtype=float)
    v_mars = np.asarray(boundary["v_mars"], dtype=float)
    earth_relative_r = r0 - r_earth
    earth_relative_v = v0 - v_earth
    mars_relative_r = r_end - r_mars
    mars_relative_v = v_end - v_mars

    if float(np.dot(earth_relative_r, earth_relative_v)) <= 0.0:
        raise RuntimeError("Spacecraft is not moving outward through Earth's SOI")
    if float(np.dot(mars_relative_r, mars_relative_v)) >= 0.0:
        raise RuntimeError("Spacecraft is not moving inward through Mars's SOI")
    earth_energy = float(
        np.dot(earth_relative_v, earth_relative_v)
        - 2.0 * physics.MU_EARTH / np.linalg.norm(earth_relative_r)
    )
    mars_energy = float(
        np.dot(mars_relative_v, mars_relative_v)
        - 2.0 * physics.MU_MARS / np.linalg.norm(mars_relative_r)
    )
    if earth_energy <= 0.0 or mars_energy <= 0.0:
        raise RuntimeError("Boundary state is not hyperbolic relative to Earth or Mars")

    v_inf_earth = math.sqrt(earth_energy)
    v_inf_mars = math.sqrt(mars_energy)
    dv_depart = physics.parking_orbit_burn(
        v_inf_earth, physics.MU_EARTH, physics.R_EARTH,
        physics.EARTH_PARKING_ALT_KM,
    )
    dv_arrive = physics.parking_orbit_burn(
        v_inf_mars, physics.MU_MARS, physics.R_MARS,
        physics.MARS_PARKING_ALT_KM,
    )
    result: Dict[str, Any] = {
        "departure_days": float(departure_days),
        "departure_date": physics.days_to_date(float(departure_days)),
        "arrival_date": physics.days_to_date(float(departure_days) + float(tof_days)),
        "tof_days": float(tof_days),
        "dv_depart_kms": float(dv_depart),
        "dv_arrive_kms": float(dv_arrive),
        "dv_total_kms": float(dv_depart + dv_arrive),
        "v_inf_earth_kms": float(v_inf_earth),
        "v_inf_mars_kms": float(v_inf_mars),
        "miss_distance_km": miss,
        "shooting_nfev": int(evaluations["n"]),
        "optimizer_success": bool(correction.success),
        "optimizer_status": int(correction.status),
        "initial_velocity_guess_kms": seed,
        "r0_km": r0,
        "v0_kms": v0,
        "r_target_km": target,
        "r_end_km": r_end,
        "v_end_kms": v_end,
        "r_earth_depart_km": r_earth,
        "v_earth_depart_kms": v_earth,
        "r_mars_arrive_km": r_mars,
        "v_mars_arrive_kms": v_mars,
    }
    if keep_path:
        result.update({
            "path_r_km": np.asarray(positions, dtype=float),
            "path_v_kms": np.asarray(velocities, dtype=float),
            "path_times_days": np.asarray(solution.t, dtype=float) / physics.DAY_S,
        })
    return result


def propagate_seed_for_plot(
    departure_days: float,
    tof_days: float,
    initial_velocity: np.ndarray,
    sample_count: int = SAMPLE_COUNT,
) -> Dict[str, np.ndarray]:
    """Propagate a benchmark's exact initial velocity without shooting correction.

    These diagnostic propagations are created for trajectory visualization and
    CSV export. Their time is deliberately NOT included in the seed-versus-
    shooting benchmark, which measures seed construction plus optimization.
    """
    boundary = physics.make_boundary_guess(float(departure_days), float(tof_days))
    r0 = np.asarray(boundary["r0"], dtype=float).reshape(3)
    v0 = np.asarray(initial_velocity, dtype=float).reshape(3)
    positions, velocities, solution = physics.integrate_spacecraft(
        float(departure_days), float(tof_days), r0, v0,
        sample_count=max(2, int(sample_count)),
    )
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if hasattr(solution, "t") and len(solution.t) == len(positions):
        times_days = np.asarray(solution.t, dtype=float) / physics.DAY_S
    else:
        times_days = np.linspace(0.0, float(tof_days), len(positions))
    return {
        "positions": positions,
        "velocities": velocities,
        "times_days": times_days,
        "r0": r0,
        "v0": v0,
    }


def benchmark_corrected_initial_guesses(
    bundle: Dict[str, Any], departure_days: float, tof_days: float
) -> Dict[str, Any]:
    """Compare Lambert- and ML-seeded N-body shooting using the same solver.

    Both methods use the same r0, SOI target, gravity model, tolerances and
    maximum function evaluations. Only the initial velocity supplied to the
    physical shooting optimizer differs.
    """
    benchmark: Dict[str, Any] = {}
    boundary = physics.make_boundary_guess(departure_days, tof_days)
    r0 = np.asarray(boundary["r0"], dtype=float).reshape(3)
    target = np.asarray(boundary["r_target"], dtype=float).reshape(3)
    dt_seconds = float(tof_days) * physics.DAY_S

    # Lambert seed: solve the Sun-only two-body boundary-value problem to the
    # same Mars-SOI target used by the physical shooting solver.
    t0 = time.perf_counter()
    v_lambert, _ = solve_lambert_universal(r0, target, dt_seconds, physics.MU_SUN)
    earth_relative_guess = v_lambert - np.asarray(boundary["v_earth"], dtype=float)
    earth_soi_offset = r0 - np.asarray(boundary["r_earth"], dtype=float)
    lambert_arc = "short-way"
    # An Earth-to-Mars Lambert solution must leave the Earth SOI, not enter it.
    # If the short-way branch points inward, try the complementary long-way arc.
    if float(np.dot(earth_soi_offset, earth_relative_guess)) <= 0.0:
        v_lambert, _ = solve_lambert_universal(
            r0, target, dt_seconds, physics.MU_SUN, long_way=True
        )
        lambert_arc = "long-way"
    if float(np.dot(earth_soi_offset, v_lambert - np.asarray(boundary["v_earth"], dtype=float))) <= 0.0:
        raise RuntimeError(
            "Neither Lambert arc gives an outward Earth-SOI departure for this "
            "date/TOF. Try a different transfer time or departure date."
        )
    lambert_guess_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        lambert_result = solve_nbody_transfer_with_seed(
            departure_days, tof_days, initial_velocity=v_lambert, keep_path=True
        )
        lambert_correction_seconds = time.perf_counter() - t0
        benchmark["lambert_seeded"] = {
            "result": lambert_result,
            "guess_seconds": lambert_guess_seconds,
            "correction_seconds": lambert_correction_seconds,
            "total_seconds": lambert_guess_seconds + lambert_correction_seconds,
            "guess_velocity_kms": np.asarray(v_lambert, dtype=float),
            "lambert_arc": lambert_arc,
            "success": True,
            "error": "",
        }
    except Exception as exc:
        lambert_correction_seconds = time.perf_counter() - t0
        benchmark["lambert_seeded"] = {
            "result": None,
            "guess_seconds": lambert_guess_seconds,
            "correction_seconds": lambert_correction_seconds,
            "total_seconds": lambert_guess_seconds + lambert_correction_seconds,
            "guess_velocity_kms": np.asarray(v_lambert, dtype=float),
            "lambert_arc": lambert_arc,
            "success": False,
            "error": str(exc),
        }

    # ML seed: predict delta-v, convert its departure magnitude to v-infinity,
    # and use the existing boundary/Lambert direction convention.
    t0 = time.perf_counter()
    dv_departure_ml, dv_arrival_ml = predict_delta_v(bundle, departure_days, tof_days)
    ml_r0, v_ml = make_ml_shooting_initial_velocity(
        departure_days, tof_days, dv_departure_ml, boundary=boundary
    )
    ml_guess_seconds = time.perf_counter() - t0
    if not np.allclose(ml_r0, r0, rtol=0.0, atol=1.0e-5):
        raise RuntimeError("ML and Lambert initial states do not share the same departure position.")
    t0 = time.perf_counter()
    try:
        ml_result = solve_nbody_transfer_with_seed(
            departure_days, tof_days, initial_velocity=v_ml, keep_path=True
        )
        ml_correction_seconds = time.perf_counter() - t0
        benchmark["ml_seeded"] = {
            "result": ml_result,
            "guess_seconds": ml_guess_seconds,
            "correction_seconds": ml_correction_seconds,
            "total_seconds": ml_guess_seconds + ml_correction_seconds,
            "guess_velocity_kms": np.asarray(v_ml, dtype=float),
            "success": True,
            "error": "",
            "dv_departure_ml": float(dv_departure_ml),
            "dv_arrival_ml": float(dv_arrival_ml),
        }
    except Exception as exc:
        ml_correction_seconds = time.perf_counter() - t0
        benchmark["ml_seeded"] = {
            "result": None,
            "guess_seconds": ml_guess_seconds,
            "correction_seconds": ml_correction_seconds,
            "total_seconds": ml_guess_seconds + ml_correction_seconds,
            "guess_velocity_kms": np.asarray(v_ml, dtype=float),
            "success": False,
            "error": str(exc),
            "dv_departure_ml": float(dv_departure_ml),
            "dv_arrival_ml": float(dv_arrival_ml),
        }
    return benchmark


def print_performance_summary(benchmark: Dict[str, Any]) -> None:
    """Print a compact time, iteration and convergence comparison."""
    print("\n" + "=" * 100)
    print("LAMBERT-SEEDED PHYSICAL MODEL vs ML-SEEDED PHYSICAL MODEL")
    print("=" * 100)
    print(f"{'Metric':<34}{'Lambert + N-body shooting':>30}{'ML + N-body shooting':>30}")
    print("-" * 94)
    for key, label in (("guess_seconds", "Initial guess time (s)"),
                       ("correction_seconds", "N-body correction time (s)"),
                       ("total_seconds", "Total elapsed time (s)")):
        vals = []
        for method in ("lambert_seeded", "ml_seeded"):
            item = benchmark[method]
            vals.append(f"{item[key]:.6f}" if item.get("success") else "FAILED")
        print(f"{label:<34}{vals[0]:>30}{vals[1]:>30}")
    for method, label in (("lambert_seeded", "Lambert + N-body shooting"),
                          ("ml_seeded", "ML + N-body shooting")):
        item = benchmark[method]
        result = item.get("result")
        if result is None:
            print(f"\n{label}: FAILED — {item.get('error', 'unknown error')}")
        else:
            print(f"\n{label}:")
            if method == "lambert_seeded":
                print(f"  Lambert branch: {item.get('lambert_arc', 'n/a')}")
            print(f"  Optimizer trajectory evaluations: {result.get('shooting_nfev', 'n/a')}")
            print(f"  Final SOI-target miss: {result.get('miss_distance_km', float('nan')):,.3f} km")
            print(f"  Total delta-v: {result.get('dv_total_kms', float('nan')):.5f} km/s")


def save_performance_csv(benchmark: Dict[str, Any], path: str) -> None:
    """Write the measured timings and convergence outcomes to CSV."""
    fields = ["method", "success", "initial_guess_seconds", "correction_seconds",
              "total_seconds", "optimizer_evaluations", "miss_distance_km",
              "dv_total_kms", "error"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method, label in (("lambert_seeded", "Lambert + physical shooting"),
                              ("ml_seeded", "ML + physical shooting")):
            item = benchmark[method]
            result = item.get("result") or {}
            writer.writerow({
                "method": label,
                "success": item.get("success", False),
                "initial_guess_seconds": item.get("guess_seconds", float("nan")),
                "correction_seconds": item.get("correction_seconds", float("nan")),
                "total_seconds": item.get("total_seconds", float("nan")),
                "optimizer_evaluations": result.get("shooting_nfev", ""),
                "miss_distance_km": result.get("miss_distance_km", ""),
                "dv_total_kms": result.get("dv_total_kms", ""),
                "error": item.get("error", ""),
            })
    print(f"Saved solver performance CSV: {os.path.abspath(path)}")


def plot_performance_comparison(benchmark: Dict[str, Any], path: str, show_plot: bool = True) -> None:
    """Plot initial-guess, correction, and total computation times."""
    labels = ["Lambert + physical", "ML + physical"]
    methods = [benchmark["lambert_seeded"], benchmark["ml_seeded"]]
    x = np.arange(2)
    guess = [m["guess_seconds"] for m in methods]
    correction = [m["correction_seconds"] if m.get("success") else 0.0 for m in methods]
    totals = [m["total_seconds"] for m in methods]
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.24
    ax.bar(x - width, guess, width, label="Initial guess")
    ax.bar(x, correction, width, label="N-body shooting correction")
    ax.bar(x + width, totals, width, label="Total")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Elapsed time (seconds)")
    ax.set_title("Computation-Time Comparison: Lambert vs ML Initial Guess")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved computation-time plot: {os.path.abspath(path)}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7b. Run the physical, Lambert, and ML trajectories at the same departure/TOF
# ---------------------------------------------------------------------------
def run_comparison(
    bundle: Dict[str, Any],
    departure_days: float,
    tof_days: float,
    sample_count: int = SAMPLE_COUNT,
) -> Dict[str, Any]:
    """Run physical shooting, ML, Lambert two-body, and Lambert+N-body paths."""
    print(f"\nPredicting ML delta-v for TOF = {tof_days:.1f} days...")
    dv_departure_ml, dv_arrival_ml = predict_delta_v(bundle, departure_days, tof_days)
    print(f"ML predicted departure delta-v: {dv_departure_ml:.4f} km/s")
    print(f"ML predicted arrival delta-v:   {dv_arrival_ml:.4f} km/s")

    print("\nBenchmarking Lambert-seeded and ML-seeded physical shooting...")
    performance = benchmark_corrected_initial_guesses(bundle, departure_days, tof_days)
    print_performance_summary(performance)

    # Plot the exact same raw velocity seeds that were supplied to the two
    # optimizers. These are distinct from the separate Lambert trajectory that
    # targets Mars's centre in the legacy two-body-vs-N-body comparison.
    seed_paths: Dict[str, Optional[Dict[str, np.ndarray]]] = {
        "lambert_seed_uncorrected": None,
        "ml_seed_uncorrected": None,
    }
    for method_key, result_key in (
        ("lambert_seeded", "lambert_seed_uncorrected"),
        ("ml_seeded", "ml_seed_uncorrected"),
    ):
        item = performance.get(method_key, {})
        velocity = item.get("guess_velocity_kms")
        if velocity is None:
            continue
        try:
            seed_paths[result_key] = propagate_seed_for_plot(
                departure_days, tof_days, velocity, sample_count=sample_count
            )
        except Exception as exc:
            print(f"Warning: could not propagate {method_key} initial guess for plotting: {exc}")

    print("\nPropagating physical N-body shooting solution...")
    physical = physics.solve_nbody_transfer(departure_days, tof_days, keep_path=True)
    if not isinstance(physical, dict):
        raise TypeError("solve_nbody_transfer() did not return a dictionary.")
    for key in ("path_r_km", "path_times_days"):
        if key not in physical:
            raise KeyError(f"Physical result is missing required field {key!r}.")

    physical_positions = np.asarray(physical["path_r_km"], dtype=float)
    physical_times = np.asarray(physical["path_times_days"], dtype=float)
    if physical_positions.ndim != 2 or physical_positions.shape[1] != 3:
        raise ValueError(f"Unexpected physical path_r_km shape: {physical_positions.shape}")

    print("Propagating uncorrected ML initial state with N-body dynamics...")
    ml = propagate_ml_trajectory(
        departure_days, tof_days, dv_departure_ml, sample_count=sample_count
    )

    print("\nSolving Lambert two-body transfer and comparing its physical propagation...")
    lambert_all = propagate_lambert_trajectory(
        departure_days, tof_days, sample_count=sample_count
    )
    lambert = {
        "positions": lambert_all["positions"],
        "velocities": lambert_all["velocities"],
        "times_days": lambert_all["times_days"],
        "r0": lambert_all["r0"],
        "v0": lambert_all["v0"],
        "v_depart_lambert": lambert_all["v_depart_lambert"],
        "v_arrive_lambert": lambert_all["v_arrive_lambert"],
        "v_inf_depart_kms": lambert_all["v_inf_depart_kms"],
        "v_inf_arrive_kms": lambert_all["v_inf_arrive_kms"],
        "dv_departure_kms": lambert_all["dv_departure_kms"],
        "dv_arrival_kms": lambert_all["dv_arrival_kms"],
        "dv_total_kms": lambert_all["dv_total_kms"],
    }
    lambert_physical = {
        "positions": lambert_all["nbody_positions"],
        "velocities": lambert_all["nbody_velocities"],
        "times_days": lambert_all["nbody_times_days"],
        "r0": lambert_all["r0"],
        "v0": lambert_all["v0"],
        "v_inf_arrive_kms": lambert_all["v_inf_arrive_nbody_kms"],
        "dv_departure_kms": lambert_all["dv_departure_kms"],
        "dv_arrival_kms": lambert_all["dv_arrival_nbody_kms"],
        "dv_total_kms": lambert_all["dv_total_nbody_kms"],
    }

    physical_distances, physical_mars_positions = calculate_mars_distances(
        physical_positions, departure_days, physical_times
    )
    ml_distances, ml_mars_positions = calculate_mars_distances(
        ml["positions"], departure_days, ml["times_days"]
    )
    lambert_distances, lambert_mars_positions = calculate_mars_distances(
        lambert["positions"], departure_days, lambert["times_days"]
    )
    lambert_physical_distances, lambert_physical_mars_positions = calculate_mars_distances(
        lambert_physical["positions"], departure_days, lambert_physical["times_days"]
    )

    p_idx = int(np.argmin(physical_distances))
    m_idx = int(np.argmin(ml_distances))
    l_idx = int(np.argmin(lambert_distances))
    lp_idx = int(np.argmin(lambert_physical_distances))

    results = {
        "departure_days": float(departure_days),
        "tof_days": float(tof_days),
        "physical": physical,
        "ml": ml,
        "lambert": lambert,
        "lambert_physical": lambert_physical,
        "physical_mars_distances": physical_distances,
        "ml_mars_distances": ml_distances,
        "lambert_mars_distances": lambert_distances,
        "lambert_physical_mars_distances": lambert_physical_distances,
        "physical_mars_positions": physical_mars_positions,
        "ml_mars_positions": ml_mars_positions,
        "lambert_mars_positions": lambert_mars_positions,
        "lambert_physical_mars_positions": lambert_physical_mars_positions,
        "physical_closest_km": float(physical_distances[p_idx]),
        "ml_closest_km": float(ml_distances[m_idx]),
        "lambert_closest_km": float(lambert_distances[l_idx]),
        "lambert_physical_closest_km": float(lambert_physical_distances[lp_idx]),
        "physical_closest_day": float(physical_times[p_idx]),
        "ml_closest_day": float(ml["times_days"][m_idx]),
        "lambert_closest_day": float(lambert["times_days"][l_idx]),
        "lambert_physical_closest_day": float(lambert_physical["times_days"][lp_idx]),
        "dv_departure_ml": float(dv_departure_ml),
        "dv_arrival_ml": float(dv_arrival_ml),
        "dv_departure_lambert": float(lambert["dv_departure_kms"]),
        "dv_arrival_lambert": float(lambert["dv_arrival_kms"]),
        "dv_total_lambert": float(lambert["dv_total_kms"]),
        "dv_departure_lambert_physical": float(lambert_physical["dv_departure_kms"]),
        "dv_arrival_lambert_physical": float(lambert_physical["dv_arrival_kms"]),
        "dv_total_lambert_physical": float(lambert_physical["dv_total_kms"]),
        "performance": performance,
        "lambert_seed_uncorrected": seed_paths["lambert_seed_uncorrected"],
        "ml_seed_uncorrected": seed_paths["ml_seed_uncorrected"],
    }
    print(f"Lambert departure v-infinity: {lambert['v_inf_depart_kms']:.4f} km/s")
    print(f"Lambert two-body arrival v-infinity: {lambert['v_inf_arrive_kms']:.4f} km/s")
    print(
        "Lambert + N-body arrival v-infinity: "
        f"{lambert_physical['v_inf_arrive_kms']:.4f} km/s"
    )
    return results

def predict_delta_v_batch(
    bundle: Dict[str, Any], departure_days: float, tof_values: np.ndarray
) -> np.ndarray:
    """Predict (departure, arrival) delta-v for many TOFs at ONE departure date."""
    tof_values = np.asarray(tof_values, dtype=float).reshape(-1)
    return predict_delta_v_pairs(
        bundle, [(float(departure_days), float(t)) for t in tof_values]
    )


def predict_delta_v_pairs(
    bundle: Dict[str, Any], pairs: Iterable[Tuple[float, float]]
) -> np.ndarray:
    """Predict (departure, arrival) delta-v for many (departure_days, tof_days) pairs."""
    rows = []
    for departure_days, tof_days in pairs:
        feature_values = physics.trajectory_features(
            float(departure_days), float(tof_days)
        )
        if isinstance(feature_values, dict):
            names = bundle.get("feature_names")
            if names is not None:
                rows.append([feature_values[n] for n in names])
            else:
                rows.append(list(feature_values.values()))
        else:
            rows.append(np.asarray(feature_values, dtype=float).reshape(-1))
    x = np.asarray(rows, dtype=float)

    x_scaler = bundle.get("x_scaler")
    if x_scaler is not None:
        x = x_scaler.transform(x)
    y_scaled = np.asarray(bundle["model"].predict(x), dtype=float)
    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)
    y_scaler = bundle.get("y_scaler")
    y = y_scaler.inverse_transform(y_scaled) if y_scaler is not None else y_scaled
    y = np.asarray(y, dtype=float)
    if y.shape[1] < 2:
        raise ValueError("The trained model must predict two delta-v values.")
    return np.maximum(y[:, :2], 0.0)


# =============================================================================
# 2. Launch Window Optimization & Trajectory Plotting
# =============================================================================

def optimize_launch_window(
    bundle: Dict[str, Any],
    start_date: str,
    end_date: str,
    tof_min: float,
    tof_max: float,
    n_departure: int = 90,
    n_tof: int = 55,
    verify_candidates: int = 8,
) -> Dict[str, Any]:
    """Find the departure date and TOF with the lowest total delta-v."""
    train_min = float(getattr(physics, "DEFAULT_TOF_MIN_DAYS", 120.0))
    train_max = float(getattr(physics, "DEFAULT_TOF_MAX_DAYS", 350.0))
    tof_lo = max(float(tof_min), train_min)
    tof_hi = min(float(tof_max), train_max)
    if tof_hi <= tof_lo:
        raise ValueError("Requested TOF range is outside the ML training range.")
    if tof_lo > tof_min or tof_hi < tof_max:
        print(
            f"NOTE: ML scan limited to {tof_lo:.0f}-{tof_hi:.0f} d "
            f"(model training range {train_min:.0f}-{train_max:.0f} d)."
        )

    dep_lo = date_to_absolute_days(start_date)
    dep_hi = date_to_absolute_days(end_date)
    if dep_hi <= dep_lo:
        raise ValueError("The window end date must be after the start date.")

    # 1. ML scan over the whole (departure, TOF) grid
    dep_grid = np.linspace(dep_lo, dep_hi, n_departure)
    tof_grid = np.linspace(tof_lo, tof_hi, n_tof)
    pairs = [(float(d), float(t)) for d in dep_grid for t in tof_grid]
    print(f"\nML scan of {len(pairs):,} (departure, TOF) pairs...")
    t_scan = time.perf_counter()
    predictions = predict_delta_v_pairs(bundle, pairs)
    totals = predictions.sum(axis=1)
    best_idx = int(np.argmin(totals))
    print(
        f"  Scan time: {time.perf_counter() - t_scan:.2f} s | ML minimum "
        f"{totals[best_idx]:.3f} km/s at {physics.days_to_date(pairs[best_idx][0])}, "
        f"TOF {pairs[best_idx][1]:.0f} d"
    )

    # 2. Shortlist lowest predictions
    shortlist = []
    for idx in np.argsort(totals):
        d, t = pairs[int(idx)]
        if all(abs(d - d0) > 35.0 or abs(t - t0) > 12.0 for d0, t0 in shortlist):
            shortlist.append((d, t))
        if len(shortlist) >= verify_candidates:
            break

    # 3. Verify shortlist with N-body shooting solver
    print(f"\nChecking {len(shortlist)} ML-shortlisted candidates with N-body shooting...")
    accepted = []
    for i, (d, t) in enumerate(shortlist, start=1):
        try:
            phys = physics.solve_nbody_transfer(d, t)
            accepted.append(phys)
            print(
                f"  {i}/{len(shortlist)} {phys['departure_date']} / {t:.1f} d: "
                f"total delta-v = {phys['dv_total_kms']:.3f} km/s, "
                f"miss = {phys['miss_distance_km']:.2f} km"
            )
        except Exception as exc:
            print(f"  {i}/{len(shortlist)} candidate rejected by physics solver: {exc}")
    if not accepted:
        raise RuntimeError(
            "None of the ML-shortlisted trajectories passed the N-body solver. "
            "Try more --window-candidates or a different window."
        )
    best = min(accepted, key=lambda p: p["dv_total_kms"])

    # 4. Local 3 x 3 physics refinement around the winner
    dep_step = (dep_hi - dep_lo) / max(n_departure - 1, 1)
    tof_step = (tof_hi - tof_lo) / max(n_tof - 1, 1)
    print("\nLocal physics refinement around the best candidate...")
    for d in np.linspace(max(dep_lo, best["departure_days"] - dep_step),
                         min(dep_hi, best["departure_days"] + dep_step), 3):
        for t in np.linspace(max(tof_lo, best["tof_days"] - tof_step),
                             min(tof_hi, best["tof_days"] + tof_step), 3):
            try:
                phys = physics.solve_nbody_transfer(float(d), float(t))
                if phys["dv_total_kms"] < best["dv_total_kms"]:
                    best = phys
            except Exception:
                pass

    print("\n--- Best validated candidate ---")
    print(f"Departure date:             {best['departure_date']}")
    print(f"Arrival date:               {best['arrival_date']}")
    print(f"Time of flight:             {best['tof_days']:.2f} days")
    print(f"Earth departure burn:       {best['dv_depart_kms']:.3f} km/s")
    print(f"Mars capture burn estimate: {best['dv_arrive_kms']:.3f} km/s")
    print(f"Total idealized delta-v:    {best['dv_total_kms']:.3f} km/s")
    print(f"Arrival SOI target miss:    {best['miss_distance_km']:.3f} km")
    return best


def save_best_candidate_csv(
    best: Dict[str, Any], path: str = "mars_nbody_best_candidate.csv"
) -> None:
    """Save the key numbers of the optimum transfer to a one-row CSV."""
    columns = (
        "departure_date", "arrival_date", "tof_days", "dv_depart_kms",
        "dv_arrive_kms", "dv_total_kms", "v_inf_earth_kms", "v_inf_mars_kms",
        "miss_distance_km",
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({k: best[k] for k in columns})
    print(f"Saved best-candidate CSV: {os.path.abspath(path)}")


def plot_optimum_trajectory(
    best: Dict[str, Any],
    save_path: str = "mars_nbody_trajectory.png",
    show_plot: bool = True,
) -> None:
    """Plot the N-body trajectory of the optimum transfer with planet orbits."""
    p = physics.solve_nbody_transfer(
        best["departure_days"], best["tof_days"], keep_path=True
    )
    sc = np.asarray(p["path_r_km"], dtype=float) / physics.AU_KM
    days = np.linspace(
        best["departure_days"] - 220.0,
        best["departure_days"] + best["tof_days"] + 120.0,
        600,
    )
    earth = get_planet_positions("earth", days) / physics.AU_KM
    mars = get_planet_positions("mars", days) / physics.AU_KM
    earth_dep = np.asarray(p["r_earth_depart_km"], dtype=float) / physics.AU_KM
    mars_arr = np.asarray(p["r_mars_arrive_km"], dtype=float) / physics.AU_KM

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(earth[:, 0], earth[:, 1], "b--", alpha=0.6, label="Earth elliptical orbit")
    ax.plot(mars[:, 0], mars[:, 1], "r--", alpha=0.6, label="Mars elliptical orbit")
    ax.plot(sc[:, 0], sc[:, 1], color="green", linewidth=2,
            label="Spacecraft path: Sun + Earth + Mars gravity")
    ax.scatter([0], [0], marker="*", s=150, label="Sun")
    ax.scatter([earth_dep[0]], [earth_dep[1]], s=45, label="Earth departure SOI")
    ax.scatter([mars_arr[0]], [mars_arr[1]], s=45, label="Mars arrival SOI")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Ecliptic X (AU)")
    ax.set_ylabel("Ecliptic Y (AU)")
    ax.set_title(
        f"Optimum Earth-Mars transfer | {physics.days_to_date(best['departure_days'])} "
        f"| TOF {best['tof_days']:.1f} d"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    print(f"Saved optimum trajectory plot: {os.path.abspath(save_path)}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_optimum_window(args: argparse.Namespace) -> Dict[str, Any]:
    """Run option 4: find the best departure date and TOF, then plot the path."""
    print("=" * 72)
    print("OPTIMUM LAUNCH WINDOW: BEST DEPARTURE DATE AND BEST TOF")
    print("=" * 72)
    print(f"Window:      {args.window_start} to {args.window_end}")
    print(f"TOF range:   {args.window_tof_min:.0f}-{args.window_tof_max:.0f} d")
    print(f"ML model:    {os.path.abspath(args.model_file)}")
    print("Note: --departure-date is ignored in this mode.")

    bundle = load_model_bundle(args.model_file)
    start = time.perf_counter()
    best = optimize_launch_window(
        bundle,
        args.window_start,
        args.window_end,
        args.window_tof_min,
        args.window_tof_max,
        verify_candidates=args.window_candidates,
    )
    print(f"\nTotal optimum-window search time: {time.perf_counter() - start:.2f} s")

    save_best_candidate_csv(best, args.best_csv)
    plot_optimum_trajectory(best, args.window_plot_file, show_plot=not args.no_show)
    return best


# ---------------------------------------------------------------------------
# 8. Plot comparisons in two separate figures
# ---------------------------------------------------------------------------
def _plot_pair_comparison(
    results: Dict[str, Any],
    comparison_model: str,
    save_path: str,
    show_plot: bool = True,
) -> None:
    """Keep the same two-panel layout for ML and Lambert comparisons.

    Figure 1: physical N-body shooting trajectory vs ML initial state.
    Figure 2: Lambert two-body prediction vs its own N-body propagation.
    """
    if comparison_model not in ("ml", "lambert"):
        raise ValueError("comparison_model must be either 'ml' or 'lambert'.")

    departure_days = float(results["departure_days"])
    tof_days = float(results["tof_days"])

    if comparison_model == "ml":
        reference = results["physical"]
        reference_times = np.asarray(reference["path_times_days"], dtype=float)
        reference_positions = np.asarray(reference["path_r_km"], dtype=float)
        reference_distances = np.asarray(results["physical_mars_distances"], dtype=float)
        reference_mars_positions = np.asarray(results["physical_mars_positions"], dtype=float)
        compared = results["ml"]
        compared_distances = np.asarray(results["ml_mars_distances"], dtype=float)
        compared_mars_positions = np.asarray(results["ml_mars_positions"], dtype=float)
        compared_label = "ML trajectory (uncorrected)"
        figure_title = "Physical N-body vs ML Trajectory"
        reference_label = "Physical N-body trajectory"
        reference_short_label = "Physical N-body"
        close_label = "ML closest approach"
        reference_close_label = "Physical closest approach"
        compared_marker = "+"
        compared_marker_size = 130
        compared_distance_label = "ML trajectory"
        reference_marker_label = "Physical closest approach"
        reference_mars_close_label = "Mars at physical closest approach"
    else:
        reference = results["lambert"]
        reference_times = np.asarray(reference["times_days"], dtype=float)
        reference_positions = np.asarray(reference["positions"], dtype=float)
        reference_distances = np.asarray(results["lambert_mars_distances"], dtype=float)
        reference_mars_positions = np.asarray(results["lambert_mars_positions"], dtype=float)
        compared = results["lambert_physical"]
        compared_distances = np.asarray(results["lambert_physical_mars_distances"], dtype=float)
        compared_mars_positions = np.asarray(results["lambert_physical_mars_positions"], dtype=float)
        compared_label = "Same Lambert state, N-body propagated"
        figure_title = "Lambert Two-Body vs Lambert + N-body Propagation"
        reference_label = "Lambert two-body trajectory"
        reference_short_label = "Lambert two-body"
        close_label = "Lambert + N-body closest approach"
        reference_close_label = "Lambert two-body closest approach"
        compared_marker = "^"
        compared_marker_size = 100
        compared_distance_label = "Lambert state + N-body gravity"
        reference_marker_label = "Lambert two-body closest approach"
        reference_mars_close_label = "Mars at Lambert two-body closest approach"

    compared_positions = np.asarray(compared["positions"], dtype=float)
    compared_times = np.asarray(compared["times_days"], dtype=float)

    # On the Lambert figure, also show the exact Lambert initial velocity used
    # by the N-body shooting benchmark, propagated without correction. This is
    # distinct from the original Lambert solution targeting Mars's centre.
    extra_lambert_times = None
    extra_lambert_positions = None
    extra_lambert_distances = None
    if comparison_model == "lambert":
        extra_seed = results.get("lambert_seed_uncorrected")
        if isinstance(extra_seed, dict):
            candidate_times = np.asarray(extra_seed.get("times_days", []), dtype=float).reshape(-1)
            candidate_positions = np.asarray(extra_seed.get("positions", []), dtype=float)
            if (candidate_times.size >= 2
                    and candidate_positions.shape == (candidate_times.size, 3)
                    and np.all(np.isfinite(candidate_times))
                    and np.all(np.isfinite(candidate_positions))):
                extra_lambert_times = candidate_times
                extra_lambert_positions = candidate_positions
                extra_lambert_distances, _ = calculate_mars_distances(
                    extra_lambert_positions, departure_days, extra_lambert_times
                )

    reference_idx = int(np.argmin(reference_distances))
    compared_idx = int(np.argmin(compared_distances))
    extra_lambert_idx = (
        int(np.argmin(extra_lambert_distances))
        if extra_lambert_distances is not None else None
    )

    # Planet orbits through a window spanning the transfer, with the same
    # extent, styling, and layout as the original plot.
    planet_times = np.linspace(
        departure_days - 100.0, departure_days + tof_days + 100.0, 800
    )
    earth_path = get_planet_positions("earth", planet_times)
    mars_path = get_planet_positions("mars", planet_times)
    earth_au = earth_path / physics.AU_KM
    mars_au = mars_path / physics.AU_KM

    mars_departure_km = get_planet_positions("mars", np.array([departure_days]))[0]
    mars_arrival_km = get_planet_positions(
        "mars", np.array([departure_days + tof_days])
    )[0]
    mars_departure_au = mars_departure_km / physics.AU_KM
    mars_arrival_au = mars_arrival_km / physics.AU_KM
    reference_mars_au = reference_mars_positions / physics.AU_KM
    compared_mars_au = compared_mars_positions / physics.AU_KM

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    ax = axes[0]
    ax.plot(earth_au[:, 0], earth_au[:, 1], "--", label="Earth orbit", alpha=0.65)
    ax.plot(mars_au[:, 0], mars_au[:, 1], "--", label="Mars orbit", alpha=0.65)
    # Use different line styles for Lambert's two uncorrected propagations so
    # they remain distinguishable even when the heliocentric paths nearly overlap.
    if comparison_model == "lambert":
        reference_style, compared_style = "--", ":"
    else:
        reference_style, compared_style = "-", "-"
    ax.plot(
        reference_positions[:, 0] / physics.AU_KM,
        reference_positions[:, 1] / physics.AU_KM,
        linewidth=2.5,
        linestyle=reference_style,
        label=reference_label,
    )
    ax.plot(
        compared_positions[:, 0] / physics.AU_KM,
        compared_positions[:, 1] / physics.AU_KM,
        linewidth=2.5,
        linestyle=compared_style,
        label=compared_label,
    )
    extra_lambert_line = None
    if extra_lambert_positions is not None:
        extra_lambert_line, = ax.plot(
            extra_lambert_positions[:, 0] / physics.AU_KM,
            extra_lambert_positions[:, 1] / physics.AU_KM,
            linewidth=2.2,
            linestyle="-.",
            label="Exact Lambert shooting seed (N-body, uncorrected)",
        )
    ax.scatter([0.0], [0.0], marker="*", s=200, label="Sun", zorder=7)

    earth_departure_km = np.asarray(
        physics.planet_state("earth", float(departure_days))[0], dtype=float
    )
    earth_departure_au = earth_departure_km / physics.AU_KM
    ax.scatter(
        earth_departure_au[0], earth_departure_au[1], marker="o", s=85,
        label="Earth at departure", zorder=7,
    )
    ax.scatter(
        mars_departure_au[0], mars_departure_au[1], marker="o", s=110,
        label="Mars at departure", zorder=8,
    )
    ax.scatter(
        mars_arrival_au[0], mars_arrival_au[1], marker="X", s=150,
        label="Mars at arrival", zorder=9,
    )
    ax.plot(
        [mars_departure_au[0], mars_arrival_au[0]],
        [mars_departure_au[1], mars_arrival_au[1]],
        ":", linewidth=1.8, label="Mars displacement", alpha=0.9,
    )

    marker_stride = max(1, len(reference_mars_au) // 12)
    ax.scatter(
        reference_mars_au[::marker_stride, 0],
        reference_mars_au[::marker_stride, 1],
        marker=".", s=35, label="Mars positions during transfer", zorder=5,
    )
    ax.scatter(
        reference_mars_au[reference_idx, 0],
        reference_mars_au[reference_idx, 1],
        marker="D", s=85, label=reference_mars_close_label, zorder=10,
    )
    ax.scatter(
        reference_positions[reference_idx, 0] / physics.AU_KM,
        reference_positions[reference_idx, 1] / physics.AU_KM,
        marker="x", s=100, label=reference_marker_label, zorder=10,
    )
    ax.scatter(
        compared_positions[compared_idx, 0] / physics.AU_KM,
        compared_positions[compared_idx, 1] / physics.AU_KM,
        marker=compared_marker, s=compared_marker_size,
        label=close_label, zorder=10,
    )
    ax.annotate(
        "Mars at departure", (mars_departure_au[0], mars_departure_au[1]),
        xytext=(8, 8), textcoords="offset points", fontsize=8,
    )
    ax.annotate(
        "Mars at arrival", (mars_arrival_au[0], mars_arrival_au[1]),
        xytext=(8, 8), textcoords="offset points", fontsize=8,
    )
    ax.set_xlabel("Heliocentric X (AU)")
    ax.set_ylabel("Heliocentric Y (AU)")
    ax.set_title(figure_title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax2 = axes[1]
    mars_soi_km = standard_mars_soi_km()
    # The Lambert two-body path is designed to reach Mars's nominal position,
    # so its distance can approach zero. Use a logarithmic axis and a 1 km
    # display floor to make both the close approach and any N-body miss visible.
    if comparison_model == "lambert":
        distance_scale = 1.0
        distance_floor_km = 1.0
        distance_unit_label = "km (log scale)"
    else:
        distance_scale = 1000.0
        distance_floor_km = 1.0
        distance_unit_label = "thousand km"

    # Log-log axes require strictly positive x and y values. Exclude the
    # departure sample at day 0 rather than artificially shifting its time.
    reference_mask = np.isfinite(reference_times) & (reference_times > 0)
    compared_mask = np.isfinite(compared_times) & (compared_times > 0)
    ax2.plot(
        reference_times[reference_mask],
        np.maximum(reference_distances[reference_mask], distance_floor_km) / distance_scale,
        linewidth=2.2, label=reference_short_label,
    )
    ax2.plot(
        compared_times[compared_mask],
        np.maximum(compared_distances[compared_mask], distance_floor_km) / distance_scale,
        linewidth=2.2, label=compared_distance_label,
    )
    if extra_lambert_distances is not None:
        extra_mask = np.isfinite(extra_lambert_times) & (extra_lambert_times > 0)
        ax2.plot(
            extra_lambert_times[extra_mask],
            np.maximum(extra_lambert_distances[extra_mask], distance_floor_km) / distance_scale,
            linewidth=2.2,
            linestyle="-.",
            color=extra_lambert_line.get_color() if extra_lambert_line is not None else None,
            label="Exact Lambert shooting seed (N-body, uncorrected)",
        )
    ax2.axhline(
        mars_soi_km / distance_scale, linestyle="--",
        label=(f"Standard Mars SOI ({mars_soi_km:,.0f} km)"
               if comparison_model == "lambert"
               else f"Standard Mars SOI ({mars_soi_km / 1000.0:,.0f} thousand km)"),
    )
    ax2.scatter(
        reference_times[reference_idx],
        max(float(reference_distances[reference_idx]), distance_floor_km) / distance_scale,
        marker="x", s=90, label=reference_close_label, zorder=5,
    )
    ax2.scatter(
        compared_times[compared_idx],
        max(float(compared_distances[compared_idx]), distance_floor_km) / distance_scale,
        marker=compared_marker, s=compared_marker_size,
        label=close_label, zorder=5,
    )
    if extra_lambert_idx is not None:
        ax2.scatter(
            extra_lambert_times[extra_lambert_idx],
            max(float(extra_lambert_distances[extra_lambert_idx]), distance_floor_km) / distance_scale,
            marker="s", s=65,
            color=extra_lambert_line.get_color() if extra_lambert_line is not None else None,
            label="Exact Lambert seed closest approach", zorder=6,
        )
    ax2.set_xlabel("Days after departure")
    ax2.set_ylabel(f"Spacecraft-to-Mars distance ({distance_unit_label})")
    ax2.set_title("Distance Between Spacecraft and Mars (Log-Log)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(
        f"Earth-to-Mars Transfer | {physics.days_to_date(departure_days)} | "
        f"TOF = {tof_days:.1f} days",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved {comparison_model.upper()} comparison plot: {os.path.abspath(save_path)}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    reference_closest_km = float(reference_distances[reference_idx])
    compared_closest_km = float(compared_distances[compared_idx])
    print(
        f"\n{reference_short_label} closest approach: {reference_closest_km:,.0f} km "
        f"at day {reference_times[reference_idx]:.2f}"
    )
    print(
        f"{compared_label} closest approach: {compared_closest_km:,.0f} km "
        f"at day {compared_times[compared_idx]:.2f}"
    )
    print(f"Standard Mars SOI radius: {mars_soi_km:,.0f} km")
    print(f"{reference_short_label} enters standard Mars SOI: {reference_closest_km <= mars_soi_km}")
    print(f"{compared_label} enters standard Mars SOI: {compared_closest_km <= mars_soi_km}")


def plot_comparison(
    results: Dict[str, Any],
    save_path: str = "mars_trajectory_comparison.png",
    show_plot: bool = True,
) -> None:
    """Original plot: physical N-body trajectory versus the ML trajectory."""
    _plot_pair_comparison(results, "ml", save_path, show_plot)


def plot_lambert_comparison(
    results: Dict[str, Any],
    save_path: str = "mars_lambert_comparison.png",
    show_plot: bool = True,
) -> None:
    """Separate plot: Lambert two-body prediction vs its N-body propagation."""
    _plot_pair_comparison(results, "lambert", save_path, show_plot)


# ---------------------------------------------------------------------------
# 9. Save all trajectories to a comparison CSV
# ---------------------------------------------------------------------------
def _interpolate_columns(
    old_times: np.ndarray, old_values: np.ndarray, new_times: np.ndarray
) -> np.ndarray:
    """Linearly interpolate each column onto a common time grid."""
    old_times = np.asarray(old_times, dtype=float).reshape(-1)
    old_values = np.asarray(old_values, dtype=float)
    if old_values.ndim == 1:
        old_values = old_values.reshape(-1, 1)
    # np.interp expects ascending abscissae.
    order = np.argsort(old_times)
    old_times = old_times[order]
    old_values = old_values[order]
    return np.column_stack(
        [np.interp(new_times, old_times, old_values[:, col])
         for col in range(old_values.shape[1])]
    )


def save_trajectory_csv(
    results: Dict[str, Any], save_path: str = "mars_trajectory_comparison.csv"
) -> None:
    """Save physical, ML, Lambert two-body, and Lambert+N-body paths."""
    trajectories = {
        "physical": (
            np.asarray(results["physical"]["path_times_days"], dtype=float),
            np.asarray(results["physical"]["path_r_km"], dtype=float),
            np.asarray(results["physical_mars_positions"], dtype=float),
        ),
        "ml": (
            np.asarray(results["ml"]["times_days"], dtype=float),
            np.asarray(results["ml"]["positions"], dtype=float),
            np.asarray(results["ml_mars_positions"], dtype=float),
        ),
        "lambert_two_body": (
            np.asarray(results["lambert"]["times_days"], dtype=float),
            np.asarray(results["lambert"]["positions"], dtype=float),
            np.asarray(results["lambert_mars_positions"], dtype=float),
        ),
        "lambert_nbody": (
            np.asarray(results["lambert_physical"]["times_days"], dtype=float),
            np.asarray(results["lambert_physical"]["positions"], dtype=float),
            np.asarray(results["lambert_physical_mars_positions"], dtype=float),
        ),
    }
    # Export the exact uncorrected seeds used by the shooting benchmark, when
    # their diagnostic N-body propagations were successful.
    for result_key, csv_name in (
        ("lambert_seed_uncorrected", "lambert_seed_uncorrected_nbody"),
        ("ml_seed_uncorrected", "ml_seed_uncorrected_nbody"),
    ):
        seed_path = results.get(result_key)
        if isinstance(seed_path, dict):
            seed_times = np.asarray(seed_path["times_days"], dtype=float)
            seed_positions = np.asarray(seed_path["positions"], dtype=float)
            seed_mars = get_planet_positions(
                "mars", float(results["departure_days"]) + seed_times
            )
            trajectories[csv_name] = (seed_times, seed_positions, seed_mars)

    # Include both corrected trajectories in the exported CSV whenever
    # their shooting optimizations converged.
    performance = results.get("performance", {})
    for method_key, csv_name in (("lambert_seeded", "lambert_seeded_nbody"),
                                 ("ml_seeded", "ml_seeded_nbody")):
        item = performance.get(method_key, {})
        result = item.get("result")
        if item.get("success") and isinstance(result, dict) and "path_r_km" in result:
            corrected_times = np.asarray(result["path_times_days"], dtype=float)
            corrected_positions = np.asarray(result["path_r_km"], dtype=float)
            corrected_mars_positions = get_planet_positions(
                "mars", float(results["departure_days"]) + corrected_times
            )
            trajectories[csv_name] = (corrected_times, corrected_positions, corrected_mars_positions)
    tof_days = float(results["tof_days"])
    n = max(*(len(item[0]) for item in trajectories.values()), SAMPLE_COUNT)
    common_t = np.linspace(0.0, tof_days, n)

    interpolated = {}
    for name, (times, positions, mars_positions) in trajectories.items():
        r_i = _interpolate_columns(times, positions, common_t)
        mars_i = _interpolate_columns(times, mars_positions, common_t)
        d_i = np.linalg.norm(r_i - mars_i, axis=1)
        interpolated[name] = (r_i, mars_i, d_i)

    header = ["elapsed_days"]
    for name in trajectories:
        header += [
            f"{name}_x_km", f"{name}_y_km", f"{name}_z_km",
            f"{name}_mars_x_km", f"{name}_mars_y_km", f"{name}_mars_z_km",
            f"{name}_distance_to_mars_km",
        ]

    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, elapsed in enumerate(common_t):
            row = [elapsed]
            for name in trajectories:
                r_i, mars_i, d_i = interpolated[name]
                row += list(r_i[i]) + list(mars_i[i]) + [d_i[i]]
            writer.writerow(row)
    print(f"Saved trajectory comparison CSV to: {os.path.abspath(save_path)}")


def print_delta_v_summary(results: Dict[str, Any]) -> None:
    """Print departure, arrival, and total delta-v terms for each approach."""
    physical = results["physical"]
    ml_depart = float(results["dv_departure_ml"])
    ml_arrive = float(results["dv_arrival_ml"])
    ml_total = ml_depart + ml_arrive

    physical_depart = physical.get("dv_depart_kms", physical.get("dv_departure_kms"))
    physical_arrive = physical.get("dv_arrive_kms", physical.get("dv_arrival_kms"))
    physical_total = physical.get("dv_total_kms", physical.get("dv_total_km_s"))
    if physical_total is None and physical_depart is not None and physical_arrive is not None:
        physical_total = float(physical_depart) + float(physical_arrive)

    print("\n" + "=" * 78)
    print("DELTA-V COMPARISON")
    print("=" * 78)
    print("Note: Lambert arrival term below is arrival v-infinity, not a capture burn.")
    print("A Mars capture burn requires a specified target orbit and periapsis altitude.\n")

    def row(name: str, departure: Any, arrival: Any, total: Any) -> None:
        def fmt(value: Any) -> str:
            try:
                f = float(value)
                return f"{f:10.4f}" if np.isfinite(f) else "     n/a  "
            except (TypeError, ValueError):
                return "     n/a  "
        print(f"{name:<36} {fmt(departure)} {fmt(arrival)} {fmt(total)}")

    print(f"{'Method':<36} {'Departure (km/s)':>16} {'Arrival term (km/s)':>20} {'Total (km/s)':>14}")
    print("-" * 92)
    row("Physical N-body shooting solver", physical_depart, physical_arrive, physical_total)
    row("ML-predicted delta-v", ml_depart, ml_arrive, ml_total)
    row(
        "Lambert two-body prediction",
        results["dv_departure_lambert"],
        results["dv_arrival_lambert"],
        results["dv_total_lambert"],
    )
    row(
        "Lambert initial state + N-body",
        results["dv_departure_lambert_physical"],
        results["dv_arrival_lambert_physical"],
        results["dv_total_lambert_physical"],
    )
    print("\nThe two Lambert rows use the same departure burn; their arrival term differs because")
    print("one is Sun-only Lambert propagation and the other includes full N-body gravity.")


# ---------------------------------------------------------------------------
# 9b. Interactive choice of TOF search method
# ---------------------------------------------------------------------------
_TOF_METHOD_ALIASES = {
    "1": "physics", "physics": "physics", "p": "physics",
    "2": "ml-guided", "ml-guided": "ml-guided", "guided": "ml-guided", "g": "ml-guided",
    "3": "ml-only", "ml-only": "ml-only", "only": "ml-only", "o": "ml-only",
}


def prompt_tof_search_method() -> str:
    """Ask the user which TOF search method to run.

    Returns "physics", "ml-guided" or "ml-only". Pressing Enter selects the
    physics-only search.

    The prompt is always attempted. Some IDE consoles report stdin as "not a
    terminal" even though they accept typed input, so isatty() is NOT used to
    decide. Only if reading input actually fails (no stdin, closed pipe) does
    the function fall back to the physics-only search and explain how to pick
    another method with a command-line flag.
    """
    print("\n" + "=" * 72)
    print("SELECT TOF SEARCH METHOD")
    print("=" * 72)
    print("  1) Physics-only grid    coarse + fine N-body solves (~37)")
    print("                          slowest, most reliable")
    print("  2) ML-guided            ML scan, then physics verifies a short list (~9 solves)")
    print("                          faster, verified against several TOFs")
    print("  3) ML-only              lowest ML-predicted delta-v, 1 physics solve")
    print("                          fastest, pick is not verified against other TOFs")
    print("=" * 72, flush=True)
    while True:
        try:
            choice = input("Enter 1, 2 or 3 [default 1]: ").strip().lower()
        except (EOFError, OSError, RuntimeError):
            print(
                "\nCould not read input (no usable stdin); using the physics-only "
                "TOF search.\nPass --ml-tof-search or --ml-only-tof to choose "
                "another method without the prompt."
            )
            return "physics"
        if choice == "":
            return "physics"
        if choice in _TOF_METHOD_ALIASES:
            return _TOF_METHOD_ALIASES[choice]
        print(f"  '{choice}' is not a valid choice. Please enter 1, 2 or 3.")


# ---------------------------------------------------------------------------
# 10. Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Earth-Mars ML, Lambert, and N-body trajectories; optimize TOF or launch window."
    )
    parser.add_argument("--model-file", default=MODEL_FILE, help="Path to the trained joblib model bundle.")
    parser.add_argument(
        "--departure-date", default=DEFAULT_DEPARTURE_DATE,
        help="Departure date for a single-date search (YYYY-MM-DD)."
    )
    parser.add_argument("--tof-days", type=float, default=None, help="Use a fixed TOF instead of searching.")
    parser.add_argument("--physics-tof-search", action="store_true", help="Physics-only coarse/fine TOF search.")
    parser.add_argument("--ml-tof-search", action="store_true", help="ML shortlist followed by physics verification.")
    parser.add_argument("--ml-only-tof", action="store_true", help="Use the ML minimum and verify it with one physics solve.")
    parser.add_argument(
        "--optimum-window", action="store_true",
        help="Optimize BOTH departure date and TOF over a date window."
    )
    parser.add_argument("--min-tof", type=float, default=MIN_TOF_DAYS, help="Minimum TOF in days.")
    parser.add_argument("--max-tof", type=float, default=MAX_TOF_DAYS, help="Maximum TOF in days.")
    parser.add_argument("--ml-candidates", type=int, default=4, help="ML-shortlisted TOFs to verify.")
    parser.add_argument("--window-start", default="2026-01-01", help="Launch-window start date (inclusive).")
    parser.add_argument("--window-end", default="2026-12-31", help="Launch-window end date.")
    parser.add_argument("--window-tof-min", type=float, default=120.0, help="Minimum TOF for launch-window optimization.")
    parser.add_argument("--window-tof-max", type=float, default=350.0, help="Maximum TOF for launch-window optimization.")
    parser.add_argument("--window-departure-samples", type=int, default=90, help="Coarse departure-date samples.")
    parser.add_argument("--window-tof-samples", type=int, default=55, help="Coarse TOF samples.")
    parser.add_argument("--window-candidates", type=int, default=8, help="Top ML launch candidates to verify.")
    parser.add_argument("--best-csv", default="mars_nbody_best_candidate.csv", help="Optimum-window result CSV.")
    parser.add_argument("--window-plot-file", default="mars_nbody_trajectory.png", help="Optimum-window trajectory plot.")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT, help="Trajectory plot sample count.")
    parser.add_argument("--plot-file", default="mars_trajectory_comparison.png", help="Physical-vs-ML plot.")
    parser.add_argument("--lambert-plot-file", default="mars_lambert_comparison.png", help="Lambert comparison plot.")
    parser.add_argument("--csv-file", default="mars_trajectory_comparison.csv", help="Trajectory data CSV.")
    parser.add_argument("--performance-csv", default="mars_solver_performance.csv", help="Solver timing CSV.")
    parser.add_argument("--performance-plot", default="mars_solver_performance.png", help="Solver timing plot.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening GUI windows.")
    args = parser.parse_args()

    modes = [args.ml_only_tof, args.ml_tof_search, args.physics_tof_search, args.optimum_window]
    if sum(modes) > 1:
        parser.error("Choose only one TOF/launch-window optimization mode.")
    if args.optimum_window and args.tof_days is not None:
        parser.error("--tof-days cannot be combined with --optimum-window.")
    if args.min_tof <= 0 or args.max_tof < args.min_tof:
        parser.error("Require 0 < --min-tof <= --max-tof.")
    if args.sample_count < 2:
        parser.error("--sample-count must be at least 2.")

    # The launch-window option optimizes departure date and TOF together.
    # It returns after saving its own result CSV and trajectory plot.
    if args.optimum_window:
        bundle = load_model_bundle(args.model_file)
        start = time.perf_counter()
        best = optimize_launch_window(
            bundle,
            args.window_start,
            args.window_end,
            args.window_tof_min,
            args.window_tof_max,
            n_departure=args.window_departure_samples,
            n_tof=args.window_tof_samples,
            verify_candidates=args.window_candidates,
        )
        elapsed = time.perf_counter() - start
        print(f"\nLaunch-window search time: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")
        save_best_candidate_csv(best, args.best_csv)
        plot_optimum_trajectory(best, args.window_plot_file, show_plot=not args.no_show)
        return

    # Ask for a method only when the user has not selected one and hasn't
    # supplied a fixed TOF.
    tof_method = None
    if args.tof_days is None:
        if args.ml_only_tof:
            tof_method = "ml-only"
        elif args.ml_tof_search:
            tof_method = "ml-guided"
        elif args.physics_tof_search:
            tof_method = "physics"
        else:
            tof_method = prompt_tof_search_method()
            if tof_method == "optimum-window":
                bundle = load_model_bundle(args.model_file)
                start = time.perf_counter()
                best = optimize_launch_window(
                    bundle, args.window_start, args.window_end,
                    args.window_tof_min, args.window_tof_max,
                    n_departure=args.window_departure_samples,
                    n_tof=args.window_tof_samples,
                    verify_candidates=args.window_candidates,
                )
                print(f"\nLaunch-window search time: {time.perf_counter() - start:.2f} s")
                save_best_candidate_csv(best, args.best_csv)
                plot_optimum_trajectory(best, args.window_plot_file, show_plot=not args.no_show)
                return

    departure_days = date_to_absolute_days(args.departure_date)
    mars_soi_km = standard_mars_soi_km()
    print("=" * 72)
    print("EARTH-TO-MARS ML SURROGATE TEST")
    print("=" * 72)
    print(f"Departure date: {args.departure_date}")
    print(f"Departure absolute days since J2000: {departure_days:.3f}")
    print(f"Standard Mars SOI radius: {mars_soi_km:,.0f} km")
    print(f"ML model file: {os.path.abspath(args.model_file)}")
    print(f"Physical module loaded from: {os.path.abspath(physics.__file__)}")
    try:
        signature = inspect.signature(physics.solve_nbody_transfer)
        print(f"Physical solver signature: {signature}")
        if "initial_velocity" not in signature.parameters:
            print("Compatibility mode: seeded shooting adapter available for older physical modules.")
    except (TypeError, ValueError):
        print("Physical solver signature could not be inspected.")

    bundle = load_model_bundle(args.model_file)
    search_seconds = None
    csv_label = None
    if args.tof_days is None:
        labels = {
            "ml-only": ("ML-only TOF search", "TOF search (ML-only)"),
            "ml-guided": ("ML-guided TOF search", "TOF search (ML-guided)"),
            "physics": ("physics-only coarse/fine TOF search", "TOF search (physics-only)"),
        }
        search_label, csv_label = labels[tof_method]
        print(f"\nStarting {search_label}...")
        started = time.perf_counter()
        if tof_method == "ml-only":
            best = ml_only_optimize_transfer(bundle, departure_days, args.min_tof, args.max_tof)
        elif tof_method == "ml-guided":
            best = ml_guided_optimize_transfer(
                bundle, departure_days, args.min_tof, args.max_tof,
                n_candidates=args.ml_candidates,
            )
        else:
            best = optimize_transfer(departure_days, args.min_tof, args.max_tof)
        search_seconds = time.perf_counter() - started
        tof_days = float(best["tof_days"])
        print("\n" + "=" * 72)
        print("TOF SEARCH RESULT")
        print("=" * 72)
        print(f"Selected TOF:      {tof_days:.1f} days")
        print(f"Total delta-v:     {best['_dv_total_kms']:.4f} km/s")
        print(f"Search time:       {search_seconds:.2f} s ({search_seconds / 60.0:.2f} min)")
    else:
        tof_days = float(args.tof_days)
        if tof_days <= 0:
            parser.error("--tof-days must be greater than zero.")

    results = run_comparison(bundle, departure_days, tof_days, sample_count=args.sample_count)
    print_delta_v_summary(results)
    physical = results["physical"]
    if "miss_distance_km" in physical:
        print(f"\nPhysical shooting residual to SOI target: {physical['miss_distance_km']:,.1f} km")

    plot_comparison(results, save_path=args.plot_file, show_plot=not args.no_show)
    plot_lambert_comparison(results, save_path=args.lambert_plot_file, show_plot=not args.no_show)
    save_trajectory_csv(results, save_path=args.csv_file)
    save_performance_csv(results["performance"], args.performance_csv)
    if search_seconds is not None:
        with open(args.performance_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([csv_label, "", "", "", search_seconds, "", "", "", ""])
        print(f"Appended TOF search time to: {os.path.abspath(args.performance_csv)}")
    plot_performance_comparison(
        results["performance"], args.performance_plot, show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()