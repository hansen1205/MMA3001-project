"""Earth-to-Mars ML surrogate with Sun-Earth-Mars trajectory correction.

This module holds the PHYSICAL MODEL and the TRAINING pipeline only:
  * ephemeris, Lambert solver, N-body integrator, shooting solver
  * dataset generation, MLP training, validation plot, porkchop plot

The search for the optimum departure date / TOF and the plot of the optimum
trajectory now live in the test script (--optimum-window).

Educational/research model, not flight-quality mission design.
Install: pip install numpy scipy pandas scikit-learn matplotlib joblib
Quick test: python mars_nbody_ml.py --samples 30 --skip-plots
Typical:    python mars_nbody_ml.py --samples 500
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

MU_SUN = 1.32712440018e11
MU_EARTH = 398600.435507
MU_MARS = 42828.375214
R_EARTH = 6378.1366
R_MARS = 3396.19
AU_KM = 149597870.7
DAY_S = 86400.0
safetyfactor= 0.7

J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
EARTH_PARKING_ALT_KM = 200.
MARS_PARKING_ALT_KM = 250.

EARTH_SOI_KM = AU_KM * (MU_EARTH / MU_SUN) ** (2/5) 
MARS_SOI_KM = 1.523679 * AU_KM * (MU_MARS / MU_SUN) ** (2/5) * safetyfactor

DEFAULT_START_DATE = '2026-01-01'
DEFAULT_END_DATE = '2040-01-01'
DEFAULT_TOF_MIN_DAYS = 120.
DEFAULT_TOF_MAX_DAYS = 350.

PLANET_ELEMENTS = {
    'earth': (
        1.00000261, .00000562, .01671123, -.00004392, -.00001531, -.01294668,
        100.46457166, 35999.37244981, 102.93768193, .32327364, 0., 0.
    ),
    'mars': (
        1.52371034, .00001847, .09339410, .00007882, 1.84969142, -.00813131,
        -4.55343205, 19140.30268499, -23.94362959, .44441088, 49.55953891, -.29257343
    )
}


# ---------------------------------------------------------------------------
# 1. Dates, ephemeris and Lambert solver
# ---------------------------------------------------------------------------
def date_to_days(date_text: str) -> float:
    """Convert an ISO date string into days elapsed since J2000."""
    s = date_text.strip()
    if len(s) == 10: 
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        if dt.tzinfo is None: 
            dt = dt.replace(tzinfo=timezone.utc)
        else: 
            dt = dt.astimezone(timezone.utc)
    return (dt - J2000).total_seconds() / DAY_S


def days_to_date(days: float) -> str: 
    """Format a J2000-relative day value as a calendar date."""
    return (J2000 + timedelta(days=float(days))).strftime('%Y-%m-%d')


def unit(vec):
    """Return a normalized vector; reject vectors with negligible length."""
    n = float(np.linalg.norm(vec))
    if n < 1e-14: 
        raise ValueError('Cannot normalize a zero-length vector')
    return np.asarray(vec, dtype=float) / n


def planet_state(planet: str, days_since_j2000: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute heliocentric position and velocity from approximate orbital elements."""
    if planet not in PLANET_ELEMENTS: 
        raise ValueError(f'Unsupported planet: {planet}')
    
    T = days_since_j2000 / 36525.
    a0, adot, e0, edot, i0, idot, L0, Ldot, varpi0, varpidot, node0, nodedot = PLANET_ELEMENTS[planet]
    
    a_au = a0 + adot * T
    e = e0 + edot * T
    inc = math.radians(i0 + idot * T)
    L = math.radians(L0 + Ldot * T)
    varpi = math.radians(varpi0 + varpidot * T)
    node = math.radians(node0 + nodedot * T)
    
    arg_peri = varpi - node
    M = (L - varpi) % (2 * math.pi)
    if M > math.pi: 
        M -= 2 * math.pi
    
    E = M + e * math.sin(M)
    for _ in range(15):
        step = (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
        E -= step
        if abs(step) < 1e-13: 
            break
            
    a = a_au * AU_KM
    root = math.sqrt(1 - e * e)
    x = a * (math.cos(E) - e)
    y = a * root * math.sin(E)
    n = math.sqrt(MU_SUN / a**3)
    Edot = n / (1 - e * math.cos(E))
    vx = -a * math.sin(E) * Edot
    vy = a * root * math.cos(E) * Edot
    
    cw, sw = math.cos(arg_peri), math.sin(arg_peri)
    cO, sO = math.cos(node), math.sin(node)
    ci, si = math.cos(inc), math.sin(inc)
    
    rot = np.array([
        [cw * cO - sw * sO * ci, -sw * cO - cw * sO * ci, 0],
        [cw * sO + sw * cO * ci, -sw * sO + cw * cO * ci, 0],
        [sw * si, cw * si, 0]
    ])
    return rot @ np.array([x, y, 0.]), rot @ np.array([vx, vy, 0.])


def stumpff_c(z):
    """Evaluate the Stumpff C function for universal-variable propagation."""
    if z > 1e-8: 
        return (1 - math.cos(math.sqrt(z))) / z
    if z < -1e-8:
        q = math.sqrt(-z)
        if q > 50: 
            return float('inf')
        return (1 - math.cosh(q)) / z
    return .5 - z / 24 + z * z / 720 - z**3 / 40320


def stumpff_s(z):
    """Evaluate the Stumpff S function for universal-variable propagation."""
    if z > 1e-8:
        q = math.sqrt(z)
        return (q - math.sin(q)) / q**3
    if z < -1e-8:
        q = math.sqrt(-z)
        if q > 50: 
            return float('inf')
        return (math.sinh(q) - q) / q**3
    return 1 / 6 - z / 120 + z * z / 5040 - z**3 / 362880


def lambert_universal(r1, r2, tof_days, mu=MU_SUN):
    """Solve Lambert’s transfer for the initial and final velocity vectors."""
    tof = float(tof_days) * DAY_S
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))
    
    if tof <= 0 or r1n <= 0 or r2n <= 0: 
        raise ValueError('TOF and position magnitudes must be positive')
        
    cosd = float(np.clip(np.dot(r1, r2) / (r1n * r2n), -1, 1))
    crossz = float(np.cross(r1, r2)[2])
    sinabs = math.sqrt(max(0, 1 - cosd * cosd))
    sind = sinabs if crossz >= 0 else -sinabs
    
    if 1 - cosd < 1e-12 or abs(sind) < 1e-10: 
        raise ValueError('Degenerate Lambert geometry')
        
    A = sind * math.sqrt(r1n * r2n / (1 - cosd))
    
    def residual(z):
        """Return the current solver mismatch for the trial variable."""
        C, S = stumpff_c(z), stumpff_s(z)
        if not math.isfinite(C) or not math.isfinite(S) or C <= 0: 
            return float('nan')
        y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
        if y <= 0 or not math.isfinite(y): 
            return float('nan')
        x = math.sqrt(y / C)
        t = (x**3 * S + A * math.sqrt(y)) / math.sqrt(mu)
        return t - tof if math.isfinite(t) else float('nan')
        
    grid = np.linspace(-20, (2 * math.pi)**2 - 1e-5, 260)
    prevz = prevf = bracket = None
    
    for z in grid:
        f = residual(float(z))
        if not math.isfinite(f): 
            prevz = prevf = None
            continue
        if abs(f) < 1e-8: 
            bracket = (float(z), float(z))
            break
        if prevf is not None and prevf * f < 0: 
            bracket = (prevz, float(z))
            break
        prevz, prevf = float(z), f
        
    if bracket is None: 
        raise ValueError('Could not bracket Lambert solution for this departure date / TOF')
        
    z = bracket[0] if bracket[0] == bracket[1] else brentq(residual, *bracket, xtol=1e-10, rtol=1e-10, maxiter=100)
    C, S = stumpff_c(z), stumpff_s(z)
    y = r1n + r2n + A * (z * S - 1) / math.sqrt(C)
    f = 1 - y / r1n
    g = A * math.sqrt(y / mu)
    gd = 1 - y / r2n
    
    if abs(g) < 1e-12: 
        raise ValueError('Lambert solution produced near-zero Lagrange g')
    return (r2 - f * r1) / g, (gd * r2 - r1) / g


# ---------------------------------------------------------------------------
# 2. N-body dynamics and shooting solver (the physical reference model)
# ---------------------------------------------------------------------------
def parking_orbit_burn(vinf, mu, radius, altitude):
    """Calculate the burn from a circular parking orbit to a hyperbolic orbit."""
    rp = radius + altitude
    vc = math.sqrt(mu / rp)
    vh = math.sqrt(vinf * vinf + 2 * mu / rp)
    return vh - vc


def nbody_acceleration(t_seconds, r_sc, t_dep_days):
    """Compute spacecraft acceleration from the Sun, Earth, and Mars."""
    r = np.asarray(r_sc, dtype=float)
    rn = float(np.linalg.norm(r))
    if rn < 1e6: 
        raise ValueError('Spacecraft approached too close to Sun for this model')
        
    acc = -MU_SUN * r / rn**3
    day = t_dep_days + t_seconds / DAY_S
    
    for body, mu in (('earth', MU_EARTH), ('mars', MU_MARS)):
        rb, _ = planet_state(body, day)
        rel = rb - r
        reln = float(np.linalg.norm(rel))
        rbn = float(np.linalg.norm(rb))
        if reln < 1e-3: 
            raise ValueError(f'Spacecraft singularity near {body}')
        acc += mu * rel / reln**3 - mu * rb / rbn**3
    return acc


def integrate_spacecraft(t_dep_days, tof_days, r0, v0, sample_count=1):
    """Numerically propagate spacecraft position and velocity over the flight."""
    end_s = float(tof_days) * DAY_S
    y0 = np.r_[np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)]
    
    def rhs(t, state): 
        """Return the position and velocity derivatives for integration."""
        return np.r_[state[3:], nbody_acceleration(t, state[:3], t_dep_days)]
        
    times = np.linspace(0, end_s, max(2, sample_count)) if sample_count > 1 else [end_s]
    sol = solve_ivp(rhs, (0, end_s), y0, method='DOP853', t_eval=times, rtol=3e-10, atol=1e-7, max_step=3 * DAY_S)
    
    if not sol.success or sol.y.shape[1] != len(times): 
        raise RuntimeError(f'N-body integration failed: {sol.message}')
    return sol.y[:3, :].T, sol.y[3:, :].T, sol


def make_boundary_guess(dep, tof):
    """Build a Lambert-based initial guess and Mars arrival target."""
    re, ve = planet_state('earth', dep)
    rm, vm = planet_state('mars', dep + tof)
    vd, va = lambert_universal(re, rm, tof)
    
    vie = vd - ve
    vim = va - vm
    r0 = re + EARTH_SOI_KM * unit(vie)
    target = rm - MARS_SOI_KM * unit(vim)
    vsoi, _ = lambert_universal(r0, target, tof)
    
    rel = vsoi - ve
    direction = unit(rel)
    vinf = max(.05, float(np.linalg.norm(vie)))
    speed = math.sqrt(vinf**2 + 2 * MU_EARTH / EARTH_SOI_KM)
    vguess = ve + speed * direction
    
    return dict(r0=r0, v0_guess=vguess, r_target=target, r_earth=re, v_earth=ve, r_mars=rm, v_mars=vm)


def solve_nbody_transfer(t_dep_days, tof_days, miss_tolerance_km=100., max_nfev=35, keep_path=False):
    """Correct departure velocity to hit the Mars SOI target and compute burns."""
    if tof_days <= 0: 
        raise ValueError('Time of flight must be positive')
        
    b = make_boundary_guess(t_dep_days, tof_days)
    r0 = b['r0']
    target = b['r_target']
    count = {'n': 0}
    
    def residual(v0):
        """Return the current solver mismatch for the trial variable."""
        count['n'] += 1
        try: 
            return (integrate_spacecraft(t_dep_days, tof_days, r0, v0)[0][-1] - target) / 100000.
        except (RuntimeError, ValueError, FloatingPointError): 
            return np.full(3, 1e3)
            
    correction = least_squares(residual, b['v0_guess'], method='lm', diff_step=2e-5, ftol=2e-9, xtol=2e-9, gtol=2e-9, max_nfev=max_nfev)
    v0 = correction.x
    ra, va, sol = integrate_spacecraft(t_dep_days, tof_days, r0, v0, sample_count=500 if keep_path else 1)
    rend, vend = ra[-1], va[-1]
    miss = float(np.linalg.norm(rend - target))
    
    if not np.isfinite(miss) or miss > miss_tolerance_km: 
        raise RuntimeError(f'N-body shooting did not converge: miss={miss:.1f} km, evaluations={count["n"]}, status={correction.message}')
        
    rs = r0 - b['r_earth']
    vs = v0 - b['v_earth']
    re = rend - b['r_mars']
    ve = vend - b['v_mars']
    
    if np.dot(rs, vs) <= 0: 
        raise RuntimeError("Spacecraft is not moving outward through Earth's SOI")
    if np.dot(re, ve) >= 0: 
        raise RuntimeError("Spacecraft is not moving inward through Mars's SOI")
        
    e2 = float(np.dot(vs, vs) - 2 * MU_EARTH / np.linalg.norm(rs))
    m2 = float(np.dot(ve, ve) - 2 * MU_MARS / np.linalg.norm(re))
    
    if e2 <= 0 or m2 <= 0: 
        raise RuntimeError('Boundary state is not hyperbolic relative to Earth or Mars')
        
    vie = math.sqrt(e2)
    vim = math.sqrt(m2)
    d1 = parking_orbit_burn(vie, MU_EARTH, R_EARTH, EARTH_PARKING_ALT_KM)
    d2 = parking_orbit_burn(vim, MU_MARS, R_MARS, MARS_PARKING_ALT_KM)
    
    out = dict(
        departure_days=float(t_dep_days),
        departure_date=days_to_date(t_dep_days),
        arrival_date=days_to_date(t_dep_days + tof_days),
        tof_days=float(tof_days),
        dv_depart_kms=float(d1),
        dv_arrive_kms=float(d2),
        dv_total_kms=float(d1 + d2),
        v_inf_earth_kms=vie,
        v_inf_mars_kms=vim,
        miss_distance_km=miss,
        shooting_nfev=count['n'],
        r0_km=r0,
        v0_kms=v0,
        r_target_km=target,
        r_end_km=rend,
        v_end_kms=vend,
        r_earth_depart_km=b['r_earth'],
        v_earth_depart_kms=b['v_earth'],
        r_mars_arrive_km=b['r_mars'],
        v_mars_arrive_kms=b['v_mars']
    )
    if keep_path: 
        out.update(path_r_km=ra, path_v_kms=va, path_times_days=sol.t / DAY_S)
    return out


# ---------------------------------------------------------------------------
# 3. ML features, dataset generation and training
# ---------------------------------------------------------------------------
def trajectory_features(dep, tof):
    """Build the normalized orbital feature vector used by the surrogate."""
    re, ve = planet_state('earth', dep)
    rm, vm = planet_state('mars', dep + tof)
    return np.r_[re / AU_KM, ve / 30., rm / AU_KM, vm / 24., tof / 300.].astype(float)


FEATURE_COLUMNS = [f'x{i:02d}' for i in range(13)]
TARGET_COLUMNS = ['dv_depart_kms', 'dv_arrive_kms']


def generate_dataset(n_samples=500, seed=42, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, tof_min=DEFAULT_TOF_MIN_DAYS, tof_max=DEFAULT_TOF_MAX_DAYS):
    """Sample feasible transfers and collect physics-based training labels."""
    if n_samples < 20: 
        raise ValueError('Use at least 20 samples; 300-2,000 is a more useful starting range.')
        
    start, end = date_to_days(start_date), date_to_days(end_date)
    if end <= start or tof_min <= 0 or tof_max <= tof_min: 
        raise ValueError('Invalid date or TOF bounds')
        
    rng = np.random.default_rng(seed)
    records = []
    attempts = 0
    max_attempts = max(n_samples * 4, n_samples + 50)
    t0 = time.time()
    
    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        dep = float(rng.uniform(start, end))
        tof = float(rng.uniform(tof_min, tof_max))
        try:
            phys = solve_nbody_transfer(dep, tof)
            feat = trajectory_features(dep, tof)
            row = {'departure_days_since_j2000': dep, 'departure_date': days_to_date(dep), 'tof_days': tof}
            row.update({f'x{i:02d}': float(x) for i, x in enumerate(feat)})
            row.update({k: float(phys[k]) for k in ('dv_depart_kms', 'dv_arrive_kms', 'dv_total_kms', 'v_inf_earth_kms', 'v_inf_mars_kms', 'miss_distance_km')})
            records.append(row)
            if len(records) % 25 == 0: 
                print(f'  Accepted {len(records):,}/{n_samples:,} trajectories...')
        except (ValueError, RuntimeError, OverflowError, FloatingPointError) as exc:
            if attempts % 25 == 0: 
                print(f'  attempted {attempts}, accepted {len(records)}; last rejection: {exc}')
                
    if len(records) < n_samples: 
        raise RuntimeError(f'Only generated {len(records)} of {n_samples} samples after {attempts} attempts. Try broader TOF range or lower --samples.')
        
    print(f'Created {len(records):,} validated samples in {time.time() - t0:.1f} s ({attempts} attempts).')
    return pd.DataFrame(records)


def train_models(df, seed=42):
    """Fit the MLP surrogate and evaluate it on held-out data."""
    X = df[FEATURE_COLUMNS].to_numpy(float)
    y = df[TARGET_COLUMNS].to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed)
    
    xs = StandardScaler()
    ys = StandardScaler()
    Xtr = xs.fit_transform(Xtr)
    Xte = xs.transform(Xte)
    ytr = ys.fit_transform(ytr)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=1e-4,
        learning_rate_init=8e-4, max_iter=700, early_stopping=True, validation_fraction=.15,
        n_iter_no_change=30, random_state=seed
    )
    
    print('Training MLP surrogate...')
    mlp.fit(Xtr, ytr)
    pred = ys.inverse_transform(mlp.predict(Xte))
    
    metrics = {
        'mae_depart_kms': float(mean_absolute_error(yte[:, 0], pred[:, 0])),
        'mae_arrive_kms': float(mean_absolute_error(yte[:, 1], pred[:, 1])),
        'mae_total_kms': float(mean_absolute_error(yte.sum(1), pred.sum(1))),
        'rmse_total_kms': float(np.sqrt(mean_squared_error(yte.sum(1), pred.sum(1)))),
        'r2_total': float(r2_score(yte.sum(1), pred.sum(1)))
    }
    
    print('\nHeld-out test-set performance (km/s):')
    for k, v in metrics.items(): 
        print(f'  {k}: {v:.4f}')
        
    return {
        'model': mlp, 'x_scaler': xs, 'y_scaler': ys, 'feature_columns': FEATURE_COLUMNS,
        'target_columns': TARGET_COLUMNS, 'metrics': metrics
    }, {'y_test_total': yte.sum(1), 'pred_total': pred.sum(1)}


def predict_delta_v(bundle, dep, tof):
    """Predict departure and arrival delta-v using the trained surrogate."""
    feat = trajectory_features(dep, tof).reshape(1, -1)
    pred = bundle['y_scaler'].inverse_transform(bundle['model'].predict(bundle['x_scaler'].transform(feat)))[0]
    return np.maximum(pred, 0.)


# ---------------------------------------------------------------------------
# 4. Training-side plots (model validation and predicted porkchop map)
# ---------------------------------------------------------------------------
def plot_model_validation(tb):
    """Plot predicted versus physical total delta-v on test samples."""
    y, p = tb['y_test_total'], tb['pred_total']
    lo = min(float(y.min()), float(p.min()))
    hi = max(float(y.max()), float(p.max()))
    
    plt.figure(figsize=(7, 6))
    plt.scatter(y, p, s=18, alpha=.55)
    plt.plot([lo, hi], [lo, hi], 'k--', label='Perfect prediction')
    plt.xlabel('N-body physics total delta-v (km/s)')
    plt.ylabel('ML-predicted total delta-v (km/s)')
    plt.title('ML surrogate validation')
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mars_nbody_validation.png', dpi=160)
    plt.show()


def plot_porkchop(bundle, start_date, end_date, tof_min, tof_max):
    """Plot predicted total delta-v across departure dates and flight times."""
    dep = np.linspace(date_to_days(start_date), date_to_days(end_date), 100)
    tof = np.linspace(tof_min, tof_max, 75)
    DEP, TOF = np.meshgrid(dep, tof)
    pairs = list(zip(DEP.ravel(), TOF.ravel()))
    
    X = np.vstack([trajectory_features(float(d), float(t)) for d, t in pairs])
    pred = np.maximum(bundle['y_scaler'].inverse_transform(bundle['model'].predict(bundle['x_scaler'].transform(X))), 0.).sum(1).reshape(DEP.shape)
    
    plt.figure(figsize=(10, 6))
    cf = plt.contourf(DEP, TOF, pred, levels=24)
    plt.colorbar(cf, label='ML-predicted total delta-v (km/s)')
    ticks = np.linspace(dep.min(), dep.max(), 6)
    plt.xticks(ticks, [days_to_date(d) for d in ticks], rotation=25, ha='right')
    plt.xlabel('Departure date')
    plt.ylabel('Time of flight (days)')
    plt.title('ML delta-v map: Sun-Earth-Mars trajectories')
    plt.tight_layout()
    plt.savefig('mars_nbody_porkchop.png', dpi=160)
    plt.show()


# ---------------------------------------------------------------------------
# 5. Main entry point: generate data, train, save, plot
# ---------------------------------------------------------------------------
def main():
    """Parse command-line arguments and run the configured workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--start-date', default=DEFAULT_START_DATE)
    parser.add_argument('--end-date', default=DEFAULT_END_DATE)
    parser.add_argument('--tof-min', type=float, default=DEFAULT_TOF_MIN_DAYS)
    parser.add_argument('--tof-max', type=float, default=DEFAULT_TOF_MAX_DAYS)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--no-save-csv', action='store_true')
    args = parser.parse_args()
    t0 = time.time()
    
    print(f'Earth Laplace SOI radius estimate: {EARTH_SOI_KM:.0f} km')
    print(f'Mars Laplace SOI radius estimate: {MARS_SOI_KM:.0f} km')
    print('1. Generating corrected multi-body trajectory dataset...')
    
    df = generate_dataset(args.samples, args.seed, args.start_date, args.end_date, args.tof_min, args.tof_max)
    
    if not args.no_save_csv: 
        df.to_csv('mars_nbody_training_data.csv', index=False)
        print('Saved mars_nbody_training_data.csv')
        
    print('\n2. Training and evaluating ML surrogate...')
    bundle, tb = train_models(df, args.seed)
    joblib.dump(bundle, 'mars_nbody_surrogate.joblib')
    print('Saved mars_nbody_surrogate.joblib')
        
    if not args.skip_plots:
        print('\n3. Plotting model validation and predicted porkchop map...')
        plot_model_validation(tb)
        plot_porkchop(bundle, args.start_date, args.end_date, args.tof_min, args.tof_max)
            
    print(f'\nCompleted in {time.time() - t0:.1f} s.')
    print('To find the optimum departure date / TOF, run the test script with --optimum-window.')


if __name__ == '__main__':
    main()
