MMA3001 PROJECT - EARTH-TO-MARS TRANSFER USING MACHINE LEARNING
================================================================

OVERVIEW
--------
This project trains a machine-learning surrogate to estimate the departure
and arrival delta-v required for an Earth-to-Mars transfer. A numerical
Sun-Earth-Mars gravity model provides reference trajectories and training
labels. The test script compares predictions with physics-based solutions,
checks encounter distances, and searches for suitable transfer durations
and departure windows.

This is an educational/research model, not flight-ready mission software.

PROJECT FILES
-------------
- mars_nbody_ml.py: Physical trajectory model, training-data generation,
  machine-learning training, validation plot, and porkchop plot.
- Test script (for example, test_mars_nbody_ml.py): Loads the trained model,
  searches transfer options, compares trajectories, and exports results.
- mars_nbody_surrogate.joblib: Trained model and scalers (generated).
- mars_nbody_training_data.csv: Training data (generated unless disabled).
- HTML documentation: Code reference, if included in this repository.

IMPORTANT: The supplied test script currently imports
'mars_nbody_mlv2 as physics'. Ensure the physical-model filename matches
that import, or adjust the import to match your actual module filename.

REQUIREMENTS
------------
Python 3 and these packages:

    pip install numpy scipy pandas scikit-learn matplotlib joblib

GETTING STARTED
---------------
1. Open a terminal in the project directory.
2. Train the surrogate model:

    python mars_nbody_ml.py --samples 500

   For a quick test:

    python mars_nbody_ml.py --samples 30 --skip-plots

3. Run the testing script after the .joblib model has been generated:

    python test_mars_nbody_ml.py --departure-date 2030-09-01 --tof-days 240

   Alternatively, search transfer durations with:

    python test_mars_nbody_ml.py --departure-date 2030-09-01 --ml-tof-search

   Or search a departure-date window:

    python test_mars_nbody_ml.py --optimum-window --window-start 2026-01-01 --window-end 2026-12-31

   Run 'python test_mars_nbody_ml.py --help' for all supported options.
   If your actual test script has a different filename, use that name.

HOW IT WORKS
------------
1. Estimate Earth and Mars positions and velocities from approximate
   orbital elements.
2. Use Lambert's problem to construct an initial transfer guess.
3. Integrate spacecraft motion with solar, Earth, and Mars gravity.
4. Correct the departure velocity using a shooting/least-squares solver.
5. Calculate departure and arrival delta-v from parking-orbit assumptions.
6. Train an MLP regressor on validated physical-model examples.
7. Evaluate predictions and compare uncorrected ML, Lambert, and corrected
   physical trajectories, including spacecraft-to-Mars distance.

GENERATED RESULTS
-----------------
Training may create:
- mars_nbody_training_data.csv
- mars_nbody_surrogate.joblib
- mars_nbody_validation.png
- mars_nbody_porkchop.png

Testing can create (depending on selected mode):
- mars_trajectory_comparison.png
- mars_trajectory_comparison.csv
- mars_lambert_comparison.png
- mars_solver_performance.csv
- mars_solver_performance.png
- mars_nbody_best_candidate.csv
- mars_nbody_trajectory.png

LIMITATIONS
-----------
- Approximate planetary ephemerides, not precision mission ephemerides.
- Simplified Sun-Earth-Mars gravitational environment.
- Impulsive departure/capture burns and idealized parking orbits.
- A small shooting residual measures accuracy to a selected target point;
  it is not the same as distance to Mars's centre.
- ML-predicted delta-v alone does not guarantee a Mars encounter; use
  physical propagation and encounter-distance checks to validate it.
