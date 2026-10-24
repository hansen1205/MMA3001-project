# Earth-to-Mars Trajectory Simulation
## With an ML Delta-v Surrogate

Purpose: generate physically corrected Earth-to-Mars transfer examples, train a neural-network surrogate to estimate departure and arrival delta-v, and then test or optimize transfers using the trained model alongside the physical N-body solver.


## 1. Project overview

This project combines an orbital-mechanics reference model with a machine-learning (ML) surrogate. The physical model estimates transfer trajectories using approximate planetary ephemerides, a Lambert initial guess, Sun-Earth-Mars gravitational acceleration, numerical integration, and a shooting correction that adjusts the spacecraft's initial velocity to reach a target near Mars's sphere of influence (SOI).
The training script uses accepted physical solutions to build a dataset and trains an MLPRegressor to predict the departure and arrival delta-v magnitudes. The testing script loads the saved model, compares ML-guided trajectories with the physical solution, evaluates closest approach to Mars, and can search for a suitable time of flight (TOF) or launch window.

## 2. Files and responsibilities

| File | Purpose | Main outputs / role |
| --- | --- | --- |
| mars_nbody_mlv2.py | Physical reference model and model training. | Generates the physics-based dataset, trains/evaluates the MLP surrogate, saves the model bundle, and creates validation/porkchop plots. |
| testmodelTOFml_cleaned.py | Model testing, trajectory comparison, and optimization. | Loads the trained model, compares ML and physical trajectories, measures Mars distance, runs TOF or launch-window searches, and writes CSV/plot outputs. |
| mars_nbody_surrogate.joblib | Trained model bundle required by the test script. | Contains the trained neural network, feature scaler, target scaler, feature/target names, and training metrics. |
Keep both Python scripts and the generated mars_nbody_surrogate.joblib file in the same folder unless you explicitly provide a different model path with --model-file. The test script imports the physical module as mars_nbody_mlv2, so the physical file should retain that filename.

## 3. Requirements and installation

Use Python 3.10 or a compatible Python version with the following packages:
`pip install numpy scipy pandas scikit-learn matplotlib joblib`
Recommended project folder layout:
project_folder/
├── mars_nbody_mlv2.py
├── testmodelTOFml_cleaned.py
└── (generated after training) mars_nbody_surrogate.joblib
Run commands from this folder in a terminal or PowerShell window. If using a virtual environment, activate it before installing packages and running the scripts.

## 4. Workflow A — generate data and train the ML model

The first script performs these steps:
- Samples departure dates and times of flight within the requested ranges.
- Computes approximate Earth and Mars positions and velocities from orbital elements.
- Uses a Lambert solution to create an initial transfer estimate.
- Integrates spacecraft motion under solar gravity and Earth/Mars perturbations, including indirect terms in the heliocentric frame.
- Uses a least-squares shooting correction to adjust the initial velocity so the spacecraft reaches the Mars-SOI target.
- Calculates departure and arrival delta-v labels from hyperbolic excess speeds and assumed circular parking orbits.
- Trains an MLPRegressor using scaled features and targets, evaluates held-out test data, and saves the trained model bundle.

### 4.1 Quick test

`python mars_nbody_mlv2.py --samples 30 --skip-plots`
This is a small smoke test, not a guarantee of high predictive accuracy. The physical trajectory solver can reject difficult samples, so dataset generation may take time.

### 4.2 Typical training run

`python mars_nbody_mlv2.py --samples 500`
Useful optional arguments:
| Argument | Meaning |
| --- | --- |
| --samples N | Number of accepted physical transfer examples to generate; default is 1000. |
| --seed N | Random seed for repeatable sampling and train/test splitting; default is 42. |
| --start-date YYYY-MM-DD | Start of the departure-date sampling range; default is 2026-01-01. |
| --end-date YYYY-MM-DD | End of the departure-date sampling range; default is 2040-01-01. |
| --tof-min DAYS | Minimum time of flight in days; default is 120. |
| --tof-max DAYS | Maximum time of flight in days; default is 350. |
| --skip-plots | Skip the validation and predicted porkchop plots. |
| --no-save-csv | Do not save the generated training dataset CSV. |

### 4.3 Training outputs

- mars_nbody_surrogate.joblib — trained ML model bundle used by the second script.
- mars_nbody_training_data.csv — generated training examples, unless --no-save-csv is used.
- mars_nbody_validation.png — predicted versus physical total delta-v for held-out test examples.
- mars_nbody_porkchop.png — ML-predicted total delta-v over departure dates and TOF values.
The two plots are produced unless --skip-plots is set. The script prints held-out test metrics including MAE, RMSE, and R² for total delta-v.

## 5. Workflow B — test trajectories and optimize transfers

After training, run the second script. It loads mars_nbody_surrogate.joblib and uses the physical module for planetary states and trajectory integration. The ML model predicts delta-v magnitudes; the ML trajectory initial velocity direction is based on the Lambert/boundary guess. The propagated trajectory is then checked against Mars's position and SOI.

### 5.1 Run a fixed time-of-flight case

`python testmodelTOFml_cleaned.py --departure-date 2031-09-01 --tof-days 240 --no-show`
This simulates the selected departure date and 240-day TOF, writes the comparison plot and CSV data, and saves solver-performance information. Remove --no-show to open plot windows interactively.

### 5.2 Search TOF for one departure date

If --tof-days is omitted, the script asks you to select a search strategy unless a strategy flag is provided.
`# Physics-only coarse/fine search`
`python testmodelTOFml_cleaned.py --departure-date 2031-09-01 --physics-tof-search`

`# ML shortlist, then verify candidates with the physical solver`
`python testmodelTOFml_cleaned.py --departure-date 2031-09-01 --ml-tof-search --ml-candidates 4`

`# Fastest option: select the ML-predicted minimum and verify it with one physical solve`
`python testmodelTOFml_cleaned.py --departure-date 2031-09-01 --ml-only-tof`
The default single-date search bounds are 150–400 days. Use --min-tof and --max-tof to change them. The physics-only method is slower but searches using physical evaluations; ML-guided search checks a shortlist; ML-only uses one ML-selected candidate for physical verification.

### 5.3 Optimize departure date and TOF together

`python testmodelTOFml_cleaned.py --optimum-window \`
  --window-start 2026-01-01 --window-end 2026-12-31 \
  --window-tof-min 120 --window-tof-max 350 \
  --window-departure-samples 90 --window-tof-samples 55 \
  --window-candidates 8 --no-show
This option searches a coarse grid of departure dates and TOFs using the ML surrogate, then verifies a shortlist of candidates with the physical model. It saves the selected candidate and its trajectory plot.

### 5.4 Common testing options

| Argument | Meaning |
| --- | --- |
| --model-file PATH | Model bundle path; default is mars_nbody_surrogate.joblib. |
| --departure-date DATE | Departure date for single-date simulation/search; default is 2030-09-01. |
| --tof-days DAYS | Use a fixed TOF rather than searching. |
| --sample-count N | Number of trajectory samples used for plotting; default is 600. |
| --plot-file PATH | Physical-versus-ML trajectory plot path. |
| --csv-file PATH | Trajectory comparison CSV path. |
| --performance-csv PATH | Solver timing/performance CSV path. |
| --performance-plot PATH | Solver timing plot path. |
| --no-show | Save plots without opening interactive plot windows. |
| --best-csv PATH | Output CSV for the best launch-window candidate. |
| --window-plot-file PATH | Output trajectory plot for the launch-window optimum. |

## 6. Outputs from the testing script

- mars_trajectory_comparison.png — plot comparing the physical N-body trajectory with the ML trajectory.
- mars_trajectory_comparison.csv — sampled trajectory and distance data for the comparison.
- mars_solver_performance.csv — timing and solver performance measurements; TOF search time is appended when a search is run.
- mars_solver_performance.png — visualization of solver performance.
- mars_nbody_best_candidate.csv — selected launch-window candidate when --optimum-window is used (or another path supplied by --best-csv).
- mars_nbody_trajectory.png — plot of the optimized launch-window trajectory (or another path supplied by --window-plot-file).

## 7. What the model predicts and how to interpret the results

The ML targets are departure delta-v and arrival delta-v, in km/s. Their sum is reported as total delta-v. The physical model calculates these from the Earth-relative and Mars-relative hyperbolic excess speeds at the SOI boundaries, with assumed parking-orbit altitudes of 200 km at Earth and 250 km at Mars.
A small shooting residual means the physical solver reached its target point near the Mars SOI. It is not the same as the spacecraft's distance from Mars's centre. The test script calculates spacecraft-to-Mars distance separately from the propagated trajectory; compare this distance with the standard Mars SOI radius printed by the script to assess SOI entry. SOI entry is not the same as atmospheric entry, surface landing, or a captured orbit.
The ML model is a fast surrogate for delta-v estimation, not a replacement for trajectory verification. ML-only and ML-guided results should be interpreted with the physical verification results and closest-approach distance, not solely by the lowest predicted delta-v.

## 8. Model assumptions and limitations

- Planetary positions and velocities use approximate Keplerian orbital elements; they are not a full high-precision ephemeris.
- The propagation includes the Sun, Earth, and Mars gravity in the model's heliocentric formulation.
- The model uses impulsive-burn/parking-orbit delta-v estimates; it does not simulate finite-duration thrust or a detailed launch vehicle.
- The model does not include a full spacecraft attitude model, launch-site geometry, or detailed atmospheric entry/landing.
- The training and test models are intended for educational and research exploration, not flight-quality mission design.
- Results depend on the chosen date and TOF search ranges, dataset size, random seed, numerical convergence, and the accuracy of the approximate orbital model.

## 9. Troubleshooting

Module import error: Confirm mars_nbody_mlv2.py is in the same folder as testmodelTOFml_cleaned.py and that the physical file has not been renamed.
Model file not found: Run the training script first and confirm mars_nbody_surrogate.joblib exists, or pass its path using --model-file.
Dataset generation is slow or fails: Physical shooting correction is computationally expensive and some date/TOF samples may be rejected. Try a smaller --samples value for a quick test or adjust the date/TOF bounds.
No plot window appears: Use --no-show to save plots without GUI windows. Check the output files in the current working directory.
A candidate fails physical verification: An ML prediction is only an estimate. Review the physical solver messages and closest-approach/SOI-distance results; a low predicted delta-v alone does not guarantee a feasible trajectory.

## 10. Quick-start checklist

| 1. | Install the required Python packages. |
| --- | --- |
| 2. | Place mars_nbody_mlv2.py and testmodelTOFml_cleaned.py in the same folder. |
| 3. | Train a model with: python mars_nbody_mlv2.py --samples 500 |
| 4. | Confirm mars_nbody_surrogate.joblib was created. |
| 5. | Test a fixed transfer with: python testmodelTOFml_cleaned.py --departure-date 2031-09-01 --tof-days 240 --no-show |
| 6. | Inspect the trajectory comparison plot, distance-to-Mars data, delta-v summary, and performance outputs. |
| 7. | Try --ml-tof-search or --optimum-window when ready to optimize. |

End of README

### NEW: Interactive testing mode

The updated testmodelTOFml_cleaned.py supports an interactive menu. When you run the script without selecting a mode or specifying a fixed TOF, it asks which search method you want to use. You can enter values at the prompts or press Enter to accept each displayed default.

### Run the testing script

Open a terminal or PowerShell in the project folder and run:
`python testmodelTOFml_cleaned.py`

### Interactive menu

The script offers these choices:
- 1 — Physics-only TOF search: performs a coarse/fine search using physical N-body evaluations; slowest, but checks candidates with the physical solver.
- 2 — ML-guided TOF search: scans transfer durations with the ML surrogate and verifies a shortlist using the physical solver.
- 3 — ML-only TOF search: selects the lowest ML-predicted delta-v candidate and performs one physical verification; fastest, but does not compare as many candidates physically.
- 4 — Optimum launch window: searches both departure date and time of flight (TOF), then verifies a shortlist of launch candidates using the physical model.

### Inputs for the optimum launch window

If you choose option 4, the script asks for the launch-window start and end dates, minimum and maximum TOF, number of departure-date samples, number of TOF samples, and number of candidates to verify physically. Press Enter at any prompt to use its default.
| Input | Default | Meaning |
| --- | --- | --- |
| Launch-window start | 2026-01-01 | First departure date considered |
| Launch-window end | 2026-12-31 | Last departure date considered |
| Minimum TOF | 120 days | Shortest transfer duration considered |
| Maximum TOF | 350 days | Longest transfer duration considered |
| Departure-date samples | 90 | Number of dates sampled across the window |
| TOF samples | 55 | Number of transfer durations sampled |
| Candidates to verify | 8 | Top ML-ranked launch candidates sent to the physical solver |
The default 90 × 55 grid evaluates 4,950 candidate combinations with the ML surrogate. The physical model then checks the shortlisted candidates. More samples give a finer initial grid but increase the amount of ML screening and do not guarantee a global optimum.

### Command-line mode is still supported

You can continue to run the script with command-line arguments for repeatable or automated runs. For example, the following Windows Command Prompt command runs a launch-window optimization without opening plot windows:
`python testmodelTOFml_cleaned.py --optimum-window --window-start 2026-01-01 --window-end 2026-12-31 --window-tof-min 120 --window-tof-max 350 --window-departure-samples 90 --window-tof-samples 55 --window-candidates 8 --no-show`
For a fixed departure date and TOF, use for example: python testmodelTOFml_cleaned.py --departure-date 2031-09-01 --tof-days 240 --no-show

### How to interpret the optimum-window result

The search returns the best candidate found within the selected date/TOF grid and shortlist that successfully passes the physical verification process. It is not guaranteed to be the global optimum. Review the reported total delta-v together with the trajectory plot and spacecraft-to-Mars distance/SOI information. A low ML-predicted delta-v by itself does not prove that a trajectory reaches Mars.

### Updated quick start: interactive mode

| 8. | Install the required packages: pip install numpy scipy pandas scikit-learn matplotlib joblib |
| --- | --- |
| 9. | Keep mars_nbody_mlv2.py, testmodelTOFml_cleaned.py, and mars_nbody_surrogate.joblib in the project folder. |
| 10. | If you have not trained the model yet, run: python mars_nbody_mlv2.py --samples 500 |
| 11. | Start the interactive test menu with: python testmodelTOFml_cleaned.py |
| 12. | Choose option 4 to optimize departure date and TOF together, then enter your search settings or press Enter for defaults. |
| 13. | Inspect mars_nbody_best_candidate.csv and mars_nbody_trajectory.png, along with any trajectory/distance outputs generated by your selected workflow. |
Documentation update: this section describes the interactive prompts added to testmodelTOFml_cleaned.py. Existing command-line arguments remain available.
