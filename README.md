# PID Control

This project simulates PID control for a spring-damper system. It includes a
discrete PID controller, response plots and measurements, automated tests, and
a short report of the results. The controller and RK4 plant simulation are
implemented in Python; NumPy and Matplotlib are used for calculations and plots.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- ReportLab (only needed to rebuild the PDF)

## Run everything

From this directory:

```powershell
python -m pip install -r requirements.txt
python pid_control.py
python build_report.py
python -m unittest
```

The simulation script recreates every plot under `figures/` and writes the
controller comparison measurements to `metrics.csv`. The report builder reads
those results and creates `PID_Control_Assignment.pdf` in this directory.
Both scripts use fixed random seeds for noisy-sensor plots so reruns are
repeatable.

To run only the simulations and figures, omit `python build_report.py`.
The final command runs the automated checks for the analytical P-only result,
force limits, anti-windup, derivative kick, and seeded noise repeatability.

## What is in the project

- `pid_control.py` — hand-written PID, spring-damper simulation, RK4 plant
  integration, performance measurements, and all plot generation.
- `build_report.py` — builds the teaching PDF from the generated
  figures and metrics.
- `APPROACH.md` — brief submission write-up: approach, results, limitations,
  and what I would explore next.
- `test_pid_control.py` — standard-library unit tests for key behaviors.
- `figures/` — feedback diagram and all generated response plots.
- `metrics.csv` — numerical P, PI, PD, and PID comparison.
- `PID_Control_Assignment.pdf` — submission-ready document.

The assignment PDF is included as supporting documentation. The repository
deliverables are the source code, this README, and APPROACH.md.

The controller supports output limits, conditional-integration anti-windup,
derivative-on-measurement, and a first-order derivative filter. The simulations
do not use a control-systems library.

## Model and approach

The plant equation is `x'' = (u - 10 x' - 20 x) / 1`. The controller updates
the force at each sample, and RK4 advances the plant state. The P-only result
is checked against the steady-state force balance. P, PI, PD, and PID responses
are compared using the same response metrics. Additional simulations cover
force saturation, sensor noise, a setpoint step, slower sampling, and
Ziegler-Nichols tuning. `APPROACH.md` summarizes the results and limitations.

## Simulation conventions

The primary plant is a 1 kg mass, 10 N s/m damper, and 20 N/m spring. Position
is in metres, force in newtons, and time in seconds. The target steps from
0 m to 1 m at the start of each standard response. The plant is integrated with
RK4 at a fixed controller update interval. Response metrics use the first
10%-to-90% rise, peak overshoot as a percent of the 1 m step, a permanent
within-2% settling band, and the mean position over the final second.

The slow-sampling example intentionally keeps the same gains while changing
the update interval; in real hardware, gains may need retuning when the sample
period changes.
