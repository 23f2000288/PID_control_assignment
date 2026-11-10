"""Reproducible, hand-written PID simulations and plots for the assignment."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
MASS = 1.0
DAMPING = 10.0
SPRING = 20.0
TARGET = 1.0

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    }
)


@dataclass(frozen=True)
class Gains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0


class PID:
    """Discrete PID with derivative filtering, optional output limits and anti-windup."""

    def __init__(
        self,
        gains: Gains,
        output_limits: tuple[float, float] | None = None,
        derivative_filter_tau: float = 0.0,
        derivative_on_measurement: bool = True,
        anti_windup: bool = True,
    ) -> None:
        self.gains = gains
        self.output_limits = output_limits
        self.derivative_filter_tau = derivative_filter_tau
        self.derivative_on_measurement = derivative_on_measurement
        self.anti_windup = anti_windup
        self.integral = 0.0
        self.previous_error: float | None = None
        self.previous_measurement: float | None = None
        self.filtered_derivative = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt must be positive")
        error = setpoint - measurement
        if self.previous_error is None:
            raw_derivative = 0.0
        elif self.derivative_on_measurement:
            raw_derivative = -(measurement - self.previous_measurement) / dt
        else:
            raw_derivative = (error - self.previous_error) / dt

        tau = max(0.0, self.derivative_filter_tau)
        if tau:
            alpha = tau / (tau + dt)
            self.filtered_derivative = (
                alpha * self.filtered_derivative + (1.0 - alpha) * raw_derivative
            )
        else:
            self.filtered_derivative = raw_derivative

        candidate_integral = self.integral + error * dt
        p_term = self.gains.kp * error
        d_term = self.gains.kd * self.filtered_derivative
        candidate_output = p_term + self.gains.ki * candidate_integral + d_term

        if self.output_limits is not None and self.anti_windup and self.gains.ki:
            low, high = self.output_limits
            pushing_high = candidate_output > high and error > 0
            pushing_low = candidate_output < low and error < 0
            if not (pushing_high or pushing_low):
                self.integral = candidate_integral
        else:
            self.integral = candidate_integral

        output = p_term + self.gains.ki * self.integral + d_term
        if self.output_limits is not None:
            output = float(np.clip(output, *self.output_limits))

        self.previous_error = error
        self.previous_measurement = measurement
        return output


def spring_damper_derivative(state: np.ndarray, force: float) -> np.ndarray:
    position, velocity = state
    acceleration = (force - DAMPING * velocity - SPRING * position) / MASS
    return np.array([velocity, acceleration], dtype=float)


def rk4_step(state: np.ndarray, force: float, dt: float) -> np.ndarray:
    k1 = spring_damper_derivative(state, force)
    k2 = spring_damper_derivative(state + 0.5 * dt * k1, force)
    k3 = spring_damper_derivative(state + 0.5 * dt * k2, force)
    k4 = spring_damper_derivative(state + dt * k3, force)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_block(
    gains: Gains,
    duration: float = 8.0,
    dt: float = 0.01,
    *,
    force_limit: float | None = None,
    noise_amplitude: float = 0.0,
    noise_seed: int = 0,
    derivative_filter_tau: float = 0.0,
    derivative_on_measurement: bool = True,
    anti_windup: bool = True,
    setpoint_function=None,
) -> dict[str, np.ndarray]:
    steps = int(round(duration / dt))
    time = np.arange(steps + 1, dtype=float) * dt
    position = np.zeros(steps + 1)
    velocity = np.zeros(steps + 1)
    measured = np.zeros(steps + 1)
    force = np.zeros(steps + 1)
    setpoint = np.ones(steps + 1) * TARGET
    if setpoint_function is not None:
        setpoint = np.asarray(setpoint_function(time), dtype=float)

    limits = (-force_limit, force_limit) if force_limit is not None else None
    controller = PID(
        gains,
        output_limits=limits,
        derivative_filter_tau=derivative_filter_tau,
        derivative_on_measurement=derivative_on_measurement,
        anti_windup=anti_windup,
    )
    rng = np.random.default_rng(noise_seed)
    state = np.zeros(2, dtype=float)

    for index in range(steps):
        observed = state[0] + (
            rng.uniform(-noise_amplitude, noise_amplitude)
            if noise_amplitude
            else 0.0
        )
        measured[index] = observed
        force[index] = controller.update(setpoint[index], observed, dt)
        state = rk4_step(state, force[index], dt)
        position[index + 1], velocity[index + 1] = state
    measured[-1] = position[-1]
    force[-1] = force[-2] if steps else 0.0
    return {
        "time": time,
        "setpoint": setpoint,
        "position": position,
        "velocity": velocity,
        "measurement": measured,
        "force": force,
    }


def response_metrics(result: dict[str, np.ndarray]) -> dict[str, float]:
    t = result["time"]
    y = result["position"]
    target = float(result["setpoint"][-1])
    start = float(y[0])
    step = target - start
    direction = 1.0 if step >= 0 else -1.0
    normalized = (y - start) * direction
    magnitude = abs(step)

    def crossing_time(fraction: float) -> float:
        indices = np.flatnonzero(normalized >= fraction * magnitude)
        if not len(indices):
            return float("nan")
        index = int(indices[0])
        if index == 0:
            return float(t[0])
        y0, y1 = normalized[index - 1], normalized[index]
        if y1 == y0:
            return float(t[index])
        return float(t[index - 1] + (fraction * magnitude - y0) * (t[index] - t[index - 1]) / (y1 - y0))

    rise10, rise90 = crossing_time(0.1), crossing_time(0.9)
    rise_time = rise90 - rise10 if np.isfinite(rise10 + rise90) else float("nan")
    peak = float(np.max(normalized))
    overshoot = max(0.0, (peak - magnitude) / magnitude * 100.0) if magnitude else 0.0
    band = 0.02 * magnitude
    outside = np.flatnonzero(np.abs(y - target) > band)
    settling = (
        float(t[outside[-1] + 1])
        if len(outside) and outside[-1] + 1 < len(t)
        else (float("nan") if len(outside) else 0.0)
    )
    tail_start = max(0, len(t) - int(round(1.0 / (t[1] - t[0]))))
    final_position = float(np.mean(y[tail_start:]))
    return {
        "rise_time_s": rise_time,
        "overshoot_percent": overshoot,
        "settling_time_s": settling,
        "steady_state_error_m": target - final_position,
        "final_position_m": final_position,
    }


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def finish_chart(fig: plt.Figure, caption: str) -> None:
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8, color="#333333")
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.94))


def make_feedback_diagram() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(8, 2.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    blocks = [
        (0.7, "Target\n$r(t)$"),
        (3.0, "PID\ncontroller"),
        (5.4, "Plant\n(block/spring)"),
        (8.0, "Sensor\nposition $x(t)$"),
    ]
    for x, label in blocks:
        box = FancyBboxPatch(
            (x, 1.1), 1.5, 0.9, boxstyle="round,pad=0.12",
            facecolor="#e8f1fb", edgecolor="#326891", linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x + 0.75, 1.55, label, ha="center", va="center")
    arrows = [(2.2, 1.55, 3.0, 1.55), (4.5, 1.55, 5.4, 1.55), (6.9, 1.55, 8.0, 1.55)]
    for x1, y1, x2, y2 in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, color="#444444"))
    ax.add_patch(FancyArrowPatch((8.75, 1.05), (3.75, 0.65), connectionstyle="arc3,rad=-0.26", arrowstyle="->", mutation_scale=14, color="#444444"))
    ax.text(6.4, 0.37, "feedback: compare target with measured position", ha="center", va="center")
    ax.text(2.55, 2.2, "error $e=r-x$", ha="center", fontsize=9)
    ax.set_title("Closed-loop feedback: measure, compare, correct", pad=5)
    finish_chart(fig, "The sensor closes the loop so the controller can correct the remaining error.")
    save_fig(fig, "feedback_loop.png")


def make_spring_response_plots() -> dict[str, dict[str, float]]:
    colors = plt.get_cmap("tab10").colors
    p_gains = [5.0, 20.0, 80.0, 200.0]
    p_results = {kp: simulate_block(Gains(kp), duration=8) for kp in p_gains}

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for color, (kp, result) in zip(colors, p_results.items()):
        ax.plot(result["time"], result["position"], label=f"P only, $K_p={kp:g}$", color=color)
    ax.axhline(TARGET, color="black", linestyle="--", linewidth=1, label="target = 1 m")
    ax.set(title="P-only control: stronger gains reduce, but do not remove, offset", xlabel="Time (s)", ylabel="Block position (m)")
    ax.legend(ncol=2)
    finish_chart(fig, "For each finite P gain, the spring pulls the block back and leaves a predictable steady-state error.")
    save_fig(fig, "p_only_sweep.png")

    p = simulate_block(Gains(80), duration=8)
    pi = simulate_block(Gains(80, 40), duration=8)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(p["time"], p["position"], label="P only: $K_p=80$")
    ax.plot(pi["time"], pi["position"], label="PI: $K_p=80, K_i=40$")
    ax.axhline(TARGET, color="black", linestyle="--", label="target = 1 m")
    ax.set(title="Adding integral action removes steady-state offset", xlabel="Time (s)", ylabel="Block position (m)")
    ax.legend()
    finish_chart(fig, "Integral action supplies the steady force needed to hold the spring at the target.")
    save_fig(fig, "adding_integral.png")

    pd_plain = simulate_block(Gains(80), duration=8)
    pd = simulate_block(Gains(80, 0, 25), duration=8)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(pd_plain["time"], pd_plain["position"], label="P only: $K_p=80$")
    ax.plot(pd["time"], pd["position"], label="PD: $K_p=80, K_d=25$")
    ax.axhline(TARGET, color="black", linestyle="--", label="target = 1 m")
    ax.set(title="Derivative action damps motion but does not remove P offset", xlabel="Time (s)", ylabel="Block position (m)")
    ax.legend()
    finish_chart(fig, "D responds to motion and adds damping; without I, a spring still causes steady-state offset.")
    save_fig(fig, "adding_derivative.png")

    cases = {
        "P": (Gains(200), "#1f77b4"),
        "PI": (Gains(120, 60), "#ff7f0e"),
        "PD": (Gains(100, 0, 30), "#2ca02c"),
        "PID": (Gains(100, 40, 30), "#d62728"),
    }
    results: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for name, (gains, color) in cases.items():
        result = simulate_block(gains, duration=8)
        results[name] = result
        metrics[name] = response_metrics(result)
        ax.plot(result["time"], result["position"], label=name, color=color)
    ax.axhline(TARGET, color="black", linestyle="--", linewidth=1, label="target = 1 m")
    ax.axhspan(0.98, 1.02, color="grey", alpha=0.12, label="±2% settling band")
    ax.set(title="P, PI, PD and PID step responses", xlabel="Time (s)", ylabel="Block position (m)")
    ax.legend(ncol=3)
    finish_chart(fig, "PI/PID can remove spring-load offset; derivative action shapes transient motion.")
    save_fig(fig, "controller_comparison.png")

    pid = results["PID"]
    pm = metrics["PID"]
    t = pid["time"]
    y = pid["position"]
    peak_i = int(np.argmax(y))
    outside = np.flatnonzero(np.abs(y - TARGET) > 0.02)
    settling_i = int(outside[-1] + 1) if len(outside) and outside[-1] + 1 < len(t) else 0
    ten = int(np.flatnonzero(y >= 0.1)[0])
    ninety = int(np.flatnonzero(y >= 0.9)[0])
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.plot(t, y, color="#2463a6", label="PID position")
    ax.axhline(TARGET, color="black", linestyle="--", label="target = 1 m")
    ax.axhspan(0.98, 1.02, color="grey", alpha=0.13, label="±2% settling band")
    ax.axvspan(t[ten], t[ninety], color="#5aa469", alpha=0.15, label="10–90% rise interval")
    ax.scatter([t[peak_i]], [y[peak_i]], color="#b22222", zorder=5, label=f"peak = {y[peak_i]:.3f} m")
    ax.axvline(t[settling_i], color="#8c564b", linestyle=":", label=f"settled by {t[settling_i]:.2f} s")
    ax.annotate(f"overshoot {pm['overshoot_percent']:.1f}%", (t[peak_i], y[peak_i]), xytext=(12, 12), textcoords="offset points")
    ax.set(title="How to read a step response (PID example)", xlabel="Time (s)", ylabel="Block position (m)")
    ax.legend(ncol=2)
    finish_chart(fig, f"Metrics use 10–90% rise, peak overshoot, permanent ±2% settling and final 1 s mean error ({pm['steady_state_error_m']:.3g} m).")
    save_fig(fig, "response_metrics.png")
    return metrics


def simulate_third_order_pid(
    gains: Gains, duration: float = 18.0, dt: float = 0.005
) -> dict[str, np.ndarray]:
    """Simulate G(s)=1/(s+1)^3 as three cascaded unit first-order states."""
    steps = int(round(duration / dt))
    time = np.arange(steps + 1) * dt
    states = np.zeros((steps + 1, 3), dtype=float)
    output = np.zeros(steps + 1)
    control = np.zeros(steps + 1)
    controller = PID(gains, derivative_on_measurement=True)

    def derivative(q: np.ndarray, u: float) -> np.ndarray:
        return np.array([-q[0] + u, -q[1] + q[0], -q[2] + q[1]])

    for index in range(steps):
        output[index] = states[index, 2]
        control[index] = controller.update(1.0, output[index], dt)
        q = states[index]
        u = control[index]
        k1 = derivative(q, u)
        k2 = derivative(q + dt * k1 / 2, u)
        k3 = derivative(q + dt * k2 / 2, u)
        k4 = derivative(q + dt * k3, u)
        states[index + 1] = q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    output[-1] = states[-1, 2]
    control[-1] = control[-2]
    return {"time": time, "position": output, "force": control, "setpoint": np.ones_like(time)}


def make_practical_problem_plots() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    plain = simulate_block(Gains(120, 200), duration=8, force_limit=30, anti_windup=False)
    fixed = simulate_block(Gains(120, 200), duration=8, force_limit=30, anti_windup=True)
    ax = axes[0, 0]
    ax.plot(plain["time"], plain["position"], label="No anti-windup")
    ax.plot(fixed["time"], fixed["position"], label="Conditional integration")
    ax.axhline(1, color="black", linestyle="--", label="target")
    ax.set(title="(a) 30 N force limit: integral windup", xlabel="Time (s)", ylabel="Position (m)")
    ax.legend()
    ax.text(0.02, 0.02, "Same PI gains; stop integrating when saturation is worsened.", transform=ax.transAxes, fontsize=7)

    noisy_raw = simulate_block(
        Gains(100, 0, 10), duration=4, noise_amplitude=0.001, noise_seed=42,
        derivative_on_measurement=True,
    )
    noisy_filtered = simulate_block(
        Gains(100, 0, 10), duration=4, noise_amplitude=0.001, noise_seed=42,
        derivative_filter_tau=0.05, derivative_on_measurement=True,
    )
    ax = axes[0, 1]
    ax.plot(noisy_raw["time"], noisy_raw["force"], label="Raw D")
    ax.plot(noisy_filtered["time"], noisy_filtered["force"], label="Filtered D, τ=0.05 s")
    ax.set(title="(b) ±1 mm sensor noise amplified by D", xlabel="Time (s)", ylabel="Controller force (N)")
    ax.legend()
    ax.text(0.02, 0.02, "Same noisy sensor samples; a first-order D filter calms force chatter.", transform=ax.transAxes, fontsize=7)

    target_fn = lambda t: np.where(t < 0.5, 0.0, 1.0)
    kick = simulate_block(
        Gains(100, 0, 10), duration=1.5, derivative_on_measurement=False,
        setpoint_function=target_fn,
    )
    no_kick = simulate_block(
        Gains(100, 0, 10), duration=1.5, derivative_on_measurement=True,
        setpoint_function=target_fn,
    )
    ax = axes[1, 0]
    ax.plot(kick["time"], kick["force"], label="D on error (kick)")
    ax.plot(no_kick["time"], no_kick["force"], label="D on measurement")
    ax.axvline(0.5, color="black", linestyle=":", label="target step")
    ax.set(title="(c) Setpoint step causes derivative kick", xlabel="Time (s)", ylabel="Controller force (N)")
    ax.legend()
    ax.text(0.02, 0.02, "Using −d(measurement)/dt excludes the setpoint jump from D.", transform=ax.transAxes, fontsize=7)

    fast = simulate_block(Gains(100, 40, 30), duration=8, dt=0.01)
    slow = simulate_block(Gains(100, 40, 30), duration=8, dt=0.2)
    ax = axes[1, 1]
    ax.plot(fast["time"], fast["position"], label="Update every 0.01 s")
    ax.plot(slow["time"], slow["position"], "o-", markevery=3, markersize=3, label="Update every 0.20 s")
    ax.axhline(1, color="black", linestyle="--", label="target")
    ax.set(title="(d) Slow sampling destabilizes this gain set", xlabel="Time (s)", ylabel="Position (m)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.legend()
    ax.text(0.02, 0.02, "The 0.20 s update diverges; the 0.01 s run tracks the target.", transform=ax.transAxes, fontsize=7)

    fig.suptitle("Practical PID problems and fixes", fontsize=14)
    finish_chart(fig, "The paired traces isolate each issue; the slow sampled loop diverges with these gains.")
    save_fig(fig, "practical_problems.png")


def make_zn_plot() -> dict[str, float]:
    ku = 8.0
    pu = 2.0 * math.pi / math.sqrt(3.0)
    zn_pid = Gains(0.6 * ku, 0.6 * ku / (0.5 * pu), 0.6 * ku * (0.125 * pu))
    improved = Gains(2.0, 0.5, 1.5)
    p_at_ku = simulate_third_order_pid(Gains(ku), duration=14)
    znpid = simulate_third_order_pid(zn_pid, duration=18)
    tuned = simulate_third_order_pid(improved, duration=18)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    ax.plot(p_at_ku["time"], p_at_ku["position"], label=f"P-only $K_u={ku:g}$")
    ax.axhline(1, color="black", linestyle="--", label="target")
    ax.set(title="Find the ultimate gain", xlabel="Time (s)", ylabel="Output (unitless)")
    ax.legend()
    ax.text(0.03, 0.04, f"$P_u={pu:.2f}$ s (marginal oscillation)", transform=ax.transAxes, fontsize=8)

    ax = axes[1]
    ax.plot(znpid["time"], znpid["position"], label="ZN PID (4.8, 2.65, 2.18)")
    ax.plot(tuned["time"], tuned["position"], label="Refined PID (2.0, 0.5, 1.5)")
    ax.axhline(1, color="black", linestyle="--", label="target")
    ax.set(title="Ziegler–Nichols starting point, then refine", xlabel="Time (s)", ylabel="Output (unitless)")
    ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Ziegler–Nichols on $G(s)=1/(s+1)^3$", fontsize=13)
    finish_chart(fig, "The ZN recipe gives a reproducible initial setting; a gentler tuned set can reduce oscillation.")
    save_fig(fig, "ziegler_nichols.png")
    return {
        "Ku": ku,
        "Pu_s": pu,
        "ZN_PID_Kp": zn_pid.kp,
        "ZN_PID_Ki": zn_pid.ki,
        "ZN_PID_Kd": zn_pid.kd,
        "refined_Kp": improved.kp,
        "refined_Ki": improved.ki,
        "refined_Kd": improved.kd,
    }


def write_metrics_csv(metrics: dict[str, dict[str, float]]) -> None:
    path = ROOT / "metrics.csv"
    fields = ["controller", "rise_time_s", "overshoot_percent", "settling_time_s", "steady_state_error_m", "final_position_m"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for controller, values in metrics.items():
            writer.writerow({"controller": controller, **values})


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_feedback_diagram()
    metrics = make_spring_response_plots()
    make_practical_problem_plots()
    zn = make_zn_plot()
    write_metrics_csv(metrics)
    print("Generated figures in:", FIGURES)
    print("\nBlock metrics (10–90% rise, ±2% settling):")
    for name, values in metrics.items():
        print(
            f"{name:>3}: rise={values['rise_time_s']:.3f}s, "
            f"overshoot={values['overshoot_percent']:.2f}%, "
            f"settling={values['settling_time_s']:.3f}s, "
            f"final error={values['steady_state_error_m']:.5f}m"
        )
    print("\nZiegler–Nichols values:", zn)


if __name__ == "__main__":
    main()
