import unittest

import numpy as np

from pid_control import Gains, PID, response_metrics, simulate_block


class PIDControlTests(unittest.TestCase):
    def test_p_only_steady_state_matches_force_balance(self):
        for kp in (5.0, 20.0, 80.0, 200.0):
            with self.subTest(kp=kp):
                result = simulate_block(Gains(kp), duration=8.0)
                expected_position = kp / (kp + 20.0)
                self.assertAlmostEqual(result["position"][-1], expected_position, places=3)

    def test_output_limits_are_respected(self):
        result = simulate_block(
            Gains(120.0, 200.0),
            duration=4.0,
            force_limit=30.0,
            anti_windup=False,
        )
        self.assertLessEqual(float(np.max(result["force"])), 30.0)
        self.assertGreaterEqual(float(np.min(result["force"])), -30.0)

    def test_anti_windup_reduces_saturation_overshoot(self):
        gains = Gains(120.0, 200.0)
        without_fix = simulate_block(
            gains, duration=8.0, force_limit=30.0, anti_windup=False
        )
        with_fix = simulate_block(
            gains, duration=8.0, force_limit=30.0, anti_windup=True
        )
        self.assertLess(
            response_metrics(with_fix)["overshoot_percent"],
            response_metrics(without_fix)["overshoot_percent"],
        )

    def test_derivative_on_measurement_avoids_setpoint_kick(self):
        error_derivative = PID(
            Gains(0.0, 0.0, 1.0), derivative_on_measurement=False
        )
        measurement_derivative = PID(
            Gains(0.0, 0.0, 1.0), derivative_on_measurement=True
        )
        error_derivative.update(0.0, 0.0, 0.01)
        measurement_derivative.update(0.0, 0.0, 0.01)
        kick = error_derivative.update(1.0, 0.0, 0.01)
        no_kick = measurement_derivative.update(1.0, 0.0, 0.01)
        self.assertEqual(kick, 100.0)
        self.assertEqual(no_kick, 0.0)

    def test_sensor_noise_is_reproducible_with_same_seed(self):
        kwargs = dict(
            gains=Gains(100.0, 0.0, 10.0),
            duration=1.0,
            noise_amplitude=0.001,
            noise_seed=42,
        )
        first = simulate_block(**kwargs)
        second = simulate_block(**kwargs)
        np.testing.assert_array_equal(first["measurement"], second["measurement"])
        np.testing.assert_array_equal(first["force"], second["force"])

    def test_zero_time_step_is_rejected(self):
        controller = PID(Gains(1.0))
        with self.assertRaises(ValueError):
            controller.update(1.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
