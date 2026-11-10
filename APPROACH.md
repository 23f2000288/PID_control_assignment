# Approach and results

## Approach

I modelled the assignment's 1 kg mass, spring, and damper with
`x'' = (u - 10 x' - 20 x) / 1`. A discrete PID controller calculates the force
at each sample, and a fourth-order Runge-Kutta step advances the plant state.
The controller is implemented directly in Python; NumPy and Matplotlib are
used for calculations and plots.

I first checked the P-only steady-state position against the force-balance
formula `x_ss = Kp / (Kp + 20)`. I then compared P, PI, PD, and PID responses
using the same target and response metrics. Finally, I simulated actuator
saturation, sensor noise, a setpoint step, slower sampling, and the specified
third-order plant for Ziegler-Nichols tuning.

## What worked

- The four simulated P-only final positions agree with the analytical
  calculation.
- Integral action removes the spring-load offset; derivative action shapes
  transient motion.
- Conditional integration prevents the overshoot seen with the 30 N force
  limit in the no-anti-windup comparison.
- Filtering derivative reduces force chatter from the seeded +/-1 mm sensor
  noise. Derivative on measurement removes the setpoint kick.
- The final PID example reaches the +/-2% band in 2.68 s with no overshoot for
  its stated gains and sample interval.

## What did not work as well

The classical Ziegler-Nichols PID gains were too aggressive for a quiet
response: the simulation overshot by 52.7% and took about 9.91 s to settle.
Reducing and refining the gains gave a slower 2.32 s rise, but removed the
overshoot and reduced settling time to about 4.22 s. Also, keeping the same
gains while increasing the sample interval from 0.01 s to 0.20 s made the
simulated loop diverge. These examples show that tuning and sample rate must
be evaluated together.

## What I would explore next

With more time, I would test the controller against actuator delay, friction,
measurement quantization, and changing mass or spring stiffness. I would also
compare more gain sets with a consistent objective that includes response
time, overshoot, control effort, and noise sensitivity. The current results
are simulations, not hardware validation.
