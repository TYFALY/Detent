//! Critically damped spring, integrated *exactly*.
//!
//! The camera follows `x'' + 2ζω₀x' + ω₀²x = ω₀²·target` with `ζ = 1`.
//! For a target held constant over a step `Δt` the closed-form solution is
//!
//! ```text
//! x(t) = target + (c₁ + c₂·t)·e^(−ω₀t),   c₁ = x₀ − target,   c₂ = v₀ + ω₀·c₁
//! v(t) = (c₂ − ω₀·(c₁ + c₂·t))·e^(−ω₀t)
//! ```
//!
//! Using the analytic step instead of Euler/Verlet means the result is
//! unconditionally stable and independent of frame rate. Because the
//! impulse response `ω₀²·t·e^(−ω₀t)` is non-negative, the position starting
//! from rest is always a convex combination of past targets: it can never
//! overshoot, and it never leaves the hull of the targets it was given. The
//! camera solver relies on that to keep crop windows inside the frame.

/// Position and velocity of a 1-D spring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpringState {
    /// Position.
    pub x: f64,
    /// Velocity (units per second).
    pub v: f64,
}

impl SpringState {
    /// A spring at rest at `x`.
    pub const fn at_rest(x: f64) -> Self {
        Self { x, v: 0.0 }
    }
}

/// A critically damped (`ζ = 1`) spring with natural frequency `omega` (rad/s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CriticalSpring {
    /// Natural angular frequency ω₀ in rad/s. Settling time (to ~0.1 %) ≈ 9.2/ω₀.
    pub omega: f64,
}

impl CriticalSpring {
    /// Builds a spring that settles to within 2 % in roughly `seconds`
    /// (`(1 + ω₀t)·e^(−ω₀t) = 0.02` ⇒ `ω₀t ≈ 5.83`).
    pub fn with_settle_time(seconds: f64) -> Self {
        Self {
            omega: 5.834 / seconds.max(1e-3),
        }
    }

    /// Advances `state` by `dt` seconds toward a constant `target`.
    pub fn step(&self, state: SpringState, target: f64, dt: f64) -> SpringState {
        let w = self.omega;
        let c1 = state.x - target;
        let c2 = state.v + w * c1;
        let e = (-w * dt).exp();
        SpringState {
            x: target + (c1 + c2 * dt) * e,
            v: (c2 - w * (c1 + c2 * dt)) * e,
        }
    }

    /// Simulates a whole target trajectory sampled every `dt`, starting at rest
    /// on the first target. Returns one position per target sample.
    pub fn simulate(&self, targets: &[f64], dt: f64) -> Vec<f64> {
        let Some(&first) = targets.first() else {
            return Vec::new();
        };
        let mut s = SpringState::at_rest(first);
        targets
            .iter()
            .map(|&t| {
                let x = s.x;
                s = self.step(s, t, dt);
                x
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn matches_closed_form_settling() {
        let s = CriticalSpring::with_settle_time(0.5);
        let mut st = SpringState::at_rest(0.0);
        for _ in 0..15 {
            st = s.step(st, 1.0, 1.0 / 30.0);
        }
        // 0.5 s elapsed: within 2 % of target.
        assert!((st.x - 1.0).abs() < 0.021, "{st:?}");
        assert!(st.x < 1.0);
    }

    #[test]
    fn step_size_independence() {
        let s = CriticalSpring { omega: 9.0 };
        let mut a = SpringState::at_rest(2.0);
        for _ in 0..100 {
            a = s.step(a, -1.0, 0.01);
        }
        let b = s.step(SpringState::at_rest(2.0), -1.0, 1.0);
        assert!((a.x - b.x).abs() < 1e-9 && (a.v - b.v).abs() < 1e-9);
    }

    proptest! {
        /// From rest, a step input converges monotonically without overshoot.
        #[test]
        fn no_overshoot_from_rest(
            x0 in -1e4f64..1e4,
            target in -1e4f64..1e4,
            omega in 0.5f64..80.0,
            dt in 1e-3f64..0.1,
            steps in 1usize..600,
        ) {
            let s = CriticalSpring { omega };
            let mut st = SpringState::at_rest(x0);
            let mut prev_gap = (x0 - target).abs();
            for _ in 0..steps {
                st = s.step(st, target, dt);
                let gap = st.x - target;
                // Same side of the target as where we started (or on it).
                prop_assert!(gap * (x0 - target) >= -1e-9);
                prop_assert!(gap.abs() <= prev_gap + 1e-9);
                prev_gap = gap.abs();
            }
        }

        /// Starting at rest, the position never leaves the convex hull of the
        /// targets seen so far — for *any* target sequence.
        #[test]
        fn stays_in_target_hull(
            targets in prop::collection::vec(-500.0f64..500.0, 1..300),
            omega in 0.5f64..60.0,
        ) {
            let s = CriticalSpring { omega };
            let xs = s.simulate(&targets, 1.0 / 30.0);
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for (x, t) in xs.iter().zip(&targets) {
                lo = lo.min(*t);
                hi = hi.max(*t);
                prop_assert!(*x >= lo - 1e-6 && *x <= hi + 1e-6, "x={x} hull=[{lo},{hi}]");
            }
        }

        /// A monotone non-decreasing target is followed monotonically from below.
        #[test]
        fn monotone_input_monotone_output(
            deltas in prop::collection::vec(0.0f64..5.0, 1..300),
            omega in 0.5f64..60.0,
        ) {
            let mut acc = 0.0;
            let targets: Vec<f64> = deltas.iter().map(|d| { acc += d; acc }).collect();
            let s = CriticalSpring { omega };
            let xs = s.simulate(&targets, 1.0 / 60.0);
            for w in xs.windows(2) {
                prop_assert!(w[1] >= w[0] - 1e-9);
            }
            for (x, t) in xs.iter().zip(&targets) {
                prop_assert!(*x <= *t + 1e-9);
            }
        }
    }
}
