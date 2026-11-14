//! Piecewise-linear functions as flat FFmpeg expressions.
//!
//! A path through keyframes `(t₀,v₀) … (tₙ,vₙ)` (held after `tₙ`) is written
//! as a sum of hinge functions instead of nested `if()`s:
//!
//! ```text
//! f(t) = v₀ + k₀·t + Σⱼ (kⱼ − kⱼ₋₁)·max(0, t − tⱼ)
//! ```
//!
//! where `kⱼ` is the slope of segment `j` and the final slope is 0.
//!
//! FFmpeg's expression evaluator is depth-limited: a flat `a+b+c+…` parses
//! into a left-deep tree and silently fails beyond ~100 terms. Terms are
//! therefore emitted as a balanced parenthesised tree (depth `O(log n)`),
//! verified with 1 500-term paths.

/// Hinge decomposition of a piecewise-linear function.
#[derive(Debug, Clone, PartialEq)]
pub struct Hinges {
    /// Value at `t = 0`.
    pub v0: f64,
    /// Initial slope.
    pub k0: f64,
    /// `(tⱼ, Δkⱼ)` slope changes.
    pub kinks: Vec<(f64, f64)>,
}

impl Hinges {
    /// Builds hinges from keyframes sorted by time (`t` in seconds, first at 0).
    pub fn from_points(points: &[(f64, f64)]) -> Self {
        let Some(&(t0, v0)) = points.first() else {
            return Self {
                v0: 0.0,
                k0: 0.0,
                kinks: Vec::new(),
            };
        };
        let slopes: Vec<f64> = points
            .windows(2)
            .map(|w| {
                let dt = w[1].0 - w[0].0;
                if dt > 0.0 {
                    (w[1].1 - w[0].1) / dt
                } else {
                    0.0
                }
            })
            .chain(std::iter::once(0.0))
            .collect();
        // Shift so the first keyframe is at t = 0 even if it is not.
        let v0 = v0 - slopes[0] * t0;
        let mut kinks = Vec::with_capacity(points.len());
        for j in 1..points.len() {
            let dk = slopes[j] - slopes[j - 1];
            if dk != 0.0 {
                kinks.push((points[j].0, dk));
            }
        }
        Self {
            v0,
            k0: slopes[0],
            kinks,
        }
    }

    /// Evaluates the function (reference implementation for tests).
    pub fn eval(&self, t: f64) -> f64 {
        self.v0
            + self.k0 * t
            + self
                .kinks
                .iter()
                .map(|&(tj, dk)| dk * (t - tj).max(0.0))
                .sum::<f64>()
    }

    /// FFmpeg expression in the variable `var` (usually `t`).
    pub fn to_expr(&self, var: &str) -> String {
        let mut terms: Vec<String> = Vec::with_capacity(self.kinks.len() + 1);
        if num(self.k0) != "0" {
            terms.push(format!("{}*{var}", num(self.k0)));
        }
        terms.extend(
            self.kinks
                .iter()
                .filter(|(_, dk)| num(*dk) != "0")
                .map(|&(tj, dk)| format!("{}*max(0,{var}-{})", num(dk), num(tj))),
        );
        if terms.is_empty() {
            return num(self.v0);
        }
        format!("{}+{}", num(self.v0), balanced_sum(&terms))
    }
}

/// `(a+b)+(c+d)`-style sum with nesting depth `ceil(log2(n))`.
fn balanced_sum(terms: &[String]) -> String {
    match terms {
        [] => "0".to_owned(),
        [one] => format!("({one})"),
        _ => {
            let (l, r) = terms.split_at(terms.len() / 2);
            format!("({}+{})", balanced_sum(l), balanced_sum(r))
        }
    }
}

/// Maximum parenthesis nesting depth of an expression.
pub fn nesting_depth(expr: &str) -> usize {
    let (mut d, mut max) = (0_usize, 0_usize);
    for c in expr.chars() {
        match c {
            '(' => {
                d += 1;
                max = max.max(d);
            }
            ')' => d = d.saturating_sub(1),
            _ => {}
        }
    }
    max
}

fn num(x: f64) -> String {
    let s = format!("{x:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() {
        "0".to_owned()
    } else {
        s.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn lerp(points: &[(f64, f64)], t: f64) -> f64 {
        let i = points.partition_point(|p| p.0 <= t);
        if i == 0 {
            return points[0].1;
        }
        let a = points[i - 1];
        match points.get(i) {
            None => a.1,
            Some(b) => a.1 + (b.1 - a.1) * (t - a.0) / (b.0 - a.0),
        }
    }

    #[test]
    fn constant_and_ramp() {
        assert_eq!(Hinges::from_points(&[(0.0, 5.0)]).to_expr("t"), "5");
        let h = Hinges::from_points(&[(0.0, 0.0), (2.0, 10.0)]);
        assert_eq!(h.to_expr("t"), "0+((5*t)+(-5*max(0,t-2)))");
        assert_eq!(h.eval(1.0), 5.0);
        assert_eq!(h.eval(9.0), 10.0);
    }

    #[test]
    fn long_paths_stay_shallow() {
        let pts: Vec<(f64, f64)> = (0..1_500)
            .map(|i| (i as f64 * 0.033, ((i * i * 37) % 101) as f64))
            .collect();
        let e = Hinges::from_points(&pts).to_expr("t");
        // max(...) adds one level per term; the sum tree adds log2(n).
        assert!(nesting_depth(&e) <= 14, "depth {}", nesting_depth(&e));
        assert!(e.matches("max(").count() > 1_000);
    }

    proptest! {
        /// The hinge form equals linear interpolation (with hold) everywhere.
        #[test]
        fn matches_linear_interpolation(
            deltas in prop::collection::vec((0.01f64..2.0, -500.0f64..500.0), 1..40),
            probes in prop::collection::vec(0.0f64..1.0, 1..50),
        ) {
            let mut t = 0.0;
            let points: Vec<(f64, f64)> = std::iter::once((0.0, 100.0)).chain(deltas.iter().map(|&(dt, v)| {
                t += dt;
                (t, v)
            })).collect();
            let h = Hinges::from_points(&points);
            let end = points.last().unwrap().0;
            for p in probes {
                let x = p * (end + 1.0);
                let want = lerp(&points, x);
                prop_assert!((h.eval(x) - want).abs() < 1e-6 * (1.0 + want.abs()), "t={x}");
            }
        }
    }
}
