//! Exact rational frame-rate arithmetic.
//!
//! All timeline math is done in integer microseconds and integer frame
//! indices. Floating point is only used for camera geometry. NTSC rates such
//! as `30000/1001` are represented exactly, so frame <-> time conversions are
//! reproducible across platforms.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};

/// Microseconds per millisecond.
pub const US_PER_MS: i64 = 1_000;
/// Microseconds per second.
pub const US_PER_S: i64 = 1_000_000;

/// A rational frame rate (`num / den` frames per second).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct FrameRate {
    /// Numerator (e.g. `30000`).
    pub num: u32,
    /// Denominator (e.g. `1001`).
    pub den: u32,
}

impl FrameRate {
    /// Constructs a validated frame rate.
    pub fn new(num: u32, den: u32) -> Result<Self> {
        if num == 0 || den == 0 {
            return Err(CoreError::InvalidFrameRate { num, den });
        }
        Ok(Self { num, den })
    }

    /// Frames per second as `f64` (display / geometry only).
    pub fn as_f64(self) -> f64 {
        f64::from(self.num) / f64::from(self.den.max(1))
    }

    /// Exact start time of `frame` in microseconds (rounded half-up).
    pub fn frame_to_us(self, frame: i64) -> i64 {
        let num = i128::from(self.num.max(1));
        let n = i128::from(frame) * i128::from(self.den) * i128::from(US_PER_S);
        div_round(n, num) as i64
    }

    /// Index of the frame boundary nearest to `us`.
    pub fn nearest_frame(self, us: i64) -> i64 {
        let n = i128::from(us) * i128::from(self.num);
        let d = i128::from(self.den.max(1)) * i128::from(US_PER_S);
        div_round(n, d) as i64
    }

    /// Index of the last frame boundary at or before `us`.
    pub fn floor_frame(self, us: i64) -> i64 {
        let n = i128::from(us) * i128::from(self.num);
        let d = i128::from(self.den.max(1)) * i128::from(US_PER_S);
        n.div_euclid(d) as i64
    }

    /// Duration of a single frame in microseconds (rounded).
    pub fn frame_duration_us(self) -> i64 {
        self.frame_to_us(1)
    }
}

impl Default for FrameRate {
    fn default() -> Self {
        Self { num: 30, den: 1 }
    }
}

/// Round-half-away-from-zero integer division for `i128`.
fn div_round(n: i128, d: i128) -> i128 {
    if d == 0 {
        return 0;
    }
    let q = n.div_euclid(d);
    let r = n.rem_euclid(d);
    if r * 2 >= d { q + 1 } else { q }
}

/// Converts milliseconds to microseconds.
pub const fn ms_to_us(ms: u64) -> i64 {
    (ms as i64).saturating_mul(US_PER_MS)
}

/// Converts microseconds to (rounded) milliseconds, clamping negatives to 0.
pub fn us_to_ms(us: i64) -> u64 {
    if us <= 0 {
        0
    } else {
        ((us + US_PER_MS / 2) / US_PER_MS) as u64
    }
}

/// A half-open microsecond interval `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct UsRange {
    /// Inclusive start (µs).
    pub start_us: i64,
    /// Exclusive end (µs).
    pub end_us: i64,
}

impl UsRange {
    /// Creates a range; callers are responsible for `start <= end`.
    pub const fn new(start_us: i64, end_us: i64) -> Self {
        Self { start_us, end_us }
    }

    /// Length in microseconds (never negative).
    pub fn len_us(self) -> i64 {
        (self.end_us - self.start_us).max(0)
    }

    /// True when the two half-open ranges share any instant.
    pub fn overlaps(self, other: Self) -> bool {
        self.start_us < other.end_us && other.start_us < self.end_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ntsc_round_trip_is_exact() {
        let fps = FrameRate::new(30_000, 1_001).unwrap();
        for frame in [0_i64, 1, 29, 30, 1_799, 107_892] {
            let us = fps.frame_to_us(frame);
            assert_eq!(fps.nearest_frame(us), frame, "frame {frame} -> {us}us");
        }
        assert_eq!(fps.frame_duration_us(), 33_367);
    }

    #[test]
    fn floor_and_nearest_differ_past_midpoint() {
        let fps = FrameRate::new(25, 1).unwrap();
        assert_eq!(fps.floor_frame(59_999), 1);
        assert_eq!(fps.nearest_frame(59_999), 1);
        assert_eq!(fps.nearest_frame(60_000), 2);
        assert_eq!(fps.floor_frame(-1), -1);
    }

    #[test]
    fn rejects_zero_rates() {
        assert!(FrameRate::new(0, 1).is_err());
        assert!(FrameRate::new(30, 0).is_err());
    }

    #[test]
    fn ms_us_conversions() {
        assert_eq!(ms_to_us(1_500), 1_500_000);
        assert_eq!(us_to_ms(1_499_600), 1_500);
        assert_eq!(us_to_ms(-5), 0);
    }
}
