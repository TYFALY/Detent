//! Streaming audio DSP: loudness envelope, zero-crossing rate, vocal-band
//! spectral flux, and pause / excitement-peak detection.
//!
//! Memory is bounded: [`AudioAnalyzer`] keeps one FFT window of history
//! (`fft_size` samples) plus at most one partial hop, independent of how much
//! audio is pushed through it. The only thing that grows is the per-hop
//! envelope (three `f32` per 10 ms = 1.2 KB per second of media).

use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::error::{CoreError, Result};
use crate::index::{AudioTelemetry, ExcitementPeak, PauseKind, SilenceInterval};

/// Analyzer configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DspConfig {
    /// Input sample rate (mono `f32` in `[-1, 1]`).
    pub sample_rate: u32,
    /// Envelope hop in milliseconds.
    pub hop_ms: u32,
    /// FFT window length in samples (power of two recommended).
    pub fft_size: usize,
    /// Lower edge of the vocal band for spectral flux (Hz).
    pub band_lo_hz: f32,
    /// Upper edge of the vocal band for spectral flux (Hz).
    pub band_hi_hz: f32,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            hop_ms: 10,
            fft_size: 512,
            band_lo_hz: 300.0,
            band_hi_hz: 3_400.0,
        }
    }
}

/// Per-hop envelope (struct-of-arrays).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Envelope {
    /// Hop size in milliseconds.
    pub hop_ms: u32,
    /// Sample rate of the analysed signal.
    pub sample_rate: u32,
    /// RMS level per hop (dBFS, floored at [`DB_FLOOR`]).
    pub rms_dbfs: Vec<f32>,
    /// Zero-crossing rate per hop.
    pub zcr: Vec<f32>,
    /// Vocal-band spectral flux per hop.
    pub flux: Vec<f32>,
}

impl Envelope {
    /// Number of hops.
    pub fn len(&self) -> usize {
        self.rms_dbfs.len()
    }

    /// True when no hop has been produced.
    pub fn is_empty(&self) -> bool {
        self.rms_dbfs.is_empty()
    }
}

/// Lowest representable level; digital silence maps here.
pub const DB_FLOOR: f32 = -100.0;

/// Streaming analyzer. Push arbitrarily sized chunks, then call [`finish`](Self::finish).
pub struct AudioAnalyzer {
    cfg: DspConfig,
    hop: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    history: Vec<f32>,
    pending: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    prev_mag: Vec<f32>,
    band: (usize, usize),
    last_sample: f32,
    samples_in: u64,
    out: Envelope,
}

impl std::fmt::Debug for AudioAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioAnalyzer")
            .field("cfg", &self.cfg)
            .field("hops", &self.out.len())
            .field("samples_in", &self.samples_in)
            .finish_non_exhaustive()
    }
}

impl AudioAnalyzer {
    /// Creates an analyzer, validating the configuration.
    pub fn new(cfg: DspConfig) -> Result<Self> {
        let hop = (u64::from(cfg.sample_rate) * u64::from(cfg.hop_ms) / 1_000) as usize;
        if hop == 0 {
            return Err(CoreError::InvalidDspConfig(
                "sample_rate * hop_ms must yield at least one sample per hop".into(),
            ));
        }
        if cfg.fft_size < hop || cfg.fft_size < 16 {
            return Err(CoreError::InvalidDspConfig(format!(
                "fft_size {} must be >= hop ({hop}) and >= 16",
                cfg.fft_size
            )));
        }
        if !(cfg.band_lo_hz >= 0.0 && cfg.band_lo_hz < cfg.band_hi_hz) {
            return Err(CoreError::InvalidDspConfig(
                "band_lo_hz < band_hi_hz".into(),
            ));
        }
        let n = cfg.fft_size;
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n);
        let window = (0..n)
            .map(|i| {
                let x = std::f32::consts::TAU * i as f32 / n as f32;
                0.5 - 0.5 * x.cos()
            })
            .collect();
        let bin_hz = cfg.sample_rate as f32 / n as f32;
        let lo = ((cfg.band_lo_hz / bin_hz).floor() as usize).max(1);
        let hi = ((cfg.band_hi_hz / bin_hz).ceil() as usize).clamp(lo + 1, n / 2);
        let scratch_len = fft.get_inplace_scratch_len();
        Ok(Self {
            cfg,
            hop,
            fft,
            window,
            history: vec![0.0; n],
            pending: Vec::with_capacity(hop),
            spectrum: vec![Complex::default(); n],
            fft_scratch: vec![Complex::default(); scratch_len],
            prev_mag: vec![0.0; hi - lo],
            band: (lo, hi),
            last_sample: 0.0,
            samples_in: 0,
            out: Envelope {
                hop_ms: cfg.hop_ms,
                sample_rate: cfg.sample_rate,
                ..Envelope::default()
            },
        })
    }

    /// Samples per hop.
    pub fn hop_samples(&self) -> usize {
        self.hop
    }

    /// Total samples consumed so far.
    pub fn samples_consumed(&self) -> u64 {
        self.samples_in
    }

    /// Feeds mono samples. Chunk boundaries do not affect the output.
    pub fn push(&mut self, mut samples: &[f32]) {
        self.samples_in += samples.len() as u64;
        // Complete a partially filled hop first.
        if !self.pending.is_empty() {
            let need = self.hop - self.pending.len();
            let take = need.min(samples.len());
            self.pending.extend_from_slice(&samples[..take]);
            samples = &samples[take..];
            if self.pending.len() == self.hop {
                let hop = std::mem::take(&mut self.pending);
                self.process_hop(&hop);
                self.pending = hop;
                self.pending.clear();
            }
        }
        let mut chunks = samples.chunks_exact(self.hop);
        for hop in chunks.by_ref() {
            self.process_hop(hop);
        }
        self.pending.extend_from_slice(chunks.remainder());
    }

    /// Flushes a trailing partial hop (zero padded) and returns the envelope.
    pub fn finish(mut self) -> Envelope {
        if !self.pending.is_empty() {
            let mut hop = std::mem::take(&mut self.pending);
            hop.resize(self.hop, 0.0);
            self.process_hop(&hop);
        }
        self.out
    }

    fn process_hop(&mut self, hop: &[f32]) {
        // Loudness + zero crossings over the hop itself.
        let mut energy = 0.0_f64;
        let mut crossings = 0_u32;
        let mut prev = self.last_sample;
        for &s in hop {
            energy += f64::from(s) * f64::from(s);
            if (s >= 0.0) != (prev >= 0.0) {
                crossings += 1;
            }
            prev = s;
        }
        self.last_sample = prev;
        let rms = (energy / hop.len() as f64).sqrt();
        let db = if rms > 0.0 {
            (20.0 * rms.log10()).max(f64::from(DB_FLOOR)) as f32
        } else {
            DB_FLOOR
        };
        self.out.rms_dbfs.push((db * 10.0).round() / 10.0);
        self.out.zcr.push(crossings as f32 / hop.len() as f32);

        // Slide the FFT history window by one hop.
        let n = self.history.len();
        self.history.copy_within(hop.len().., 0);
        self.history[n - hop.len()..].copy_from_slice(hop);
        for ((dst, &x), &w) in self
            .spectrum
            .iter_mut()
            .zip(&self.history)
            .zip(&self.window)
        {
            *dst = Complex::new(x * w, 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.spectrum, &mut self.fft_scratch);

        // Half-wave rectified, log-compressed flux in the vocal band.
        let (lo, hi) = self.band;
        let mut flux = 0.0_f32;
        for (prev, c) in self.prev_mag.iter_mut().zip(&self.spectrum[lo..hi]) {
            let mag = (1.0 + 100.0 * c.norm()).ln();
            flux += (mag - *prev).max(0.0);
            *prev = mag;
        }
        let flux = flux / (hi - lo) as f32;
        self.out.flux.push((flux * 1_000.0).round() / 1_000.0);
    }
}

/// Pause / peak detection parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PauseConfig {
    /// Minimum pause length (ms).
    pub min_pause_ms: u32,
    /// Non-quiet gaps up to this length (ms) are bridged inside a pause.
    pub bridge_ms: u32,
    /// Minimum dB above the noise floor for the quiet threshold.
    pub min_margin_db: f32,
    /// Fraction of the floor->speech distance used as quiet threshold.
    pub margin_ratio: f32,
    /// The quiet threshold is always at least this far below speech level.
    pub min_speech_gap_db: f32,
    /// Zero-crossing rate above which a quiet-ish hop counts as breath.
    pub breath_zcr: f32,
    /// Breath hops must be at least this many dB below speech level.
    pub breath_below_speech_db: f32,
    /// Excitement peak threshold in standard deviations.
    pub peak_sigma: f32,
    /// Minimum spacing between excitement peaks (ms).
    pub peak_spacing_ms: u32,
}

impl Default for PauseConfig {
    fn default() -> Self {
        Self {
            min_pause_ms: 100,
            bridge_ms: 20,
            min_margin_db: 6.0,
            margin_ratio: 0.35,
            min_speech_gap_db: 15.0,
            breath_zcr: 0.30,
            breath_below_speech_db: 12.0,
            peak_sigma: 1.5,
            peak_spacing_ms: 400,
        }
    }
}

/// Runs pause and peak detection over an envelope and assembles telemetry.
pub fn analyze(env: Envelope, cfg: &PauseConfig) -> AudioTelemetry {
    let hop_ms = u64::from(env.hop_ms.max(1));
    let n = env.len();
    if n == 0 {
        return AudioTelemetry {
            hop_ms: env.hop_ms,
            analysis_rate: env.sample_rate,
            noise_floor_dbfs: DB_FLOOR,
            speech_level_dbfs: DB_FLOOR,
            ..AudioTelemetry::default()
        };
    }
    let floor = percentile(&env.rms_dbfs, 0.05);
    let voiced: Vec<f32> = env
        .rms_dbfs
        .iter()
        .copied()
        .filter(|&d| d > floor + cfg.min_margin_db)
        .collect();
    let speech = if voiced.is_empty() {
        floor
    } else {
        percentile(&voiced, 0.5)
    };
    // Between floor and speech, but never within `min_speech_gap_db` of speech:
    // material with almost no silence has a percentile "floor" inside speech.
    let quiet_threshold = (floor + (cfg.margin_ratio * (speech - floor)).max(cfg.min_margin_db))
        .min(speech - cfg.min_speech_gap_db);
    let smoothed = median3(&env.rms_dbfs);

    // Classify each hop.
    #[derive(Clone, Copy, PartialEq)]
    enum Hop {
        Quiet,
        Breath,
        Active,
    }
    let class: Vec<Hop> = (0..n)
        .map(|i| {
            if smoothed[i] < quiet_threshold {
                Hop::Quiet
            } else if env.zcr[i] > cfg.breath_zcr
                && smoothed[i] < speech - cfg.breath_below_speech_db
            {
                Hop::Breath
            } else {
                Hop::Active
            }
        })
        .collect();

    // Run-length encode pause hops, bridging short active gaps.
    let bridge = (cfg.bridge_ms as usize).div_ceil(hop_ms as usize);
    let min_len = (cfg.min_pause_ms as usize).div_ceil(hop_ms as usize).max(1);
    let mut pauses = Vec::new();
    let mut i = 0;
    while i < n {
        if class[i] == Hop::Active {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i; // exclusive end of last pause hop
        let mut j = i;
        while j < n {
            if class[j] != Hop::Active {
                end = j + 1;
                j += 1;
            } else {
                let gap_end = (j..n).find(|&k| class[k] != Hop::Active).unwrap_or(n);
                if gap_end - j <= bridge && gap_end < n {
                    j = gap_end;
                } else {
                    break;
                }
            }
        }
        if end - start >= min_len {
            let slice = &env.rms_dbfs[start..end];
            let (valley_off, &min_db) = slice
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &DB_FLOOR));
            let breath_hops = class[start..end]
                .iter()
                .filter(|c| **c == Hop::Breath)
                .count();
            let kind = if breath_hops * 10 >= (end - start) * 3 {
                PauseKind::Breath
            } else {
                PauseKind::Silence
            };
            pauses.push(SilenceInterval {
                start_ms: start as u64 * hop_ms,
                end_ms: end as u64 * hop_ms,
                min_dbfs: min_db,
                valley_ms: (start + valley_off) as u64 * hop_ms + hop_ms / 2,
                kind,
            });
        }
        i = end.max(start + 1);
    }

    let peaks = detect_peaks(&env, floor, speech, cfg);

    AudioTelemetry {
        hop_ms: env.hop_ms,
        analysis_rate: env.sample_rate,
        noise_floor_dbfs: floor,
        speech_level_dbfs: speech,
        pauses,
        peaks,
        rms_dbfs: env.rms_dbfs,
        zcr: env.zcr,
        flux: env.flux,
    }
}

fn detect_peaks(env: &Envelope, floor: f32, speech: f32, cfg: &PauseConfig) -> Vec<ExcitementPeak> {
    let n = env.len();
    let hop_ms = u64::from(env.hop_ms.max(1));
    let active: Vec<usize> = (0..n)
        .filter(|&i| env.rms_dbfs[i] > floor + cfg.min_margin_db)
        .collect();
    if active.len() < 8 {
        return Vec::new();
    }
    let (fm, fs) = mean_std(active.iter().map(|&i| env.flux[i]));
    let (rm, rs) = mean_std(active.iter().map(|&i| env.rms_dbfs[i]));
    let raw: Vec<f32> = (0..n)
        .map(|i| {
            let fz = (env.flux[i] - fm) / fs.max(1e-6);
            let rz = (env.rms_dbfs[i] - rm) / rs.max(1e-6);
            fz + 0.5 * rz
        })
        .collect();
    let curve = moving_average(&raw, 5);
    let (cm, cs) = mean_std(active.iter().map(|&i| curve[i]));
    let threshold = cm + cfg.peak_sigma * cs;
    let radius = (cfg.peak_spacing_ms as usize / hop_ms as usize / 2).max(1);

    let mut peaks = Vec::new();
    for i in 0..n {
        let v = curve[i];
        if v < threshold || env.rms_dbfs[i] < speech - 6.0 {
            continue;
        }
        let lo = i.saturating_sub(radius);
        let hi = (i + radius + 1).min(n);
        // Strict on the left, non-strict on the right: plateaus yield one peak.
        let is_max =
            curve[lo..i].iter().all(|&x| x < v) && curve[i + 1..hi].iter().all(|&x| x <= v);
        if is_max {
            peaks.push(ExcitementPeak {
                t_ms: i as u64 * hop_ms + hop_ms / 2,
                score: ((v - cm) / cs.max(1e-6) * 100.0).round() / 100.0,
            });
        }
    }
    peaks
}

fn percentile(xs: &[f32], q: f32) -> f32 {
    if xs.is_empty() {
        return DB_FLOOR;
    }
    let mut v = xs.to_vec();
    let k = ((v.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    let (_, nth, _) = v.select_nth_unstable_by(k, f32::total_cmp);
    *nth
}

fn median3(xs: &[f32]) -> Vec<f32> {
    let n = xs.len();
    (0..n)
        .map(|i| {
            if i == 0 || i + 1 == n {
                xs[i]
            } else {
                let (a, b, c) = (xs[i - 1], xs[i], xs[i + 1]);
                a.max(b).min(a.min(b).max(c))
            }
        })
        .collect()
}

fn moving_average(xs: &[f32], width: usize) -> Vec<f32> {
    let half = width / 2;
    let n = xs.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0_f64);
    for &x in xs {
        let last = prefix.last().copied().unwrap_or(0.0);
        prefix.push(last + f64::from(x));
    }
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            ((prefix[hi] - prefix[lo]) / (hi - lo) as f64) as f32
        })
        .collect()
}

fn mean_std(it: impl Iterator<Item = f32>) -> (f32, f32) {
    let (mut n, mut sum, mut sq) = (0_u64, 0.0_f64, 0.0_f64);
    for x in it {
        n += 1;
        sum += f64::from(x);
        sq += f64::from(x) * f64::from(x);
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / n as f64;
    let var = (sq / n as f64 - mean * mean).max(0.0);
    (mean as f32, var.sqrt() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    /// Deterministic xorshift noise in [-1, 1].
    fn noise(seed: &mut u32) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        (*seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Builds a test signal from `(kind, seconds)` segments.
    /// kinds: 'v' voiced (harmonic tone, -12 dBFS-ish), 's' silence (-70 dB noise),
    /// 'b' breath (-38 dB white noise).
    fn synth(segments: &[(char, f32)]) -> Vec<f32> {
        let mut out = Vec::new();
        let mut seed = 0x1234_5678;
        for &(kind, secs) in segments {
            let n = (secs * SR as f32) as usize;
            for i in 0..n {
                let t = (out.len() + i) as f32 / SR as f32;
                let s = match kind {
                    'v' => {
                        let f0 = 160.0;
                        let mut v = 0.0;
                        for h in 1..=8 {
                            v += (std::f32::consts::TAU * f0 * h as f32 * t).sin() / h as f32;
                        }
                        0.15 * v + 0.002 * noise(&mut seed)
                    }
                    'b' => 0.012 * noise(&mut seed),
                    _ => 0.0003 * noise(&mut seed),
                };
                out.push(s);
            }
        }
        out
    }

    fn run(signal: &[f32], chunk: usize) -> Envelope {
        let mut a = AudioAnalyzer::new(DspConfig::default()).unwrap();
        for c in signal.chunks(chunk) {
            a.push(c);
        }
        a.finish()
    }

    #[test]
    fn rms_of_full_scale_sine_is_minus_3db() {
        let sig: Vec<f32> = (0..SR)
            .map(|i| (std::f32::consts::TAU * 1_000.0 * i as f32 / SR as f32).sin())
            .collect();
        let env = run(&sig, 4096);
        assert_eq!(env.len(), 100);
        for &db in &env.rms_dbfs[1..] {
            assert!((db - -3.0).abs() < 0.1, "got {db}");
        }
    }

    #[test]
    fn digital_silence_hits_floor() {
        let env = run(&vec![0.0; 1_600], 333);
        assert!(env.rms_dbfs.iter().all(|&d| d == DB_FLOOR));
    }

    #[test]
    fn chunking_is_invariant() {
        let sig = synth(&[('v', 0.5), ('s', 0.3), ('v', 0.4)]);
        let a = run(&sig, 160);
        let b = run(&sig, 7);
        let c = run(&sig, 100_000);
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn partial_hop_is_flushed() {
        let env = run(&vec![0.1; 250], 64);
        assert_eq!(env.len(), 2);
    }

    #[test]
    fn rejects_degenerate_config() {
        let bad = DspConfig {
            sample_rate: 50,
            ..DspConfig::default()
        };
        assert!(AudioAnalyzer::new(bad).is_err());
        let bad = DspConfig {
            fft_size: 64,
            ..DspConfig::default()
        };
        assert!(AudioAnalyzer::new(bad).is_err());
    }

    #[test]
    fn zcr_separates_noise_from_tone() {
        let env = run(&synth(&[('v', 0.3), ('b', 0.3)]), 512);
        let tone = env.zcr[5..25].iter().sum::<f32>() / 20.0;
        let breath = env.zcr[35..55].iter().sum::<f32>() / 20.0;
        assert!(tone < 0.1, "tone zcr {tone}");
        assert!(breath > 0.4, "breath zcr {breath}");
    }

    #[test]
    fn detects_silences_at_the_right_place() {
        let sig = synth(&[('v', 1.0), ('s', 0.4), ('v', 1.0), ('s', 0.6), ('v', 1.0)]);
        let tel = analyze(run(&sig, 1024), &PauseConfig::default());
        let silences: Vec<_> = tel
            .pauses
            .iter()
            .filter(|p| p.kind == PauseKind::Silence)
            .collect();
        assert_eq!(silences.len(), 2, "{:?}", tel.pauses);
        let near = |a: u64, b: u64| a.abs_diff(b) <= 30;
        assert!(near(silences[0].start_ms, 1_000) && near(silences[0].end_ms, 1_400));
        assert!(near(silences[1].start_ms, 2_400) && near(silences[1].end_ms, 3_000));
        for s in silences {
            assert!(s.start_ms <= s.valley_ms && s.valley_ms <= s.end_ms);
            assert!(s.min_dbfs < -60.0);
        }
    }

    #[test]
    fn classifies_breath_between_phrases() {
        // Room tone at both ends establishes the noise floor, as in real recordings.
        let sig = synth(&[('s', 0.4), ('v', 1.0), ('b', 0.35), ('v', 1.0), ('s', 0.4)]);
        let tel = analyze(run(&sig, 1024), &PauseConfig::default());
        let breath = tel
            .pauses
            .iter()
            .find(|p| p.kind == PauseKind::Breath)
            .expect("breath detected");
        assert!(breath.start_ms.abs_diff(1_400) <= 40, "{breath:?}");
        assert!(breath.end_ms.abs_diff(1_750) <= 40, "{breath:?}");
        assert!(tel.noise_floor_dbfs < -65.0);
    }

    #[test]
    fn short_dips_are_not_pauses() {
        let sig = synth(&[('v', 1.0), ('s', 0.05), ('v', 1.0)]);
        let tel = analyze(run(&sig, 1024), &PauseConfig::default());
        assert!(
            tel.pauses
                .iter()
                .all(|p| p.end_ms <= 1_000 || p.start_ms >= 1_050),
            "{:?}",
            tel.pauses
        );
        assert!(
            !tel.pauses
                .iter()
                .any(|p| p.start_ms > 900 && p.end_ms < 1_200)
        );
    }

    #[test]
    fn excitement_peaks_follow_onsets() {
        // Quiet-ish speech with three loud bursts.
        let mut sig = Vec::new();
        for _ in 0..3 {
            sig.extend(synth(&[('s', 0.8)]));
            let burst = synth(&[('v', 0.3)]);
            sig.extend(burst.iter().map(|x| x * 3.0));
            let calm = synth(&[('v', 0.9)]);
            sig.extend(calm.iter().map(|x| x * 0.4));
        }
        let tel = analyze(run(&sig, 2048), &PauseConfig::default());
        assert!(!tel.peaks.is_empty(), "no peaks");
        // Every peak sits in (or right after) a burst: 0.8s + k*2.0s.
        for p in &tel.peaks {
            let local = p.t_ms % 2_000;
            assert!((780..=1_250).contains(&local), "peak at {} ms", p.t_ms);
        }
        // Peak spacing is respected.
        for w in tel.peaks.windows(2) {
            assert!(w[1].t_ms - w[0].t_ms >= 200);
        }
    }

    #[test]
    fn empty_envelope_is_handled() {
        let tel = analyze(Envelope::default(), &PauseConfig::default());
        assert!(tel.pauses.is_empty() && tel.peaks.is_empty());
    }

    #[test]
    fn median3_and_percentile() {
        assert_eq!(median3(&[1.0, 5.0, 2.0, 8.0]), vec![1.0, 2.0, 5.0, 8.0]);
        assert_eq!(percentile(&[3.0, 1.0, 2.0], 0.5), 2.0);
    }
}
