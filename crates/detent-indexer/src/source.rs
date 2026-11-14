//! Decoder abstractions. The pipeline only depends on these traits, so it is
//! testable with synthetic sources and independent of how media is decoded.

use crate::error::Result;

/// Packed RGB24 frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 3` bytes, row-major RGB.
    pub data: Vec<u8>,
}

impl RgbFrame {
    /// A black frame.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0; width as usize * height as usize * 3],
        }
    }

    /// A solid-colour frame.
    pub fn solid(width: u32, height: u32, rgb: [u8; 3]) -> Self {
        let mut f = Self::new(width, height);
        for px in f.data.as_chunks_mut::<3>().0 {
            *px = rgb;
        }
        f
    }

    /// Byte length for the given dimensions.
    pub fn byte_len(width: u32, height: u32) -> usize {
        width as usize * height as usize * 3
    }
}

/// A source of mono `f32` PCM at a fixed sample rate.
pub trait PcmSource {
    /// Fills `out` with up to `out.len()` samples; returns the count, `0` at EOF.
    fn read(&mut self, out: &mut [f32]) -> Result<usize>;
    /// Releases resources and reports decoder errors.
    fn close(self) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// A source of fixed-size RGB frames at the source frame rate.
pub trait FrameSource {
    /// Frame width.
    fn width(&self) -> u32;
    /// Frame height.
    fn height(&self) -> u32;
    /// Decodes the next frame into `frame`; returns `false` at EOF.
    fn next_frame(&mut self, frame: &mut RgbFrame) -> Result<bool>;
    /// Releases resources and reports decoder errors.
    fn close(self) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// In-memory PCM source (tests, benchmarks).
#[derive(Debug, Clone)]
pub struct VecPcm {
    samples: Vec<f32>,
    pos: usize,
}

impl VecPcm {
    /// Wraps a sample buffer.
    pub fn new(samples: Vec<f32>) -> Self {
        Self { samples, pos: 0 }
    }
}

impl PcmSource for VecPcm {
    fn read(&mut self, out: &mut [f32]) -> Result<usize> {
        let n = out.len().min(self.samples.len() - self.pos);
        out[..n].copy_from_slice(&self.samples[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// In-memory frame source (tests).
#[derive(Debug, Clone)]
pub struct VecFrames {
    frames: std::vec::IntoIter<RgbFrame>,
    width: u32,
    height: u32,
}

impl VecFrames {
    /// Wraps frames; all must share the given dimensions.
    pub fn new(width: u32, height: u32, frames: Vec<RgbFrame>) -> Self {
        Self {
            frames: frames.into_iter(),
            width,
            height,
        }
    }
}

impl FrameSource for VecFrames {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn next_frame(&mut self, frame: &mut RgbFrame) -> Result<bool> {
        match self.frames.next() {
            Some(f) => {
                *frame = f;
                Ok(true)
            }
            None => Ok(false),
        }
    }
}
