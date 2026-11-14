//! Keyframe contact sheets: a timestamped JPEG grid an LLM can look at to
//! judge framing without ingesting video. A 4x4 grid of 160 px tiles costs a
//! few hundred image tokens regardless of media length.

use std::path::Path;

use detent_core::index::ContactSheet;
use image::{Rgb, RgbImage, imageops};

use crate::error::{IndexError, Result};
use crate::source::RgbFrame;

/// Grid layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSpec {
    /// Columns.
    pub columns: u32,
    /// Rows.
    pub rows: u32,
    /// Tile width (px).
    pub tile_width: u32,
    /// Tile height (px).
    pub tile_height: u32,
}

/// Collects tiles from a frame stream at evenly spaced target times.
#[derive(Debug)]
pub struct ContactSheetBuilder {
    spec: GridSpec,
    targets: Vec<u64>,
    captured: Vec<Option<u64>>,
    next: usize,
    canvas: RgbImage,
}

impl ContactSheetBuilder {
    /// Tiles are sampled at the centres of `columns*rows` equal slices of
    /// `[start_ms, end_ms)`.
    pub fn new(spec: GridSpec, start_ms: u64, end_ms: u64) -> Self {
        let n = u64::from(spec.columns * spec.rows).max(1);
        let span = end_ms.saturating_sub(start_ms);
        let targets = (0..n)
            .map(|i| start_ms + span * (2 * i + 1) / (2 * n))
            .collect();
        Self {
            spec,
            targets,
            captured: vec![None; n as usize],
            next: 0,
            canvas: RgbImage::new(spec.columns * spec.tile_width, spec.rows * spec.tile_height),
        }
    }

    /// True once every tile is filled (callers may stop decoding).
    pub fn is_complete(&self) -> bool {
        self.next >= self.targets.len()
    }

    /// Offers a decoded frame; it is used for every target time it has reached.
    pub fn offer(&mut self, frame: &RgbFrame, t_ms: u64) {
        while self.next < self.targets.len() && t_ms >= self.targets[self.next] {
            self.paste(self.next, frame, t_ms);
            self.next += 1;
        }
    }

    fn paste(&mut self, slot: usize, frame: &RgbFrame, t_ms: u64) {
        let Some(img) = RgbImage::from_raw(frame.width, frame.height, frame.data.clone()) else {
            return;
        };
        let tile = if (img.width(), img.height()) == (self.spec.tile_width, self.spec.tile_height) {
            img
        } else {
            imageops::resize(
                &img,
                self.spec.tile_width,
                self.spec.tile_height,
                imageops::FilterType::Triangle,
            )
        };
        let col = slot as u32 % self.spec.columns;
        let row = slot as u32 / self.spec.columns;
        let (x0, y0) = (col * self.spec.tile_width, row * self.spec.tile_height);
        imageops::replace(&mut self.canvas, &tile, i64::from(x0), i64::from(y0));
        draw_label(&mut self.canvas, x0 + 2, y0 + 2, &format_ms(t_ms));
        self.captured[slot] = Some(t_ms);
    }

    /// Encodes the grid to `path` (JPEG, quality 82).
    pub fn finish(self, path: &Path) -> Result<ContactSheet> {
        let file = std::fs::File::create(path).map_err(|source| IndexError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut w = std::io::BufWriter::new(file);
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, 82)
            .encode_image(&self.canvas)?;
        Ok(ContactSheet {
            path: crate::probe::display_path(path),
            columns: self.spec.columns,
            rows: self.spec.rows,
            tile_width: self.spec.tile_width,
            tile_height: self.spec.tile_height,
            tile_ms: self
                .captured
                .iter()
                .zip(&self.targets)
                .map(|(c, t)| c.unwrap_or(*t))
                .collect(),
        })
    }
}

/// `m:ss.d` (or `h:mm:ss` past an hour).
pub fn format_ms(ms: u64) -> String {
    let s = ms / 1_000;
    if s >= 3_600 {
        format!("{}:{:02}:{:02}", s / 3_600, (s / 60) % 60, s % 60)
    } else {
        format!("{}:{:02}.{}", s / 60, s % 60, (ms % 1_000) / 100)
    }
}

/// 3x5 bitmap glyphs for `0-9 : .`, one `u8` per row (low 3 bits).
fn glyph(c: char) -> [u8; 5] {
    match c {
        '0' => [7, 5, 5, 5, 7],
        '1' => [2, 6, 2, 2, 7],
        '2' => [7, 1, 7, 4, 7],
        '3' => [7, 1, 7, 1, 7],
        '4' => [5, 5, 7, 1, 1],
        '5' => [7, 4, 7, 1, 7],
        '6' => [7, 4, 7, 5, 7],
        '7' => [7, 1, 1, 1, 1],
        '8' => [7, 5, 7, 5, 7],
        '9' => [7, 5, 7, 1, 7],
        ':' => [0, 2, 0, 2, 0],
        '.' => [0, 0, 0, 0, 2],
        _ => [0; 5],
    }
}

/// Draws white 2x-scaled text on a black box.
fn draw_label(img: &mut RgbImage, x0: u32, y0: u32, text: &str) {
    const S: u32 = 2;
    let w = text.chars().count() as u32 * 4 * S + S;
    let h = 5 * S + 2 * S;
    for y in y0..(y0 + h).min(img.height()) {
        for x in x0..(x0 + w).min(img.width()) {
            img.put_pixel(x, y, Rgb([0, 0, 0]));
        }
    }
    for (i, c) in text.chars().enumerate() {
        let g = glyph(c);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3_u32 {
                if bits & (4 >> col) == 0 {
                    continue;
                }
                for dy in 0..S {
                    for dx in 0..S {
                        let x = x0 + S + i as u32 * 4 * S + col * S + dx;
                        let y = y0 + S + row as u32 * S + dy;
                        if x < img.width() && y < img.height() {
                            img.put_pixel(x, y, Rgb([255, 255, 255]));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_slice_centres_and_writes_jpeg() {
        let spec = GridSpec {
            columns: 2,
            rows: 2,
            tile_width: 32,
            tile_height: 18,
        };
        let mut b = ContactSheetBuilder::new(spec, 0, 4_000);
        assert_eq!(b.targets, vec![500, 1_500, 2_500, 3_500]);
        for k in 0..100_u64 {
            let shade = (k * 2) as u8;
            b.offer(&RgbFrame::solid(64, 36, [shade, 0, 0]), k * 40);
        }
        assert!(b.is_complete());
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("grid.jpg");
        let sheet = b.finish(&p).unwrap();
        assert_eq!(sheet.tile_ms, vec![520, 1_520, 2_520, 3_520]);
        let img = image::open(&p).unwrap();
        assert_eq!((img.width(), img.height()), (64, 36));
    }

    #[test]
    fn formats_timestamps() {
        assert_eq!(format_ms(83_456), "1:23.4");
        assert_eq!(format_ms(3_725_000), "1:02:05");
    }
}
