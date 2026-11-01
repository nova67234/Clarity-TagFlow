//! Spatial Scene layers: a picture and its depth map made into two layers that
//! move apart as the view moves — the picture with its own depth (the people
//! and things in front), and a background rebuilt behind them — so people and
//! objects slide over what's behind them instead of the whole picture bending
//! like a rubber sheet.
//!
//! After SLIDE (Jampani et al., ICCV 2021) — a foreground that turns
//! see-through where the depth jumps, over a background rebuilt only where it
//! can be uncovered — and 3D Photo Inpainting (Shih et al., CVPR 2020) —
//! depth edges sharpened first, short ones dropped as noise, and the rebuilt
//! region grown a few pixels past each edge so the mixed pixels there don't
//! leave a halo. Their settings are for pictures 1024 pixels across, which is
//! what the maps here are.
//!
//! Built once per picture off the UI thread (`crate::depth`'s worker) and
//! drawn by [`view`] over the viewer's own texture of the picture: a shader
//! that finds, for each pixel on screen, the nearest surface that lands there.

use std::sync::atomic::{AtomicU64, Ordering};

use image::imageops::FilterType;
use image::RgbImage;

use crate::depth::DepthMap;

pub mod view;

/// The maps' long edge (px). The depth model works at 518 anyway.
const WORK_EDGE: u32 = 1024;
/// A step in depth (normalized inverse depth, 0 far .. 1 near) this big
/// between pixels a pixel or two apart is an edge: one thing in front of
/// another.
const EDGE_STEP: f32 = 0.05;
/// Depth edges shorter than this (px at 1024) are noise (3D Photo Inpainting
/// drops those under 10).
const MIN_EDGE: f32 = 10.0;
/// How far the view moves things at most, as a share of the picture's short
/// side. The background is rebuilt this far in behind each edge.
pub const PARALLAX: f32 = 0.03;
/// The share of the picture nearer than the depth that stays still: the
/// subject (near) stays put while what's behind it moves, as in Apple's.
const FOCUS_QUANTILE: f32 = 0.85;

/// A picture made into layers, ready to draw.
pub struct Scene {
    /// Tells scenes apart (the GPU keeps one scene's textures).
    pub id: u64,
    /// The maps' size.
    pub w: u32,
    pub h: u32,
    /// Per pixel (RGBA8): the depth (0 far .. 255 near), how much the picture
    /// shows there (0 where it's torn at a depth edge), the background's
    /// depth, 255.
    pub maps: Vec<u8>,
    /// The background (RGBA8): the picture, rebuilt behind things in front.
    pub back: Vec<u8>,
    /// The depth that stays still as the view moves.
    pub focus: f32,
    /// The model's depth map, for the plain mesh where the shader can't run.
    pub depth: DepthMap,
}

/// Make `img` (with its `depth`) into a [`Scene`].
pub fn build(img: &RgbImage, depth: DepthMap) -> Scene {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let (w, h) = fit(img.width(), img.height(), WORK_EDGE);
    let small = image::imageops::resize(img, w, h, FilterType::Triangle);
    let rgb: Vec<[f32; 3]> =
        small.pixels().map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0]).collect();
    let (wu, hu) = (w as usize, h as usize);
    let mut d = vec![0.0f32; wu * hu];
    for y in 0..hu {
        for x in 0..wu {
            d[y * wu + x] = depth.sample((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
        }
    }
    // The papers' sizes are for 1024 across.
    let scale = w.max(h) as f32 / 1024.0;
    sharpen(&mut d, &rgb, wu, hu, scale);
    let edges = depth_edges(&d, wu, hu, (MIN_EDGE * scale).round() as usize);
    let reach = (PARALLAX * w.min(h) as f32).ceil() as usize + 2;
    let fade = (3.0 * scale).round().max(1.0) as usize;
    // Rebuilt a little past the fade, so it's rebuilt from colours clear of
    // the edge (3D Photo Inpainting grows its region 5 px).
    let (back_rgb, back_depth) = background(&d, &rgb, &edges, wu, hu, reach, fade + 2);
    let shows = visibility(&d, &edges, wu, hu, fade);

    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut maps = Vec::with_capacity(wu * hu * 4);
    let mut back = Vec::with_capacity(wu * hu * 4);
    for i in 0..wu * hu {
        maps.extend_from_slice(&[byte(d[i]), shows[i], byte(back_depth[i]), 255]);
        let c = back_rgb[i];
        back.extend_from_slice(&[byte(c[0]), byte(c[1]), byte(c[2]), 255]);
    }
    Scene {
        id: NEXT.fetch_add(1, Ordering::Relaxed),
        w,
        h,
        maps,
        back,
        focus: quantile(&d, FOCUS_QUANTILE),
        depth,
    }
}

/// `(w, h)` scaled down (never up) to fit `edge` across its long side.
fn fit(w: u32, h: u32, edge: u32) -> (u32, u32) {
    let long = w.max(h).max(1);
    if long <= edge {
        return (w.max(1), h.max(1));
    }
    let k = edge as f32 / long as f32;
    (((w as f32 * k).round() as u32).max(1), ((h as f32 * k).round() as u32).max(1))
}

/// The value `q` (0..1) of the way up `v` sorted.
fn quantile(v: &[f32], q: f32) -> f32 {
    if v.is_empty() {
        return 0.5;
    }
    // A histogram (the values are 0..1): no copy of the whole map to sort.
    let mut bins = [0usize; 256];
    for &x in v {
        bins[(x.clamp(0.0, 1.0) * 255.0) as usize] += 1;
    }
    let want = (q * v.len() as f32) as usize;
    let mut seen = 0;
    for (i, n) in bins.iter().enumerate() {
        seen += n;
        if seen > want {
            return i as f32 / 255.0;
        }
    }
    1.0
}

/// Sharpen the depth model's soft edges onto the picture's. Where the depth
/// ramps from one thing to another, each pixel takes the depth of the side
/// whose colour it shares — the near side's or the far side's — so a ramp
/// becomes a step at the colour edge (as 3D Photo Inpainting sharpens its
/// depth, and colour-guided "flying pixel" fixes put a mixed pixel on the
/// side it looks like). Smooth surfaces are left alone.
fn sharpen(d: &mut [f32], rgb: &[[f32; 3]], w: usize, h: usize, scale: f32) {
    let r = (6.0 * scale).round().max(2.0) as i32;
    for _ in 0..2 {
        let src = d.to_vec();
        // The nearest and farthest depth around each pixel.
        let lo = min_filter(&src, w, h, r as usize);
        let negated: Vec<f32> = src.iter().map(|v| -v).collect();
        let hi: Vec<f32> = min_filter(&negated, w, h, r as usize).into_iter().map(|v| -v).collect();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (lo, hi) = (lo[i], hi[i]);
                // Only on a ramp (not the flat either side of it) between two
                // things (not a gentle slope).
                if hi - lo < 2.0 * EDGE_STEP || !on_a_ramp(&src, x, y, w, h) {
                    continue;
                }
                // Each side: the pixels clearly on it (every other one is
                // plenty), their depth and colour.
                let quarter = (hi - lo) * 0.25;
                let (mut far, mut near) = (Side::default(), Side::default());
                for dy in (-r..=r).step_by(2) {
                    for dx in (-r..=r).step_by(2) {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx < 0 || ny < 0 || nx as usize >= w || ny as usize >= h {
                            continue;
                        }
                        let j = ny as usize * w + nx as usize;
                        if src[j] < lo + quarter {
                            far.add(src[j], rgb[j]);
                        } else if src[j] > hi - quarter {
                            near.add(src[j], rgb[j]);
                        }
                    }
                }
                if far.n == 0.0 || near.n == 0.0 {
                    continue;
                }
                let (to_far, to_near) = (far.distance(rgb[i]), near.distance(rgb[i]));
                // Where the colours can't tell, the depth itself does.
                let is_near = if (to_far - to_near).abs() < 0.01 { src[i] > (lo + hi) * 0.5 } else { to_near < to_far };
                d[i] = if is_near { near.depth() } else { far.depth() };
            }
        }
    }
}

/// Whether the depth changes noticeably at (`x`, `y`): the pixel is on a
/// ramp, not on the flat either side of one.
fn on_a_ramp(d: &[f32], x: usize, y: usize, w: usize, h: usize) -> bool {
    let c = d[y * w + x];
    neighbours4(x, y, w, h).any(|(nx, ny)| (c - d[ny * w + nx]).abs() > EDGE_STEP * 0.15)
}

/// One side of a depth edge: the sums of its pixels' depths and colours.
#[derive(Default)]
struct Side {
    n: f32,
    depth: f32,
    rgb: [f32; 3],
}

impl Side {
    fn add(&mut self, depth: f32, c: [f32; 3]) {
        self.n += 1.0;
        self.depth += depth;
        for (sum, v) in self.rgb.iter_mut().zip(c) {
            *sum += v;
        }
    }

    fn depth(&self) -> f32 {
        self.depth / self.n.max(1.0)
    }

    /// How far `c` is from this side's mean colour (squared).
    fn distance(&self, c: [f32; 3]) -> f32 {
        let n = self.n.max(1.0);
        (0..3).map(|k| (c[k] - self.rgb[k] / n).powi(2)).sum()
    }
}

/// The pixels just behind a depth edge: on the far side of a step of
/// [`EDGE_STEP`] to a pixel one or two away (with the pixel between, which
/// mixes the two). There the picture turns see-through, so what's in front
/// keeps its whole outline (thin things too: handlebars, hair) and comes away
/// clean from what's behind. Edges shorter than `min_len` are dropped.
fn depth_edges(d: &[f32], w: usize, h: usize, min_len: usize) -> Vec<bool> {
    let mut e = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            for (dx, dy) in [(1usize, 0usize), (0, 1), (2, 0), (0, 2)] {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= w || ny >= h {
                    continue;
                }
                let j = ny * w + nx;
                let step = d[i] - d[j];
                if step.abs() <= EDGE_STEP {
                    continue;
                }
                e[if step > 0.0 { j } else { i }] = true;
                // The pixel between, if it's in between (not the near side).
                if dx == 2 || dy == 2 {
                    let m = (y + dy / 2) * w + x + dx / 2;
                    if d[m] < d[i].max(d[j]) - EDGE_STEP * 0.5 {
                        e[m] = true;
                    }
                }
            }
        }
    }
    // An edge here is about two pixels thick.
    drop_small(&mut e, w, h, 2 * min_len);
    e
}

/// How much the picture shows at each pixel (0..255): none at the far side
/// of a depth edge, fading back in over `fade` pixels behind it. Just behind
/// an edge the picture still has some of what's in front mixed in (a colour
/// edge is a pixel or two wide, and the depth's edge never falls exactly on
/// it); there the rebuilt background shows through instead, so nothing of
/// the thing in front is left behind as it moves (SLIDE's soft visibility).
/// What's in front keeps its outline.
fn visibility(d: &[f32], edges: &[bool], w: usize, h: usize, fade: usize) -> Vec<u8> {
    let (dist, behind) = nearest_edge(d, edges, w, h, fade);
    (0..w * h)
        .map(|i| {
            if dist[i] == u32::MAX || d[i] - behind[i] > EDGE_STEP * 0.5 {
                255
            } else {
                (255.0 * dist[i] as f32 / (fade + 1) as f32).round() as u8
            }
        })
        .collect()
}

/// Clear the 8-connected groups of `set` smaller than `min` pixels.
fn drop_small(set: &mut [bool], w: usize, h: usize, min: usize) {
    let mut seen = vec![false; set.len()];
    let mut group = Vec::new();
    let mut stack = Vec::new();
    for start in 0..set.len() {
        if !set[start] || seen[start] {
            continue;
        }
        group.clear();
        stack.push(start);
        seen[start] = true;
        while let Some(i) = stack.pop() {
            group.push(i);
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let j = ny as usize * w + nx as usize;
                        if set[j] && !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        if group.len() < min {
            for &i in &group {
                set[i] = false;
            }
        }
    }
}

/// The background layer: the picture with whatever is in front of something
/// farther across a depth edge (`edges`, the far sides) taken away as far
/// in as the view can move it — `reach` for a step from farthest to
/// nearest, less for a smaller step — and rebuilt from the farther side,
/// plus `halo` pixels past each edge (their colours mix in what's in front).
/// Its colours and depth.
fn background(
    d: &[f32],
    rgb: &[[f32; 3]],
    edges: &[bool],
    w: usize,
    h: usize,
    reach: usize,
    halo: usize,
) -> (Vec<[f32; 3]>, Vec<f32>) {
    let n = w * h;
    // How far each pixel is from the nearest edge, and the depth behind that
    // edge: what a pixel in front of it could uncover.
    let (dist, behind) = nearest_edge(d, edges, w, h, reach);
    let in_front = |i: usize| {
        let step = d[i] - behind[i];
        step > EDGE_STEP && dist[i] as f32 <= step * reach as f32 + 1.0
    };
    let mut hole: Vec<bool> = (0..n).map(in_front).collect();
    // What it rebuilds lies at the depth behind the edge; the halo at its own.
    let far: Vec<f32> = (0..n).map(|i| if hole[i] { behind[i] } else { d[i] }).collect();
    let mut ring: Vec<usize> = (0..n).filter(|&i| hole[i]).collect();
    for _ in 0..halo {
        let mut next = Vec::new();
        for &i in &ring {
            let (x, y) = (i % w, i / w);
            for (nx, ny) in neighbours4(x, y, w, h) {
                let j = ny * w + nx;
                if !hole[j] {
                    hole[j] = true;
                    next.push(j);
                }
            }
        }
        ring = next;
    }
    let mut col = rgb.to_vec();
    let mut depth = d.to_vec();

    // First guess: ring by ring from the edge of the rebuilt region inward,
    // colour and depth from the neighbours no nearer than what a pixel
    // uncovers — the far side, never the thing in front.
    let mut todo: Vec<usize> = (0..n).filter(|&i| hole[i]).collect();
    let mut done: Vec<bool> = hole.iter().map(|&h| !h).collect();
    for strict in [true, false] {
        loop {
            let mut fills: Vec<(usize, [f32; 3], f32)> = Vec::new();
            for &i in &todo {
                let (x, y) = (i % w, i / w);
                let (mut sum, mut dsum, mut wsum) = ([0.0f32; 3], 0.0f32, 0.0f32);
                for (nx, ny, wt) in neighbours8(x, y, w, h) {
                    let j = ny * w + nx;
                    if done[j] && (!strict || depth[j] <= far[i] + EDGE_STEP) {
                        for k in 0..3 {
                            sum[k] += col[j][k] * wt;
                        }
                        dsum += depth[j] * wt;
                        wsum += wt;
                    }
                }
                if wsum > 0.0 {
                    fills.push((i, [sum[0] / wsum, sum[1] / wsum, sum[2] / wsum], dsum / wsum));
                }
            }
            if fills.is_empty() {
                break;
            }
            for &(i, c, z) in &fills {
                col[i] = c;
                depth[i] = z.min(d[i]);
                done[i] = true;
            }
            todo.retain(|&i| !done[i]);
        }
    }
    // Then smooth it out (the ring fill leaves blocks and streaks): each
    // rebuilt pixel the mean of its neighbours on the far side — the
    // smoothest fill that meets the far side's colours, as heat would spread
    // (Jacobi steps of Laplace's equation; what's in front is no boundary).
    let rebuilt: Vec<usize> = (0..n).filter(|&i| hole[i]).collect();
    for _ in 0..(2 * reach).max(12) {
        let (src, zsrc) = (col.clone(), depth.clone());
        for &i in &rebuilt {
            let (x, y) = (i % w, i / w);
            let (mut sum, mut dsum, mut wsum) = ([0.0f32; 3], 0.0f32, 0.0f32);
            for (nx, ny) in neighbours4(x, y, w, h) {
                let j = ny * w + nx;
                if zsrc[j] <= far[i] + EDGE_STEP {
                    for k in 0..3 {
                        sum[k] += src[j][k];
                    }
                    dsum += zsrc[j];
                    wsum += 1.0;
                }
            }
            if wsum > 0.0 {
                col[i] = [sum[0] / wsum, sum[1] / wsum, sum[2] / wsum];
                depth[i] = (dsum / wsum).min(d[i]);
            }
        }
    }
    (col, depth)
}

/// For each pixel within `reach` (city-block) of an edge pixel: how far, and
/// the depth of that edge pixel (the far side of the edge). Pixels out of
/// reach get `u32::MAX` and depth 1 (nothing behind them).
fn nearest_edge(d: &[f32], edges: &[bool], w: usize, h: usize, reach: usize) -> (Vec<u32>, Vec<f32>) {
    let mut dist = vec![u32::MAX; w * h];
    let mut behind = vec![1.0f32; w * h];
    let mut ring: Vec<usize> = (0..w * h).filter(|&i| edges[i]).collect();
    for &i in &ring {
        dist[i] = 0;
        behind[i] = d[i];
    }
    for step in 1..=reach as u32 {
        let mut next = Vec::new();
        for &i in &ring {
            let (x, y) = (i % w, i / w);
            for (nx, ny) in neighbours4(x, y, w, h) {
                let j = ny * w + nx;
                if dist[j] == u32::MAX {
                    dist[j] = step;
                    behind[j] = behind[i];
                    next.push(j);
                }
            }
        }
        ring = next;
    }
    (dist, behind)
}

fn neighbours4(x: usize, y: usize, w: usize, h: usize) -> impl Iterator<Item = (usize, usize)> {
    [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)].into_iter().filter_map(move |(dx, dy)| {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        (nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h).then_some((nx as usize, ny as usize))
    })
}

/// The 8 neighbours, weighted (diagonals a little less).
fn neighbours8(x: usize, y: usize, w: usize, h: usize) -> impl Iterator<Item = (usize, usize, f32)> {
    const AROUND: [(i32, i32, f32); 8] =
        [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 0.7), (1, -1, 0.7), (-1, 1, 0.7), (1, 1, 0.7)];
    AROUND.into_iter().filter_map(move |(dx, dy, wt)| {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        (nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h).then_some((nx as usize, ny as usize, wt))
    })
}

/// The least value within `r` (a square) of each pixel: rows, then columns,
/// each in one pass (van Herk / Gil–Werman).
fn min_filter(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut rows = vec![0.0f32; v.len()];
    let mut line = Vec::new();
    for y in 0..h {
        line.clear();
        line.extend_from_slice(&v[y * w..(y + 1) * w]);
        let out = min_1d(&line, r);
        rows[y * w..(y + 1) * w].copy_from_slice(&out);
    }
    let mut out = vec![0.0f32; v.len()];
    for x in 0..w {
        line.clear();
        line.extend((0..h).map(|y| rows[y * w + x]));
        for (y, m) in min_1d(&line, r).into_iter().enumerate() {
            out[y * w + x] = m;
        }
    }
    out
}

/// The least value within `r` of each element of `v`.
fn min_1d(v: &[f32], r: usize) -> Vec<f32> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    // Padded so every window is a whole 2r + 1: running minimums from the
    // left and the right within blocks of that size then give any window's
    // minimum from just its two ends.
    let k = 2 * r + 1;
    let mut p = vec![f32::INFINITY; n + 2 * r];
    p[r..r + n].copy_from_slice(v);
    let m = p.len();
    let mut left = vec![0.0f32; m];
    let mut right = vec![0.0f32; m];
    for i in 0..m {
        left[i] = if i % k == 0 { p[i] } else { left[i - 1].min(p[i]) };
    }
    for i in (0..m).rev() {
        right[i] = if i % k == k - 1 || i == m - 1 { p[i] } else { right[i + 1].min(p[i]) };
    }
    (0..n).map(|i| right[i].min(left[i + 2 * r])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A near square (depth `near`, colour `fg`) on a far ground (`far`, `bg`).
    fn square(w: usize, h: usize, near: f32, far: f32) -> (Vec<f32>, Vec<[f32; 3]>) {
        let inside = |x: usize, y: usize| (w / 4..3 * w / 4).contains(&x) && (h / 4..3 * h / 4).contains(&y);
        let mut d = vec![far; w * h];
        let mut c = vec![[0.1, 0.3, 0.9]; w * h];
        for y in 0..h {
            for x in 0..w {
                if inside(x, y) {
                    d[y * w + x] = near;
                    c[y * w + x] = [0.9, 0.2, 0.1];
                }
            }
        }
        (d, c)
    }

    /// The model's soft edge becomes a step where the colours change.
    #[test]
    fn sharpening_snaps_a_soft_edge_onto_the_colour_edge() {
        let (w, h) = (64, 8);
        // Colour changes at x = 32; the depth ramps from 0.2 to 0.8 over x = 26..38.
        let rgb: Vec<[f32; 3]> = (0..w * h).map(|i| if i % w < 32 { [0.1, 0.3, 0.9] } else { [0.9, 0.2, 0.1] }).collect();
        let ramp = |x: usize| (0.2 + 0.6 * ((x as f32 - 26.0) / 12.0).clamp(0.0, 1.0)).clamp(0.2, 0.8);
        let mut d: Vec<f32> = (0..w * h).map(|i| ramp(i % w)).collect();
        sharpen(&mut d, &rgb, w, h, 1.0);
        let row = |x: usize| d[4 * w + x];
        assert!(row(29) < 0.3 && row(31) < 0.35, "far side stays far: {} {}", row(29), row(31));
        assert!(row(32) > 0.65 && row(34) > 0.7, "near side snaps near: {} {}", row(32), row(34));
    }

    /// A step's far side is an edge (its near side keeps its outline); a
    /// gentle slope and a tiny speck aren't.
    #[test]
    fn depth_edges_are_steps_not_slopes_or_specks() {
        let (w, h) = (40, 40);
        let (mut d, _) = square(w, h, 0.8, 0.2);
        // A slope across the ground, and a 2-pixel speck.
        for y in 0..h {
            for x in 0..w {
                if d[y * w + x] < 0.5 {
                    d[y * w + x] = 0.2 + 0.004 * x as f32;
                }
            }
        }
        d[2 * w + 2] = 0.9;
        d[2 * w + 3] = 0.9;
        let e = depth_edges(&d, w, h, 10);
        assert!(e[20 * w + 9], "the ground just left of the square's left side");
        assert!(!e[20 * w + 10], "the square itself keeps its outline");
        assert!(!e[20 * w + 4], "the slope isn't an edge");
        assert!(!e[2 * w + 2], "nor the speck");
        assert!(!e[20 * w + 20], "nor the square's middle");
    }

    /// Behind a near thing, the background takes the far side's colour and
    /// depth — none of the near thing's colour — as far in as the view can
    /// move it; farther in, it's left alone.
    #[test]
    fn the_background_is_rebuilt_behind_the_edge_from_the_far_side() {
        let (w, h) = (80, 80);
        let (d, rgb) = square(w, h, 0.9, 0.2);
        let edges = depth_edges(&d, w, h, 10);
        let (col, depth) = background(&d, &rgb, &edges, w, h, 6, 2);
        // Just inside the square's left side (x = 20), as far in as the view
        // can move it (0.7 of the way from far to near: 0.7 × 6 + 1 px):
        // rebuilt, blue.
        for x in 20..25 {
            let c = col[40 * w + x];
            assert!(c[2] > 0.8 && c[0] < 0.2, "x {x}: {c:?}");
            assert!(depth[40 * w + x] < 0.3, "x {x}: {}", depth[40 * w + x]);
        }
        // Farther in is out of reach: still the square.
        assert_eq!(col[40 * w + 27], [0.9, 0.2, 0.1]);
        assert_eq!(depth[40 * w + 27], 0.9);
        // The ground just outside the edge is rebuilt (the halo), still blue.
        assert!(col[40 * w + 19][2] > 0.8);
    }

    /// A floor coming nearer toward the bottom isn't in front of anything:
    /// nothing of it is rebuilt.
    #[test]
    fn a_slope_isnt_rebuilt() {
        let (w, h) = (60, 60);
        let d: Vec<f32> = (0..w * h).map(|i| 0.1 + 0.8 * (i / w) as f32 / h as f32).collect();
        let rgb: Vec<[f32; 3]> = (0..w * h).map(|i| [(i % 7) as f32 / 7.0, 0.5, 0.5]).collect();
        let edges = depth_edges(&d, w, h, 10);
        assert!(edges.iter().all(|&e| !e));
        let (col, depth) = background(&d, &rgb, &edges, w, h, 12, 3);
        assert_eq!(col, rgb);
        assert_eq!(depth, d);
    }

    /// The whole scene: torn only along the square's outline, the
    /// background rebuilt behind it, the square what stays still.
    #[test]
    fn a_scene_tears_at_the_outline_only() {
        let (w, h) = (96u32, 96u32);
        let mut img = RgbImage::from_pixel(w, h, image::Rgb([30, 80, 230]));
        let mut data = vec![0.2f32; 64 * 64];
        for y in 16..48 {
            for x in 16..48 {
                data[y * 64 + x] = 0.9;
            }
        }
        for y in 24..72 {
            for x in 24..72 {
                img.put_pixel(x, y, image::Rgb([230, 50, 30]));
            }
        }
        let s = build(&img, DepthMap { w: 64, h: 64, data });
        assert_eq!((s.w, s.h), (96, 96));
        let at = |x: usize, y: usize| &s.maps[(y * 96 + x) * 4..(y * 96 + x) * 4 + 4];
        assert_eq!(at(48, 48)[1], 255, "the square's middle shows");
        assert_eq!(at(5, 5)[1], 255, "so does the far ground");
        assert!((20..28).any(|x| at(x, 48)[1] == 0), "torn at the square's left side");
        assert!(s.focus > 0.5, "the square stays still: {}", s.focus);
        assert!(at(26, 48)[2] < 100, "the background there is far: {}", at(26, 48)[2]);
        assert_eq!(s.back.len(), 96 * 96 * 4);
    }

    /// Behind an edge the picture fades back in over a few pixels; in front
    /// of it, it shows in full.
    #[test]
    fn the_picture_fades_in_behind_an_edge() {
        let (w, h) = (40, 40);
        let (d, _) = square(w, h, 0.8, 0.2);
        let edges = depth_edges(&d, w, h, 10);
        let v = visibility(&d, &edges, w, h, 3);
        let at = |x: usize| v[20 * w + x];
        // The square starts at x = 10; the edge is the two pixels left of it.
        assert_eq!((at(8), at(9)), (0, 0));
        assert!(0 < at(7) && at(7) < at(6) && at(6) < at(5) && at(5) < 255, "{:?}", (4..11).map(at).collect::<Vec<_>>());
        assert_eq!(at(3), 255);
        assert_eq!(at(10), 255, "the square keeps its outline");
        assert_eq!(at(20), 255);
    }

    #[test]
    fn min_filter_finds_the_least_within_reach() {
        let v: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(min_1d(&v, 2), vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let v2 = [5.0, 1.0, 5.0, 5.0, 5.0, 5.0, 0.5];
        assert_eq!(min_1d(&v2, 1), vec![1.0, 1.0, 1.0, 5.0, 5.0, 0.5, 0.5]);
    }

    #[test]
    fn fit_scales_down_only() {
        assert_eq!(fit(4000, 3000, 1024), (1024, 768));
        assert_eq!(fit(500, 300, 1024), (500, 300));
        assert_eq!(fit(300, 1200, 1024), (256, 1024));
    }
}
