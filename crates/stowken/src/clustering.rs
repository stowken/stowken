//! In-process k-means++ clustering for L2-normalized vectors.
//!
//! Distance is `1 - dot(a, b)` (cosine distance for unit vectors).
//! Centroids are re-normalized after each iteration to stay on the unit
//! sphere — so cluster assignment can keep using a dot product.
//!
//! Used for `Stowken::cluster_conversations` and `Stowken::find_outliers`.

/// Tunables for `kmeans`.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub k: usize,
    pub max_iterations: usize,
    pub seed: u64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 8,
            max_iterations: 100,
            seed: 0xC0FFEE,
        }
    }
}

/// Result of a `kmeans` run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// One unit vector per cluster.
    pub centroids: Vec<Vec<f32>>,
    /// Cluster index per input point (parallel to the input slice).
    pub assignments: Vec<u32>,
}

/// k-means++ initialization for cosine distance, followed by Lloyd's
/// iteration. Centroids are re-normalized between iterations.
///
/// Empty clusters are reseeded from the point farthest from any current
/// centroid — this prevents the algorithm collapsing to fewer than `k`
/// clusters on degenerate inputs.
pub fn kmeans(points: &[Vec<f32>], config: KMeansConfig) -> KMeansResult {
    let k = config.k;
    if points.is_empty() || k == 0 {
        return KMeansResult {
            centroids: vec![],
            assignments: vec![],
        };
    }
    let dim = points[0].len();
    let n = points.len();
    let effective_k = k.min(n);

    let mut rng = SplitMix64::new(config.seed);

    // ── k-means++ seeding ──
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(effective_k);
    let first = (rng.next_u64() % n as u64) as usize;
    centroids.push(normalize(&points[first]));

    while centroids.len() < effective_k {
        // For each point, distance² to nearest centroid.
        let dists: Vec<f32> = points
            .iter()
            .map(|p| {
                centroids
                    .iter()
                    .map(|c| {
                        let sim: f32 = c.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
                        let d = (1.0 - sim).max(0.0);
                        d * d
                    })
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            // All points coincide with an existing centroid — pad with the
            // first point until we hit k. Degenerate but well-defined.
            centroids.push(centroids[0].clone());
            continue;
        }
        let target = rng.next_f32() * total;
        let mut acc = 0.0;
        let mut chosen = n - 1;
        for (i, d) in dists.iter().enumerate() {
            acc += *d;
            if acc >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(normalize(&points[chosen]));
    }

    // ── Lloyd's iteration ──
    let mut assignments = vec![0u32; n];
    for _iter in 0..config.max_iterations {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let mut best_c = 0u32;
            let mut best_sim = f32::MIN;
            for (ci, c) in centroids.iter().enumerate() {
                let sim: f32 = c.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
                if sim > best_sim {
                    best_sim = sim;
                    best_c = ci as u32;
                }
            }
            if assignments[i] != best_c {
                assignments[i] = best_c;
                changed = true;
            }
        }

        // Recompute centroids as the (normalized) mean of their members.
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; effective_k];
        let mut counts: Vec<u32> = vec![0; effective_k];
        for (i, p) in points.iter().enumerate() {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for (s, x) in sums[c].iter_mut().zip(p.iter()) {
                *s += *x;
            }
        }

        let mut new_centroids: Vec<Vec<f32>> = Vec::with_capacity(effective_k);
        for ci in 0..effective_k {
            if counts[ci] == 0 {
                // Empty cluster — reseed from the point farthest from all
                // existing centroids (highest min-distance).
                let farthest = points
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let min_sim: f32 = new_centroids
                            .iter()
                            .chain(centroids.iter())
                            .map(|c| c.iter().zip(p.iter()).map(|(a, b)| a * b).sum::<f32>())
                            .fold(f32::INFINITY, f32::min);
                        (i, 1.0 - min_sim)
                    })
                    .max_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                new_centroids.push(normalize(&points[farthest]));
                continue;
            }
            let inv = 1.0 / counts[ci] as f32;
            let mean: Vec<f32> = sums[ci].iter().map(|x| *x * inv).collect();
            new_centroids.push(normalize(&mean));
        }
        centroids = new_centroids;

        if !changed {
            break;
        }
    }

    KMeansResult {
        centroids,
        assignments,
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Tiny deterministic PRNG so k-means seeding is reproducible. SplitMix64
/// is a 64-bit mixing function; good enough for picking seed indices.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero state that would make the stream constant.
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits as a uniform float in [0, 1).
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(v: Vec<f32>) -> Vec<f32> {
        normalize(&v)
    }

    #[test]
    fn kmeans_separates_two_clusters() {
        // Cluster A around [1, 0], cluster B around [0, 1] — orthogonal.
        let points: Vec<Vec<f32>> = vec![
            norm(vec![1.0, 0.05]),
            norm(vec![0.95, 0.0]),
            norm(vec![1.0, 0.1]),
            norm(vec![0.0, 1.0]),
            norm(vec![0.05, 0.95]),
            norm(vec![0.0, 0.9]),
        ];
        let result = kmeans(
            &points,
            KMeansConfig {
                k: 2,
                max_iterations: 50,
                seed: 42,
            },
        );
        // Group A indices and group B indices should each share a cluster id.
        let a_clusters: std::collections::HashSet<_> =
            result.assignments[0..3].iter().copied().collect();
        let b_clusters: std::collections::HashSet<_> =
            result.assignments[3..6].iter().copied().collect();
        assert_eq!(a_clusters.len(), 1, "A points spread across clusters");
        assert_eq!(b_clusters.len(), 1, "B points spread across clusters");
        assert_ne!(
            a_clusters, b_clusters,
            "A and B should be in different clusters"
        );
    }

    #[test]
    fn empty_input_returns_empty_result() {
        let r = kmeans(&[], KMeansConfig::default());
        assert!(r.centroids.is_empty());
        assert!(r.assignments.is_empty());
    }

    #[test]
    fn k_larger_than_n_clamps() {
        let points = vec![norm(vec![1.0, 0.0]), norm(vec![0.0, 1.0])];
        let r = kmeans(
            &points,
            KMeansConfig {
                k: 10,
                ..KMeansConfig::default()
            },
        );
        assert_eq!(r.centroids.len(), 2);
        assert_eq!(r.assignments.len(), 2);
    }
}
