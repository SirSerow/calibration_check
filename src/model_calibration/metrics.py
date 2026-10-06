"""Detection-weighted point metrics, image-level percentile bootstrap."""
import numpy as np

METRICS = ('ece', 'nll', 'brier', 'mean_confidence', 'precision')
EPS = 1e-12


def validate(scores, labels):
    p, y = np.asarray(scores, float), np.asarray(labels, float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError('Scores and labels must be parallel vectors')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Scores must be finite probabilities')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be binary')
    return p, y


def bin_indices(p, bins):
    if bins < 1:
        raise ValueError('Need at least one bin')
    return np.minimum((p * bins).astype(int), bins - 1)


def statistics(scores, labels, bins=15):
    p, y = validate(scores, labels)
    idx = bin_indices(p, bins)
    q = np.clip(p, EPS, 1 - EPS)
    return np.r_[len(p), p.sum(), y.sum(),
                 (-y * np.log(q) - (1-y) * np.log1p(-q)).sum(),
                 ((p-y)**2).sum(),
                 np.bincount(idx, minlength=bins),
                 np.bincount(idx, weights=p, minlength=bins),
                 np.bincount(idx, weights=y, minlength=bins)]


def from_statistics(s, bins=15):
    n = s[0]
    if n == 0:
        return dict.fromkeys(METRICS, None) | {'count': 0, 'bins': []}
    counts, ps, ys = np.split(s[5:], 3)
    table = [dict(bin=i, lower=i/bins, upper=(i+1)/bins, count=int(counts[i]),
                  confidence=float(ps[i]/counts[i]) if counts[i] else None,
                  precision=float(ys[i]/counts[i]) if counts[i] else None)
             for i in range(bins)]
    return dict(ece=float(np.abs(ps-ys).sum()/n), nll=float(s[3]/n),
                brier=float(s[4]/n), mean_confidence=float(s[1]/n),
                precision=float(s[2]/n), count=int(n), bins=table)


def metrics(scores, labels, bins=15):
    return from_statistics(statistics(scores, labels, bins), bins)


def image_statistics(rows, image_ids, bins=15, corrected=None):
    groups = {int(i): ([], []) for i in image_ids}
    for r in rows:
        if r['ignored']:
            continue
        p, y = groups[r['image_id']]
        p.append(r['score'] if corrected is None else corrected[r['detection_id']])
        y.append(r['correct'])
    return np.array([statistics(*groups[int(i)], bins) for i in image_ids])


def bootstrap(rows, image_ids, bins=15, samples=1000, seed=42, corrected=None):
    if samples < 1 or len(image_ids) == 0:
        raise ValueError('Bootstrap needs samples and images')
    base = image_statistics(rows, image_ids, bins)
    after = image_statistics(rows, image_ids, bins, corrected) if corrected is not None else None
    rng = np.random.default_rng(seed)
    draws, deltas = [], []
    for _ in range(samples):
        # Empty images remain in the sampling universe. Duplicate sampled images
        # contribute all their detections each time; labels are already matched.
        counts = np.bincount(rng.integers(0, len(image_ids), len(image_ids)), minlength=len(image_ids))
        m = from_statistics(counts @ base, bins)
        if m['count'] == 0:
            continue
        draws.append([m[k] for k in METRICS])
        if after is not None:
            a = from_statistics(counts @ after, bins)
            deltas.append([a[k]-m[k] for k in METRICS])
    def intervals(values):
        if not values:
            return dict.fromkeys(METRICS, None)
        lo, hi = np.percentile(values, [2.5, 97.5], axis=0)
        return {k: [float(l), float(h)] for k, l, h in zip(METRICS, lo, hi)}
    return {'ci': intervals(draws), 'paired_delta_ci': intervals(deltas) if after is not None else None,
            'valid_samples': len(draws), 'samples': samples, 'seed': seed}
