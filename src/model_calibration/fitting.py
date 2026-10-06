from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .metrics import EPS, validate


@dataclass(frozen=True)
class Correction:
    method: str
    a: float
    b: float = 0.0

    def __post_init__(self):
        if self.method not in ('temperature', 'platt') or not np.isfinite([self.a,self.b]).all() or self.a <= 0:
            raise ValueError('Correction must have a finite positive slope')

    def logits(self, scores):
        p = np.clip(np.asarray(scores, float), EPS, 1-EPS)
        return self.a*(np.log(p)-np.log1p(-p)) + self.b

    def apply(self, scores):
        return expit(self.logits(scores))

    def to_dict(self):
        return asdict(self) | ({'T': 1/self.a} if self.method == 'temperature' else {})


def fit(scores, labels, method):
    p, y = validate(scores, labels)
    if len(p) == 0 or len(np.unique(y)) < 2:
        raise ValueError('Fitting requires detections of both correctness labels')
    z = np.log(np.clip(p, EPS, 1-EPS))-np.log1p(-np.clip(p, EPS, 1-EPS))
    def objective(theta):
        a = np.exp(theta[0])
        b = theta[1] if method == 'platt' else 0.0
        logits = a*z+b
        residual = expit(logits)-y
        gradient = [np.mean(residual*z)*a]
        if method == 'platt':
            gradient.append(np.mean(residual))
        return np.mean(np.logaddexp(0,logits)-y*logits), np.array(gradient)
    if method not in ('temperature', 'platt'):
        raise ValueError(method)
    result = minimize(objective, np.zeros(2 if method=='platt' else 1), jac=True,
                      method='L-BFGS-B', bounds=[(-10,10)]+([(-30,30)] if method=='platt' else []),
                      options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 2000})
    if not result.success:
        raise RuntimeError(f'Calibration optimizer failed: {result.message}')
    return Correction(method, float(np.exp(result.x[0])), float(result.x[1]) if method=='platt' else 0.0)


def grouped_validation(rows, image_ids, folds=5, seed=42, tolerance=1e-9):
    ids = np.array(sorted(image_ids))
    if len(ids) < folds:
        raise ValueError('Need at least one image per fold')
    np.random.default_rng(seed).shuffle(ids)
    fold_ids = [list(map(int, x)) for x in np.array_split(ids, folds)]
    outcomes = {}
    for method in ('temperature','platt'):
        total, count, losses = 0., 0, []
        for heldout in fold_ids:
            valset = set(heldout)
            train = [r for r in rows if r['image_id'] not in valset]
            val = [r for r in rows if r['image_id'] in valset]
            c = fit([r['score'] for r in train], [r['correct'] for r in train], method)
            if not val:
                raise ValueError('A validation fold has no eligible detections')
            logits = c.logits([r['score'] for r in val])
            y = np.array([r['correct'] for r in val])
            loss = float((np.logaddexp(0,logits)-y*logits).sum())
            total += loss
            count += len(val)
            losses.append({'nll': loss/len(val), 'count': len(val)})
        outcomes[method] = {'nll': total/count, 'folds': losses}
    winner = 'platt' if outcomes['platt']['nll'] < outcomes['temperature']['nll']-tolerance else 'temperature'
    fitted = {m: fit([r['score'] for r in rows], [r['correct'] for r in rows], m).to_dict() for m in outcomes}
    return dict(selected_method=winner, validation=outcomes, fold_image_ids=fold_ids, parameters=fitted)
