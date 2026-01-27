import warnings
import numpy as np
from scipy import stats, optimize


def _check_counts(counts) -> np.ndarray:
    """
    Validate counts are non-negative integers.
    Returns a 1D numpy array of dtype int.
    Raises ValueError on invalid inputs.
    """
    x = np.atleast_1d(np.asarray(counts))
    if x.size == 0:
        raise ValueError("Empty offspring array.")
    if np.any(~np.isfinite(x)):
        raise ValueError("Counts contain non-finite values.")
    if np.any(x < 0):
        raise ValueError("Counts must be non-negative.")
    # Ensure integer-valued: allow float inputs if they are integers numerically
    if np.any(np.abs(x - np.rint(x)) > 1e-12):
        raise ValueError("Counts must be integer-valued.")
    return np.rint(x).astype(int)


def _moments_nb(counts: np.ndarray, tol: float = 1e-12) -> tuple[float, float]:
    """
    Method of moments for NB under mean–dispersion parameterisation:
        mean = R, var = R + R^2/k  (k > 0)
    => k = R^2 / (var - R)
    Returns (R_mom, k_mom) where k_mom may be np.nan if var <= R.
    """
    x = counts
    R = float(np.mean(x))
    n = x.size

    if n < 2:
        warnings.warn(
            "Method-of-moments: n < 2, cannot estimate dispersion k; returning k = NaN.",
            RuntimeWarning
        )
        return R, np.nan

    var = float(np.var(x, ddof=1))

    if var <= R + tol:
        # Under-dispersed or Poisson boundary; NB dispersion not identifiable
        warnings.warn(
            "Method-of-moments: sample variance <= mean; NB dispersion k not identifiable. "
            "Returning k = NaN (treat as Poisson-like).",
            RuntimeWarning
        )
        return R, np.nan

    k = (R ** 2) / (var - R)
    if not np.isfinite(k) or k <= 0:
        warnings.warn("Method-of-moments: computed invalid k; returning k = NaN.", RuntimeWarning)
        k = np.nan
    return R, float(k)


def _fit_nb_mle(counts: np.ndarray, eps: float = 1e-9) -> tuple[float, float]:
    """
    Fit Negative Binomial by MLE under mean–dispersion parameterisation:
        mean = R, variance = R + R^2 / k   (k > 0)
    scipy.stats.nbinom uses parameters (r=k, p=k/(k+R)).

    Returns (R_hat, k_hat). Raises RuntimeError if optimisation fails.
    """
    x = np.asarray(counts, dtype=int)
    if x.size == 0:
        raise ValueError("Empty offspring array.")
    if np.any(x < 0):
        raise ValueError("Counts must be non-negative integers.")

    # Initialise with method-of-moments
    Rt = max(x.mean(), eps)
    var = x.var(ddof=1) if x.size > 1 else Rt + 1.0
    k0 = max((Rt ** 2) / max(var - Rt, eps), 0.3)

    def nll(params):
        R, logk = params
        if R <= 0:
            return np.inf
        k = np.exp(logk)
        p = k / (k + R)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll = stats.nbinom.logpmf(x, k, p)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -np.sum(ll)

    res = optimize.minimize(
        nll,
        x0=np.array([Rt, np.log(k0)]),
        method="L-BFGS-B",
        bounds=[(1e-12, None), (np.log(1e-8), np.log(1e8))]
    )
    if not res.success:
        raise RuntimeError(f"NB MLE failed: {res.message}")

    R_hat = float(res.x[0])
    k_hat = float(np.exp(res.x[1]))
    return R_hat, k_hat


def _fit_nb_safely(counts: np.ndarray, tol: float = 1e-12) -> tuple[float, float, str, str]:
    """
    Safe NB fitting: prefer MLE when identifiable (var > mean); otherwise use MoM.
    Returns (R_hat, k_hat, method, notes). k_hat may be NaN if not identifiable.
    """
    x = np.asarray(counts, dtype=int)
    n = x.size
    R_emp = float(np.mean(x))

    # Handle trivial cases
    if n < 2:
        R_mom, k_mom = _moments_nb(x, tol=tol)
        return R_mom, k_mom, "moments", "n<2"

    var = float(np.var(x, ddof=1))

    if var <= R_emp + tol:
        # NB dispersion not identifiable; MoM returns NaN for k
        R_mom, k_mom = _moments_nb(x, tol=tol)
        return R_mom, k_mom, "moments", "variance<=mean"

    # Try MLE, fall back to MoM if optimisation fails
    try:
        R_mle, k_mle = _fit_nb_mle(x)
        return R_mle, k_mle, "mle", ""
    except Exception as e:
        warnings.warn(f"MLE failed ({e}); falling back to method-of-moments.", RuntimeWarning)
        R_mom, k_mom = _moments_nb(x, tol=tol)
        return R_mom, k_mom, "moments-fallback", "mle-failed"


def _prop_for_80_percent(counts: np.ndarray) -> float:
    """
    Minimum fraction of individuals accounting for 80% of transmissions.
    """
    x = np.asarray(counts, dtype=int)
    n = x.size
    if n == 0:
        return np.nan
    tot = x.sum()
    if tot == 0:
        return 1.0
    order = np.sort(x)[::-1]
    cum = np.cumsum(order)
    idx = np.searchsorted(cum, 0.8 * tot, side="left")
    return (idx + 1) / n


def _bootstrap_nb(counts: np.ndarray, B: int = 200, seed: int | None = 123) -> dict[str, object]:
    """
    Non-parametric bootstrap CIs for (R, k). Returns dict with arrays and 95% CIs.
    Uses the same safe fitter as the main estimate.
    """
    x = np.asarray(counts, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2 or B <= 0:
        return {
            "R_samples": np.array([]),
            "k_samples": np.array([]),
            "R_CI95": (np.nan, np.nan),
            "k_CI95": (np.nan, np.nan),
            "kept": 0
        }

    Rs, ks = [], []
    for _ in range(B):
        sample = x[rng.integers(0, n, n)]
        Rb, kb, _, _ = _fit_nb_safely(sample)
        if np.isfinite(Rb):
            Rs.append(Rb)
        if np.isfinite(kb):
            ks.append(kb)

    Rs = np.asarray(Rs)
    ks = np.asarray(ks)

    def _ci95(arr):
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    out = {
        "R_samples": Rs,
        "k_samples": ks,
        "R_CI95": _ci95(Rs),
        "k_CI95": _ci95(ks),
        "kept": int(max(len(Rs), len(ks)))
    }
    return out


def heterogeneity(
    offspring_dist,
    bootstrap: int = 200,
    bootstrap_seed: int | None = 123,
    superspreading_quantile: float = 0.99,
    superspreading_reference: str = "poisson",
    tol: float = 1e-12
) -> dict[str, object]:
    """
    Enhanced heterogeneity estimator.

    Parameters
    ----------
    offspring_dist : array-like of non-negative ints
        Offspring counts for ALL cases (including zeros).
    bootstrap : int
        Number of non-parametric bootstrap replicates for CI (0 to disable).
    bootstrap_seed : int or None
        RNG seed for bootstrap.
    superspreading_quantile : float in (0,1)
        Quantile for the 'superspreading threshold' (default 0.99).
    superspreading_reference : {"poisson","nb"}
        Distribution used to define the threshold: Poisson(mean=R) or NB(R,k).
        If k is not finite, Poisson is used.
    tol : float
        Tolerance for variance <= mean checks.

    Returns
    -------
    dict with keys:
        mean_Rt
        dispersion_k
        Rt_CI95 (tuple)                # if bootstrapped, else (nan, nan)
        k_CI95 (tuple)                 # if bootstrapped, else (nan, nan)
        superspreading_threshold
        prop_80_percent_transmitters
        pct_zero_transmitters
        pct_superspreaders
        meta (diagnostics)
    """
    x = _check_counts(offspring_dist)
    n = x.size

    if not (0.0 < float(superspreading_quantile) < 1.0):
        raise ValueError("superspreading_quantile must be in (0, 1).")
    q = float(superspreading_quantile)

    ref = str(superspreading_reference).lower().strip()
    if ref not in {"poisson", "nb"}:
        raise ValueError("superspreading_reference must be 'poisson' or 'nb'.")

    mean_Rt_emp = float(np.mean(x))
    pct_zero = float(np.mean((x == 0)) * 100)

    # If mean = 0 (no transmissions), simple deterministic outputs
    if mean_Rt_emp == 0.0:
        return {
            'dispersion_k': np.nan,
            'mean_Rt': 0.0,
            'Rt_CI95': (np.nan, np.nan),
            'k_CI95': (np.nan, np.nan),
            'superspreading_threshold': 0.0,
            'prop_80_percent_transmitters': 1.0,
            'pct_zero_transmitters': pct_zero,
            'pct_superspreaders': 0.0,
            'meta': {
                'n': int(n),
                'total_transmissions': 0,
                'bootstrap_kept': 0,
                'fit_method': 'degenerate',
                'fit_notes': 'mean=0',
                'superspreading_reference': ref,
                'quantile': q
            }
        }

    # Fit NB safely (MLE if possible, otherwise MoM with warnings)
    R_hat, k_hat, method, notes = _fit_nb_safely(x, tol=tol)

    # Superspreading threshold based on chosen reference
    if ref == "nb" and np.isfinite(k_hat) and k_hat > 0:
        p = k_hat / (k_hat + R_hat)
        sse_thr = int(stats.nbinom.ppf(q, k_hat, p))
    else:
        # Poisson fallback when NB not identifiable
        sse_thr = int(stats.poisson.ppf(q, R_hat))

    # Concentration metrics
    prop80 = float(_prop_for_80_percent(x))
    pct_superspreaders = float((x >= sse_thr).mean() * 100.0)

    # Bootstrap CIs
    R_CI = (np.nan, np.nan)
    k_CI = (np.nan, np.nan)
    kept = 0
    if isinstance(bootstrap, int) and bootstrap > 0 and n >= 2:
        boot = _bootstrap_nb(x, B=bootstrap, seed=bootstrap_seed)
        R_CI, k_CI, kept = boot["R_CI95"], boot["k_CI95"], boot["kept"]

    return {
        'dispersion_k': float(k_hat) if np.isfinite(k_hat) else np.nan,
        'mean_Rt': float(R_hat),
        'Rt_CI95': R_CI,
        'k_CI95': k_CI,
        'superspreading_threshold': float(sse_thr),
        'prop_80_percent_transmitters': prop80,
        'pct_zero_transmitters': pct_zero,
        'pct_superspreaders': pct_superspreaders,
        'meta': {
            'n': int(n),
            'total_transmissions': int(x.sum()),
            'bootstrap_kept': int(kept),
            'fit_method': method,
            'fit_notes': notes,
            'superspreading_reference': ref,
            'quantile': q
        }
    }


if __name__ == '__main__':
    # Example usage
    example_data = [0, 1, 0, 2, 3, 0, 0, 5, 1, 0, 0, 4, 2, 0]
    results = heterogeneity(example_data, bootstrap=100)
    for key, value in results.items():
        print(f"{key}: {value}")
