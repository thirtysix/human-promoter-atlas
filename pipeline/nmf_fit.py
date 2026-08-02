#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One NMF fit, with the collapse guard every caller needs.

The multiplicative-update solver started from a random point can drive a
component to zero. At k=18 on the 1,793-TF matrix, 1 of 20 seeds collapsed to a
SINGLE component: ``reconstruction_err_`` came back ``nan`` and the partition
scored an ARI of exactly 0.000 against every healthy fit. That is a solver
failure, not instability in the data, and it reads *backwards* in the
diagnostics -- a collapsed fit puts every module in one cluster, which inflates
the consensus matrix, so the worse the fit the better the cophenetic score
looks.

The fix is to detect the collapse and draw a new seed. It is deliberately not
to change the initialiser, because two alternatives were measured at k=18 over
20 seeds and both break the metric rather than the artifact:

    nndsvdar + mu   no collapses, but ARI rises to 0.819 (0.98 at low k) --
                    every seed starts from nearly the same NNDSVD point, so the
                    statistic measures init determinism, not the data.
    random + cd     no collapses, but ARI halves to 0.233, changing the scale of
                    the statistic and breaking comparison with the published
                    selection.

Re-seeding keeps the median intact (0.420 -> 0.425) while lifting the minimum
off the floor (0.000 -> 0.308): it removes the artifact and nothing else.

``init="random"`` therefore stays. For the k-selection sweep that is load
bearing -- the metric asks how consistently *independent* starts converge on
the same partition, so the starts must genuinely differ.

A healthy seed is bit-identical to what an unguarded fit returns, so adopting
this in a script that pins one seed changes nothing until that seed collapses.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF

# Retry seeds come from a disjoint high range so a retry can never collide with
# another slot's seed and silently turn two independent fits into one.
_RETRY_BASE = 100_000


def fit_nmf(M, k: int, seed: int, max_iter: int = 300):
    """Return ``(W, H, err, collapsed)`` for a single fit.

    ``collapsed`` is true when the error is not finite *or* some component never
    wins a single row -- a fit can be degenerate with a perfectly finite error,
    so both are checked.
    """
    nmf = NMF(n_components=k, init="random", solver="mu",
              beta_loss="frobenius", max_iter=max_iter,
              random_state=seed, tol=1e-4)
    W = nmf.fit_transform(M)
    err = float(nmf.reconstruction_err_)
    collapsed = (not np.isfinite(err)) or (len(np.unique(W.argmax(axis=1))) < k)
    return W, nmf.components_, err, collapsed


def fit_nmf_stable(M, k: int, seed: int, max_iter: int = 300,
                   max_extra: int = 50):
    """:func:`fit_nmf`, re-seeding past collapsed fits.

    Returns ``(W, H, err, n_retries)``. Raises rather than returning a collapsed
    fit: a caller that silently accepted one would write a W/H where a program
    is empty, and every downstream summary would describe it as real.
    """
    W, H, err, collapsed = fit_nmf(M, k, seed, max_iter)
    if not collapsed:
        return W, H, err, 0
    for i in range(max_extra):
        alt = _RETRY_BASE + seed * max_extra + i
        W, H, err, collapsed = fit_nmf(M, k, alt, max_iter)
        if not collapsed:
            return W, H, err, i + 1
    raise RuntimeError(
        f"k={k} seed={seed}: NMF collapsed on all {max_extra} retry seeds")


def fit_nmf_dominant(M, k: int, seed: int, max_iter: int = 300):
    """``(dominant_component_per_row, err, collapsed)`` -- the k-selection view.

    Only the partition matters there, and W is large, so it is not returned.
    """
    W, _H, err, collapsed = fit_nmf(M, k, seed, max_iter)
    return W.argmax(axis=1).astype(np.int32), err, collapsed


def fit_nmf_dominant_stable(M, k: int, seed: int, max_iter: int = 300,
                            max_extra: int = 50):
    """:func:`fit_nmf_dominant`, re-seeding past collapsed fits.

    Returns ``(dom, err, n_retries)``.
    """
    dom, err, collapsed = fit_nmf_dominant(M, k, seed, max_iter)
    if not collapsed:
        return dom, err, 0
    for i in range(max_extra):
        alt = _RETRY_BASE + seed * max_extra + i
        dom, err, collapsed = fit_nmf_dominant(M, k, alt, max_iter)
        if not collapsed:
            return dom, err, i + 1
    raise RuntimeError(
        f"k={k} seed={seed}: NMF collapsed on all {max_extra} retry seeds")
