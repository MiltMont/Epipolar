"""Tests for eval/align.py, eval/ate.py, eval/rte.py."""
import numpy as np
import pytest

from tray.eval.align import apply_alignment, umeyama
from tray.eval.ate import compute_ate
from tray.eval.rte import compute_rte


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random rotation via Gram-Schmidt on a random 3x3."""
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _make_trajectory(N: int, rng: np.random.Generator) -> np.ndarray:
    """Build a smooth (N, 4, 4) SE(3) trajectory."""
    traj = np.zeros((N, 4, 4))
    traj[:, 3, 3] = 1.0
    T_acc = np.eye(4)
    for i in range(N):
        R = _random_rotation(rng)
        t = rng.standard_normal(3) * 0.1
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t
        T_acc = T_acc @ T_rel
        traj[i] = T_acc
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Umeyama
# ─────────────────────────────────────────────────────────────────────────────

class TestUmeyama:
    def test_identity(self):
        rng = np.random.default_rng(0)
        P = rng.standard_normal((50, 3))
        R, t, c = umeyama(P, P, with_scale=True)
        assert c == pytest.approx(1.0, abs=1e-10)
        assert np.allclose(R, np.eye(3), atol=1e-10)
        assert np.allclose(t, 0.0, atol=1e-10)

    def test_pure_translation(self):
        rng = np.random.default_rng(1)
        P = rng.standard_normal((30, 3))
        shift = np.array([1.5, -2.0, 0.7])
        Q = P + shift
        R, t, c = umeyama(P, Q, with_scale=False)
        P_aligned = apply_alignment(P, R, t, c)
        assert np.allclose(P_aligned, Q, atol=1e-10)

    def test_scale_recovery(self):
        rng = np.random.default_rng(2)
        P = rng.standard_normal((40, 3))
        true_c = 2.5
        true_R = _random_rotation(rng)
        true_t = rng.standard_normal(3)
        Q = true_c * (P @ true_R.T) + true_t
        R, t, c = umeyama(P, Q, with_scale=True)
        assert c == pytest.approx(true_c, rel=1e-6)
        P_aligned = apply_alignment(P, R, t, c)
        assert np.allclose(P_aligned, Q, atol=1e-8)

    def test_rotation_only(self):
        rng = np.random.default_rng(3)
        P = rng.standard_normal((50, 3))
        true_R = _random_rotation(rng)
        Q = P @ true_R.T
        R, t, c = umeyama(P, Q, with_scale=False)
        P_aligned = apply_alignment(P, R, t, c)
        assert np.allclose(P_aligned, Q, atol=1e-8)

    def test_with_scale_false_does_not_change_scale(self):
        rng = np.random.default_rng(4)
        P = rng.standard_normal((20, 3))
        # Apply scale=3 to create Q but ask for scale=False
        Q = 3.0 * P
        R, t, c = umeyama(P, Q, with_scale=False)
        assert c == pytest.approx(1.0, abs=1e-12)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            umeyama(np.ones((10, 3)), np.ones((10, 4)), with_scale=True)


# ─────────────────────────────────────────────────────────────────────────────
# ATE
# ─────────────────────────────────────────────────────────────────────────────

class TestATE:
    def test_zero_error_on_identical(self):
        rng = np.random.default_rng(10)
        traj = _make_trajectory(20, rng)
        result = compute_ate(traj, traj, with_scale=False)
        assert result.mean == pytest.approx(0.0, abs=1e-10)
        assert result.rmse == pytest.approx(0.0, abs=1e-10)

    def test_noisy_trajectory(self):
        rng = np.random.default_rng(11)
        gt = _make_trajectory(50, rng)
        noise_level = 0.05
        est = gt.copy()
        est[:, :3, 3] += rng.standard_normal((50, 3)) * noise_level
        result = compute_ate(est, gt, with_scale=False)
        # mean error should be in the right ballpark of noise_level
        assert result.mean < noise_level * 5
        assert result.rmse >= 0

    def test_all_stats_nonnegative(self):
        rng = np.random.default_rng(12)
        est = _make_trajectory(30, rng)
        gt = _make_trajectory(30, rng)
        result = compute_ate(est, gt, with_scale=True)
        assert result.mean >= 0
        assert result.median >= 0
        assert result.std >= 0
        assert result.max >= 0
        assert result.rmse >= 0
        assert result.rmse >= result.mean  # RMSE ≥ mean for non-negative errors

    def test_as_dict_keys(self):
        rng = np.random.default_rng(13)
        traj = _make_trajectory(10, rng)
        d = compute_ate(traj, traj, with_scale=False).as_dict()
        assert set(d.keys()) == {"ate_mean", "ate_median", "ate_std", "ate_max", "ate_rmse"}


# ─────────────────────────────────────────────────────────────────────────────
# RTE
# ─────────────────────────────────────────────────────────────────────────────

class TestRTE:
    def test_zero_error_on_identical(self):
        rng = np.random.default_rng(20)
        traj = _make_trajectory(20, rng)
        result = compute_rte(traj, traj, delta=1)
        assert result.mean == pytest.approx(0.0, abs=1e-10)
        assert result.rmse == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_error_on_different(self):
        rng = np.random.default_rng(21)
        est = _make_trajectory(20, rng)
        gt = _make_trajectory(20, rng)
        result = compute_rte(est, gt, delta=1)
        assert result.mean > 0

    def test_delta_2(self):
        rng = np.random.default_rng(22)
        traj = _make_trajectory(30, rng)
        result = compute_rte(traj, traj, delta=2)
        assert len(result.errors) == 28  # N - delta
        assert result.mean == pytest.approx(0.0, abs=1e-10)

    def test_all_stats_nonnegative(self):
        rng = np.random.default_rng(23)
        est = _make_trajectory(25, rng)
        gt = _make_trajectory(25, rng)
        result = compute_rte(est, gt, delta=1)
        for val in (result.mean, result.median, result.std, result.max, result.rmse):
            assert val >= 0

    def test_as_dict_keys(self):
        rng = np.random.default_rng(24)
        traj = _make_trajectory(10, rng)
        d = compute_rte(traj, traj).as_dict()
        assert set(d.keys()) == {"rte_mean", "rte_median", "rte_std", "rte_max", "rte_rmse"}
