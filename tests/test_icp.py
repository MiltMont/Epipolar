"""Tests for icp/pointcloud.py, icp/align.py, icp/iterate.py."""
import numpy as np
import pytest

from tray.icp.align import align_svd
from tray.icp.iterate import icp
from tray.icp.pointcloud import depth_to_pointcloud, downsample, transform_cloud


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _make_K(fx=525.0, fy=525.0, cx=319.5, cy=239.5) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# pointcloud
# ─────────────────────────────────────────────────────────────────────────────

class TestDepthToPointcloud:
    def test_single_pixel_at_principal_point(self):
        K = _make_K(fx=500.0, fy=500.0, cx=5.0, cy=5.0)
        depth = np.zeros((11, 11), dtype=np.uint16)
        # Pixel (5, 5) is at principal point → x=0, y=0
        depth[5, 5] = 2500   # z = 2500/5000 = 0.5 m
        cloud = depth_to_pointcloud(depth, K)
        assert cloud.shape == (1, 3)
        np.testing.assert_allclose(cloud[0], [0.0, 0.0, 0.5], atol=1e-9)

    def test_off_center_pixel(self):
        K = _make_K(fx=400.0, fy=400.0, cx=10.0, cy=10.0)
        depth = np.zeros((21, 21), dtype=np.uint16)
        # Pixel (u=12, v=10): Δu = 2, z = 1.0 m → x = 2*1.0/400 = 0.005
        depth[10, 12] = 5000  # z = 1.0
        cloud = depth_to_pointcloud(depth, K)
        assert cloud.shape == (1, 3)
        np.testing.assert_allclose(cloud[0, 0], 2.0 / 400.0, atol=1e-9)
        np.testing.assert_allclose(cloud[0, 1], 0.0, atol=1e-9)
        np.testing.assert_allclose(cloud[0, 2], 1.0, atol=1e-9)

    def test_zero_depth_masked(self):
        K = _make_K()
        depth = np.zeros((10, 10), dtype=np.uint16)
        cloud = depth_to_pointcloud(depth, K)
        assert cloud.shape == (0, 3)

    def test_max_depth_filter(self):
        K = _make_K()
        depth = np.zeros((5, 5), dtype=np.uint16)
        # Far pixel: z = 12 m (exceeds max_depth=10); 60000/5000 = 12, fits uint16
        depth[2, 2] = 60000
        cloud = depth_to_pointcloud(depth, K, max_depth=10.0)
        assert cloud.shape == (0, 3)

    def test_full_image_shape(self):
        K = _make_K()
        rng = np.random.default_rng(0)
        depth = rng.integers(1, 10000, size=(480, 640), dtype=np.uint16)
        cloud = depth_to_pointcloud(depth, K)
        assert cloud.ndim == 2
        assert cloud.shape[1] == 3
        assert len(cloud) == 480 * 640  # all pixels valid


class TestDownsample:
    def test_factor_1_noop(self):
        cloud = np.ones((100, 3))
        assert downsample(cloud, 1) is cloud

    def test_reduces_size(self):
        cloud = np.ones((1000, 3))
        small = downsample(cloud, 10, rng=np.random.default_rng(0))
        assert len(small) == pytest.approx(100, abs=5)

    def test_output_is_subset(self):
        rng = np.random.default_rng(1)
        cloud = rng.standard_normal((500, 3))
        small = downsample(cloud, 5, rng=rng)
        # Every returned point must exist in the original cloud
        for pt in small:
            assert any(np.allclose(pt, cloud[i]) for i in range(len(cloud)))


class TestTransformCloud:
    def test_identity(self):
        cloud = np.random.default_rng(0).standard_normal((50, 3))
        out = transform_cloud(cloud, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(out, cloud)

    def test_translation(self):
        cloud = np.zeros((10, 3))
        t = np.array([1.0, 2.0, 3.0])
        out = transform_cloud(cloud, np.eye(3), t)
        np.testing.assert_allclose(out, np.tile(t, (10, 1)))

    def test_rotation(self):
        rng = np.random.default_rng(2)
        cloud = rng.standard_normal((20, 3))
        R = _random_rotation(rng)
        out = transform_cloud(cloud, R, np.zeros(3))
        np.testing.assert_allclose(out, cloud @ R.T, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# align_svd
# ─────────────────────────────────────────────────────────────────────────────

class TestAlignSVD:
    def test_identity_transform(self):
        rng = np.random.default_rng(10)
        cloud = rng.standard_normal((50, 3))
        R, t = align_svd(cloud, cloud)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, 0.0, atol=1e-10)

    def test_pure_translation(self):
        rng = np.random.default_rng(11)
        source = rng.standard_normal((40, 3))
        shift = np.array([1.5, -0.5, 2.0])
        target = source + shift
        R, t = align_svd(source, target)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(t, shift, atol=1e-9)

    def test_rotation_recovery(self):
        rng = np.random.default_rng(12)
        source = rng.standard_normal((60, 3))
        true_R = _random_rotation(rng)
        target = source @ true_R.T
        R, t = align_svd(source, target)
        np.testing.assert_allclose(R, true_R, atol=1e-9)
        np.testing.assert_allclose(t, 0.0, atol=1e-9)

    def test_rigid_recovery(self):
        rng = np.random.default_rng(13)
        source = rng.standard_normal((80, 3))
        true_R = _random_rotation(rng)
        true_t = rng.standard_normal(3)
        target = source @ true_R.T + true_t
        R, t = align_svd(source, target)
        aligned = source @ R.T + t
        np.testing.assert_allclose(aligned, target, atol=1e-8)

    def test_det_positive(self):
        rng = np.random.default_rng(14)
        source = rng.standard_normal((30, 3))
        true_R = _random_rotation(rng)
        target = source @ true_R.T
        R, _ = align_svd(source, target)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            align_svd(np.ones((10, 3)), np.ones((10, 2)))


# ─────────────────────────────────────────────────────────────────────────────
# icp
# ─────────────────────────────────────────────────────────────────────────────

class TestICP:
    def test_identity_on_same_cloud(self):
        rng = np.random.default_rng(20)
        cloud = rng.standard_normal((200, 3))
        result = icp(cloud, cloud, max_iter=10, tolerance=1e-6)
        aligned = transform_cloud(cloud, result.R, result.t)
        np.testing.assert_allclose(aligned, cloud, atol=1e-6)

    def test_recovers_small_translation(self):
        rng = np.random.default_rng(21)
        source = rng.standard_normal((300, 3))
        true_t = np.array([0.05, 0.03, -0.04])
        target = source + true_t
        result = icp(source, target, max_iter=50, tolerance=1e-6, reject_dist=1.0)
        t_err = np.linalg.norm(result.t - true_t)
        assert t_err < 0.01, f"Translation error {t_err:.4f} too large"

    def test_recovers_small_rotation(self):
        rng = np.random.default_rng(22)
        source = rng.standard_normal((300, 3))
        angle = 0.05   # ~3 degrees
        # Rotate around z-axis
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        target = source @ R_true.T
        result = icp(source, target, max_iter=50, tolerance=1e-7, reject_dist=1.0)
        # Check that aligned source is close to target
        aligned = transform_cloud(source, result.R, result.t)
        mean_err = float(np.linalg.norm(aligned - target, axis=1).mean())
        assert mean_err < 0.01, f"Mean alignment error {mean_err:.4f} too large"

    def test_returns_icp_result_fields(self):
        rng = np.random.default_rng(23)
        cloud = rng.standard_normal((100, 3))
        result = icp(cloud, cloud, max_iter=5)
        assert result.R.shape == (3, 3)
        assert result.t.shape == (3,)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.converged, bool)
        assert len(result.mean_dist_history) == result.n_iterations

    def test_initial_transform(self):
        rng = np.random.default_rng(24)
        source = rng.standard_normal((200, 3))
        true_t = np.array([0.1, 0.0, 0.0])
        target = source + true_t
        # Provide initial transform that is already the answer
        result = icp(
            source, target,
            max_iter=20, tolerance=1e-6, reject_dist=1.0,
            init_R=np.eye(3), init_t=true_t,
        )
        aligned = transform_cloud(source, result.R, result.t)
        mean_err = float(np.linalg.norm(aligned - target, axis=1).mean())
        assert mean_err < 1e-5

    def test_det_R_positive(self):
        rng = np.random.default_rng(25)
        source = rng.standard_normal((150, 3))
        target = source + rng.standard_normal(3) * 0.05
        result = icp(source, target, max_iter=20)
        assert np.linalg.det(result.R) == pytest.approx(1.0, abs=1e-6)
