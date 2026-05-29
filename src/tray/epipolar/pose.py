"""Essential matrix and camera pose recovery from fundamental matrix.

Steps implemented from scratch (no OpenCV for the core math):
  1. essential_from_F  — E = K^T F K
  2. decompose_E       — SVD of E → 4 (R, t) candidates; enforces det(R) = +1
  3. triangulate_dlt   — DLT linear triangulation for a batch of correspondences
  4. recover_pose      — cheirality check; select (R, t) maximising points in
                         front of both cameras; also returns validity mask

Cross-checked against cv2.recoverPose in tests/.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Essential matrix
# ─────────────────────────────────────────────────────────────────────────────

def essential_from_F(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute the essential matrix E = K^T F K.

    Args:
        F: (3, 3) fundamental matrix.
        K: (3, 3) camera intrinsics.  If the two cameras have different
           intrinsics, pass K2 here (E = K2^T F K1 is handled by the caller
           supplying the appropriate combined expression, or call with K1 and
           multiply externally — the standard monocular SLAM case uses one K).

    Returns:
        E: (3, 3) essential matrix (not normalised).
    """
    F = np.asarray(F, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    return K.T @ F @ K


# ─────────────────────────────────────────────────────────────────────────────
# E decomposition
# ─────────────────────────────────────────────────────────────────────────────

# Rotation helper used in the SVD-based decomposition (Nister 2004).
_W = np.array([[0., -1., 0.],
               [1.,  0., 0.],
               [0.,  0., 1.]], dtype=np.float64)


def decompose_E(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decompose E via SVD into 4 (R, t) candidate pose pairs.

    The essential matrix satisfies E = U diag(σ, σ, 0) V^T.  Two rotation
    candidates (W and W^T) combined with two translation signs (±u3) give
    the four solutions.

    Args:
        E: (3, 3) essential matrix.

    Returns:
        List of 4 (R (3,3), t (3,)) tuples.  Each R satisfies det(R) = +1
        and ‖t‖ = 1.
    """
    E = np.asarray(E, dtype=np.float64)
    U, _, Vt = np.linalg.svd(E)

    # Enforce right-handedness: det(U) and det(Vt) must each be +1.
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    R1 = U @ _W @ Vt
    R2 = U @ _W.T @ Vt
    t_pos = U[:, 2]
    t_neg = -U[:, 2]

    # det should already be +1 after the sign fixes above, but guard anyway.
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    return [(R1, t_pos), (R1, t_neg), (R2, t_pos), (R2, t_neg)]


# ─────────────────────────────────────────────────────────────────────────────
# DLT triangulation
# ─────────────────────────────────────────────────────────────────────────────

def triangulate_dlt(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Triangulate N 3-D points from two projection matrices via DLT.

    For each correspondence (x1, x2) the cross-product form of the projection
    equations yields a 4×4 homogeneous system A X = 0:

        [x1 * p3_1^T - p1_1^T]
        [y1 * p3_1^T - p2_1^T]  X = 0
        [x2 * p3_2^T - p1_2^T]
        [y2 * p3_2^T - p2_2^T]

    The solution is the right singular vector of A for the smallest singular
    value; we then dehomogenise by dividing by the fourth coordinate.

    Args:
        P1:   (3, 4) projection matrix for camera 1.
        P2:   (3, 4) projection matrix for camera 2.
        pts1: (N, 2) pixel coordinates in image 1.
        pts2: (N, 2) pixel coordinates in image 2.

    Returns:
        X: (N, 3) triangulated 3-D points in the reference frame of camera 1.
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    N = len(pts1)
    X = np.empty((N, 3), dtype=np.float64)

    p1_rows = P1[[0, 1, 2], :]   # rows of P1
    p2_rows = P2[[0, 1, 2], :]   # rows of P2

    for i in range(N):
        x1, y1 = pts1[i, 0], pts1[i, 1]
        x2, y2 = pts2[i, 0], pts2[i, 1]

        A = np.array([
            x1 * p1_rows[2] - p1_rows[0],
            y1 * p1_rows[2] - p1_rows[1],
            x2 * p2_rows[2] - p2_rows[0],
            y2 * p2_rows[2] - p2_rows[1],
        ])  # (4, 4)

        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        Xh = Vt[-1]              # homogeneous solution
        X[i] = Xh[:3] / Xh[3]

    return X


# ─────────────────────────────────────────────────────────────────────────────
# Pose recovery via cheirality
# ─────────────────────────────────────────────────────────────────────────────

def recover_pose(
    E: np.ndarray,
    K: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the unique (R, t) that triangulates the most points in front of
    both cameras (cheirality check).

    Args:
        E:    (3, 3) essential matrix.
        K:    (3, 3) camera intrinsics (shared by both views).
        pts1: (N, 2) matched pixel coordinates in image 1.
        pts2: (N, 2) matched pixel coordinates in image 2.

    Returns:
        R:    (3, 3) rotation matrix, det = +1.
        t:    (3,) unit translation vector (up-to-scale for monocular input).
        mask: (N,) bool — True where the 3-D point satisfies cheirality for
              the winning (R, t).
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])   # K [I | 0]

    best_R: np.ndarray | None = None
    best_t: np.ndarray | None = None
    best_mask = np.zeros(len(pts1), dtype=bool)
    best_count = -1

    for R, t in decompose_E(E):
        P2 = K @ np.hstack([R, t.reshape(3, 1)])        # K [R | t]
        X = triangulate_dlt(P1, P2, pts1, pts2)

        # Depth in camera-1 frame.
        z1 = X[:, 2]

        # Depth in camera-2 frame.
        z2 = ((R @ X.T).T + t)[:, 2]

        valid = (z1 > 0) & (z2 > 0)
        count = int(valid.sum())

        if count > best_count:
            best_count = count
            best_R = R
            best_t = t
            best_mask = valid

    assert best_R is not None, "decompose_E returned no candidates"
    return best_R, best_t, best_mask
