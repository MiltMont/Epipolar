# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Plan and current progress

The implementation plan lives at `~/.claude/plans/compressed-bubbling-falcon.md` — read it before starting work to see the full Phase 2 step list, the five required configurations per pipeline, and the "gotchas" section (TUM depth scale, intrinsics per sequence, Umeyama scale-on-vs-off, 7-point cubic ambiguity, etc.). `git log --oneline` is the source of truth for what's been built.

As of the latest commit:

- ✅ **Phase 0** — uv bootstrap, deps, scaffolding, git init, CLAUDE.md commands section.
- ✅ **Phase 2 step 1** — `src/tray/data/tum.py` (Intrinsics + freiburg1/2/3, file parsing, quaternion→SE(3), TUM `associate.py`-style nearest-timestamp pairing, `load_sequence`). Real `freiburg1_xyz` loads with 99.7% GT coverage; the 2 unmatched frames sit inside a genuine ~110 ms mocap dropout.
- ✅ **Phase 2 step 2** — `src/tray/features/` (ORB/SIFT factory, `Features`/`Matches` dataclasses, `match_top_n` + cross-check, `match_lowe`). 26 tests pass.
- ⏭ **Phase 2 step 3 (next)** — `src/tray/epipolar/fundamental.py`: implement `eight_point` (Hartley normalization + SVD + rank-2 projection), `seven_point` (cubic det(F1 + α F2) = 0 → up to 3 candidates), and `ransac` from scratch. Cross-check against `cv2.findFundamentalMat` in tests.

## Repository status

`uv`-managed Python 3.12 package (`tray`) with a src-layout under `src/tray/`. The Spanish spec is split across files indexed by `INDEX.md`:

- `GUIDE.md` — overall project description and objectives
- `INSTRUCTIONS.md` — step-by-step algorithmic requirements (epipolar geometry + ICP)
- `EXPERIMENTS.md` — required experimental configurations
- `PATHS.md` — trajectory error metrics that must be reported
- `VISUALISATION.md` — required plots and visualizations
- `PARAMETERS.md` — parameter-tuning requirements

Documents are written in Spanish — match the language of the spec when writing the final report. Code identifiers and inline comments stay in English.

## Commands

All commands run from the repo root.

```bash
uv sync                              # install/update deps from pyproject.toml + uv.lock
uv add <pkg>                         # add a runtime dep
uv add --dev <pkg>                   # add a dev dep
uv run python -m tray --help         # CLI entrypoint
uv run python -m tray run --config configs/<name>.yaml
uv run pytest -q                     # run the test suite
uv run pytest tests/test_x.py::test_y  # run a single test
uv run ruff check src tests          # lint
uv run ruff format src tests         # format
bash scripts/download_tum.sh         # fetch TUM xyz + pioneer_slam into data/
```

## Code layout

- `src/tray/data/` — TUM RGB-D loader (associations, ground truth, intrinsics per freiburg sequence)
- `src/tray/features/` — ORB/SIFT extractors and matchers (BFMatcher + Lowe ratio)
- `src/tray/epipolar/` — 8-point, 7-point, RANSAC fundamental-matrix estimation; E decomposition; pairwise SE(3) chain
- `src/tray/icp/` — depth → point cloud, SVD rigid alignment, iterative refinement (KDTree NN), pairwise chain
- `src/tray/eval/` — Umeyama alignment, ATE (mean/median/std/max/RMSE), RTE
- `src/tray/viz/` — 3D trajectory plots and `cv2.drawMatches` wrapper
- `src/tray/experiments/runner.py` — YAML config → pipeline → `results/<config_name>/`
- `src/tray/cli.py` — `python -m tray run --config …` entrypoint
- `configs/` — 5 epipolar + 5 ICP YAML configs (see plan file)
- `tests/` — synthetic-data unit tests; cross-check from-scratch math against OpenCV references
- `data/`, `results/` — gitignored

The from-scratch algorithm files (`epipolar/fundamental.py`, `epipolar/pose.py`, `icp/align.py`, `icp/iterate.py`, etc.) are filled in as Phase 2 progresses — see the "Plan and current progress" section above for what's already landed.

## What the project must do

Estimate and evaluate camera trajectories from the **TUM RGB-D** dataset (download from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) using **two distinct pipelines** that map to two dataset subsets:

1. **Epipolar geometry** on the `xyz` subset:
   - Feature detection: ORB or SIFT
   - Descriptor matching (incl. Lowe ratio test where applicable)
   - Fundamental matrix **F** via **three separate algorithms**: 8-point, 7-point, and RANSAC with 100-point subsets — all three are required deliverables, not alternatives
   - Essential matrix `E = K^T F K` using intrinsics K from TUM calibration (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
   - SVD of E → candidate (R, t); pick the solution that triangulates the most points in front of both cameras
   - Chain relative transforms into a global trajectory

2. **ICP** on the `pioneer_slam` subset:
   - Build point clouds from consecutive depth maps
   - Nearest-neighbor search (direct or SciPy KDTree)
   - Rigid transform via centroid alignment + SVD of `H = Σ p'_i q'_i^T`; if `det(R) < 0`, negate R
   - Iterate: transform → match → reject distant correspondences → re-estimate → accumulate → check error tolerance

Both pipelines must be evaluated against ground truth with **absolute trajectory error** (mean, median, std, max, RMSE) and **relative trajectory error**.

## Required experiments

At least **5 configurations per method** must be implemented and compared, with written justification for each. Examples called out in the spec: ORB 20 vs 100 layers, RANSAC threshold sweep, top-N matches, ICP distance tolerance. `PARAMETERS.md` also calls out using the epipolar estimate as ICP initialization as one tuning lever.

## Stack

Already installed: **OpenCV (contrib)** for feature detection, `cv2.drawMatches`, and reference implementations to cross-check against in tests; **SciPy** for `cKDTree` in ICP; **Matplotlib** for 3D trajectory plots; **PyYAML** for configs; **tqdm** for progress bars. **Open3D is intentionally not included** — point-cloud work stays in numpy + scipy. Every plot must carry title, axis labels, legend, interpretable scale, and the dataset+configuration name (per `VISUALISATION.md`).

## Implementation conventions

- **Hybrid algorithm depth**: F-estimation (8-pt, 7-pt, RANSAC), E decomposition, and the ICP loop are implemented from scratch (the spec details the math). OpenCV is used for I/O, feature extraction, drawing, and as a reference in unit tests — never as the production estimator for those required algorithms.
- **TUM specifics that aren't in the spec**: depth PNG scale factor is **5000** (divide for meters); RGB↔depth need nearest-timestamp pairing (max Δ ≈ 0.02 s); `xyz` uses freiburg1 intrinsics, `pioneer_slam` uses freiburg2 — different `K` and distortion.
- **Trajectory alignment for ATE**: Umeyama with **scale ON** for the monocular epipolar pipeline (up-to-scale), **scale OFF** for the metric ICP pipeline.
- **Configs are reproducible artifacts**: every experiment run lands in `results/<config_name>/` (metrics JSON, trajectory `.npy`, plot PNGs). No ad-hoc parameter editing in notebooks for runs that go in the report.

## Final deliverable

Per `GUIDE.md`: code, plots, and a **comparative report** with results, analysis, and conclusions. Treat the report as a first-class artifact, not an afterthought.
