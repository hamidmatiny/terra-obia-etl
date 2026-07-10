"""Fair comparison experiments: same features, same split, same metrics.

Does NOT change production training. Reconstructs the documented 80/20
stratified holdout from labeled_stands.csv (random_state=42) and evaluates:

1. Existing committed artifacts (baseline / balanced) on that holdout
2. Fresh LightGBM + XGBoost trained only on the train partition
3. Hierarchical (coarse group → fine class) GBM on the same partition

The held-out test partition is never used for fitting or tuning.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from terra_core.classification.evaluation import compute_object_classification_metrics
from terra_core.classification.features import labeled_frame_to_features
from terra_core.classification.registry import load_model_artifact

FEATURE_COLUMNS = [
    "area_m2",
    "perimeter_m",
    "compactness",
    "l1_ds",
    "l1_sc",
    "l1_vs",
    "l1_pstock",
    "lc_code",
    "wri_code",
    "spvc",
]

# Coarse groups for hierarchical experiment (inventory semantics).
COARSE_OF: dict[str, str] = {
    "conifer": "forest",
    "deciduous": "forest",
    "mixed": "forest",
    "bog": "wetland",
    "fen": "wetland",
    "wetland_forest": "wetland",
    "wetland_marsh": "wetland",
    "wetland_shrub": "wetland",
    "wetland_unknown": "wetland",
    "coastal_marsh": "wetland",
    "aquatic_bed": "wetland",
    "water": "water",
    "developed": "anthropogenic",
    "agriculture": "anthropogenic",
    "industrial": "anthropogenic",
    "infrastructure": "anthropogenic",
    "defense": "anthropogenic",
    "recreational": "anthropogenic",
    "barren": "other",
    "beach": "other",
    "dune": "other",
    "rocky_shore": "other",
    "tidal_flat": "other",
    "scrub": "other",
    "shrub": "other",
    "herbaceous": "other",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class ExperimentResult:
    """Holdout metrics for one algorithm / artifact experiment."""

    name: str
    train_rows: int
    test_rows: int
    cover_accuracy: float
    canopy_accuracy: float
    overall_accuracy_mean: float
    cover_macro_f1: float
    cover_weighted_f1: float
    forest_cover_accuracy: float | None
    conifer_recall: float | None
    deciduous_recall: float | None
    mixed_recall: float | None
    top_cover_confusions: list[dict[str, object]]
    notes: str
    wall_seconds: float


def _top_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 8
) -> list[dict[str, object]]:
    pairs = Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
    return [{"true": t, "predicted": p, "count": int(n)} for (t, p), n in pairs.most_common(top_n)]


def _forest_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float | None, float | None, float | None, float | None]:
    forest = {"conifer", "deciduous", "mixed"}
    mask = np.isin(y_true, list(forest))
    if not mask.any():
        return None, None, None, None
    forest_acc = float(accuracy_score(y_true[mask], y_pred[mask]))

    def recall(label: str) -> float | None:
        m = y_true == label
        if not m.any():
            return None
        return float((y_pred[m] == label).mean())

    return forest_acc, recall("conifer"), recall("deciduous"), recall("mixed")


def evaluate_predictions(
    name: str,
    y_cover_true: np.ndarray,
    y_cover_pred: np.ndarray,
    y_canopy_true: np.ndarray,
    y_canopy_pred: np.ndarray,
    *,
    train_rows: int,
    notes: str,
    wall_seconds: float,
) -> ExperimentResult:
    """Score cover/canopy predictions with the production metric helpers."""
    cover_acc = float(accuracy_score(y_cover_true, y_cover_pred))
    canopy_acc = float(accuracy_score(y_canopy_true, y_canopy_pred))
    report = compute_object_classification_metrics(
        y_cover_true, y_cover_pred, y_canopy_true, y_canopy_pred
    )
    forest_acc, con_r, dec_r, mix_r = _forest_metrics(y_cover_true, y_cover_pred)
    return ExperimentResult(
        name=name,
        train_rows=train_rows,
        test_rows=len(y_cover_true),
        cover_accuracy=cover_acc,
        canopy_accuracy=canopy_acc,
        overall_accuracy_mean=float(report.overall_accuracy),
        cover_macro_f1=float(
            f1_score(y_cover_true, y_cover_pred, average="macro", zero_division=0)
        ),
        cover_weighted_f1=float(
            f1_score(y_cover_true, y_cover_pred, average="weighted", zero_division=0)
        ),
        forest_cover_accuracy=forest_acc,
        conifer_recall=con_r,
        deciduous_recall=dec_r,
        mixed_recall=mix_r,
        top_cover_confusions=_top_confusions(y_cover_true, y_cover_pred),
        notes=notes,
        wall_seconds=wall_seconds,
    )


def make_split(labeled: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Identical split contract as train_stand_classifier / ETL train script."""
    x_matrix, y_cover, y_canopy, _cols = labeled_frame_to_features(
        labeled, feature_columns=FEATURE_COLUMNS
    )
    return train_test_split(
        x_matrix,
        y_cover,
        y_canopy,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cover,
    )


def eval_existing_artifact(
    name: str,
    model_dir: Path,
    x_test: np.ndarray,
    y_cover_test: np.ndarray,
    y_canopy_test: np.ndarray,
    train_rows: int,
    notes: str,
) -> ExperimentResult:
    """Evaluate a committed joblib artifact on a fixed holdout."""
    t0 = time.perf_counter()
    art = load_model_artifact(model_dir)
    cover_pred = art.cover_type_model.predict(x_test)
    canopy_pred = art.canopy_closure_model.predict(x_test)
    return evaluate_predictions(
        name,
        y_cover_test,
        cover_pred,
        y_canopy_test,
        canopy_pred,
        train_rows=train_rows,
        notes=notes,
        wall_seconds=time.perf_counter() - t0,
    )


def train_sklearn_gbm(
    name: str,
    x_train: np.ndarray,
    y_cover_train: np.ndarray,
    y_canopy_train: np.ndarray,
    x_test: np.ndarray,
    y_cover_test: np.ndarray,
    y_canopy_test: np.ndarray,
    *,
    class_weight: str | None,
    n_estimators: int = 100,
) -> ExperimentResult:
    """Train sklearn GBM cover/canopy heads on the train partition only."""
    t0 = time.perf_counter()
    cover_m = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=3, random_state=RANDOM_STATE, verbose=0
    )
    canopy_m = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=3, random_state=RANDOM_STATE, verbose=0
    )
    cw = compute_sample_weight(class_weight, y_cover_train) if class_weight else None
    kw = compute_sample_weight(class_weight, y_canopy_train) if class_weight else None
    cover_m.fit(x_train, y_cover_train, sample_weight=cw)
    canopy_m.fit(x_train, y_canopy_train, sample_weight=kw)
    return evaluate_predictions(
        name,
        y_cover_test,
        cover_m.predict(x_test),
        y_canopy_test,
        canopy_m.predict(x_test),
        train_rows=len(x_train),
        notes=f"sklearn GBM n_estimators={n_estimators} class_weight={class_weight}",
        wall_seconds=time.perf_counter() - t0,
    )


def train_lightgbm(
    name: str,
    x_train: np.ndarray,
    y_cover_train: np.ndarray,
    y_canopy_train: np.ndarray,
    x_test: np.ndarray,
    y_cover_test: np.ndarray,
    y_canopy_test: np.ndarray,
    *,
    class_weight: str | None,
) -> ExperimentResult:
    """Train LightGBM cover/canopy heads on the train partition only."""
    import lightgbm as lgb

    t0 = time.perf_counter()
    common = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=4,
    )
    cover_m = lgb.LGBMClassifier(**common, class_weight=class_weight)
    canopy_m = lgb.LGBMClassifier(**common, class_weight=class_weight)
    cover_m.fit(x_train, y_cover_train)
    canopy_m.fit(x_train, y_canopy_train)
    return evaluate_predictions(
        name,
        y_cover_test,
        cover_m.predict(x_test),
        y_canopy_test,
        canopy_m.predict(x_test),
        train_rows=len(x_train),
        notes=f"LightGBM n_estimators=100 max_depth=3 class_weight={class_weight}",
        wall_seconds=time.perf_counter() - t0,
    )


def train_xgboost(
    name: str,
    x_train: np.ndarray,
    y_cover_train: np.ndarray,
    y_canopy_train: np.ndarray,
    x_test: np.ndarray,
    y_cover_test: np.ndarray,
    y_canopy_test: np.ndarray,
    *,
    balanced: bool,
) -> ExperimentResult:
    """Train XGBoost cover/canopy heads on the train partition only."""
    from xgboost import XGBClassifier

    t0 = time.perf_counter()
    # XGBoost needs integer labels; map via sorted unique from train only.
    cover_classes = sorted(set(y_cover_train))
    canopy_classes = sorted(set(y_canopy_train))
    cover_to_i = {c: i for i, c in enumerate(cover_classes)}
    canopy_to_i = {c: i for i, c in enumerate(canopy_classes)}
    i_to_cover = {i: c for c, i in cover_to_i.items()}
    i_to_canopy = {i: c for c, i in canopy_to_i.items()}

    y_c_tr = np.array([cover_to_i[c] for c in y_cover_train])
    y_k_tr = np.array([canopy_to_i[c] for c in y_canopy_train])

    # Drop test labels unseen in train (should be none with stratified split).
    cover_ok = np.isin(y_cover_test, cover_classes)
    canopy_ok = np.isin(y_canopy_test, canopy_classes)
    assert cover_ok.all() and canopy_ok.all(), "Test labels missing from train — leakage/split bug"

    common = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
        objective="multi:softprob",
    )
    # scale_pos_weight is binary-only; for multiclass use sample_weight when balanced.
    cover_m = XGBClassifier(**common, num_class=len(cover_classes))
    canopy_m = XGBClassifier(**common, num_class=len(canopy_classes))
    if balanced:
        cw = compute_sample_weight("balanced", y_c_tr)
        kw = compute_sample_weight("balanced", y_k_tr)
        cover_m.fit(x_train, y_c_tr, sample_weight=cw)
        canopy_m.fit(x_train, y_k_tr, sample_weight=kw)
    else:
        cover_m.fit(x_train, y_c_tr)
        canopy_m.fit(x_train, y_k_tr)

    cover_pred = np.array([i_to_cover[i] for i in cover_m.predict(x_test)])
    canopy_pred = np.array([i_to_canopy[i] for i in canopy_m.predict(x_test)])
    return evaluate_predictions(
        name,
        y_cover_test,
        cover_pred,
        y_canopy_test,
        canopy_pred,
        train_rows=len(x_train),
        notes=f"XGBoost hist n_estimators=100 max_depth=3 balanced={balanced}",
        wall_seconds=time.perf_counter() - t0,
    )


def train_hierarchical_lgbm(
    name: str,
    x_train: np.ndarray,
    y_cover_train: np.ndarray,
    y_canopy_train: np.ndarray,
    x_test: np.ndarray,
    y_cover_test: np.ndarray,
    y_canopy_test: np.ndarray,
) -> ExperimentResult:
    """Coarse group → fine cover (LightGBM); canopy is a flat LightGBM head.

    Uses LightGBM instead of sklearn GBM so the multi-model hierarchy finishes
    on ~1M rows in reasonable wall time; hyperparameters match the flat LGBM run
    (n_estimators=100, max_depth=3).
    """
    import lightgbm as lgb

    t0 = time.perf_counter()
    y_coarse_train = np.array([COARSE_OF.get(c, "other") for c in y_cover_train])
    y_coarse_test = np.array([COARSE_OF.get(c, "other") for c in y_cover_test])

    common = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=4,
    )
    coarse = lgb.LGBMClassifier(**common)
    coarse.fit(x_train, y_coarse_train)

    fine_models: dict[str, object] = {}
    single_label: dict[str, str] = {}
    for group in sorted(set(y_coarse_train)):
        mask = y_coarse_train == group
        classes_in_group = sorted(set(y_cover_train[mask]))
        if len(classes_in_group) == 1:
            single_label[group] = classes_in_group[0]
            continue
        m = lgb.LGBMClassifier(**common)
        m.fit(x_train[mask], y_cover_train[mask])
        fine_models[group] = m

    coarse_pred = coarse.predict(x_test)
    cover_pred = np.empty(len(x_test), dtype=object)
    for group in sorted(set(coarse_pred)):
        idx = np.where(coarse_pred == group)[0]
        if group in single_label:
            cover_pred[idx] = single_label[group]
        elif group in fine_models:
            cover_pred[idx] = fine_models[group].predict(x_test[idx])  # type: ignore[union-attr]
        else:
            cover_pred[idx] = Counter(y_cover_train).most_common(1)[0][0]

    canopy_m = lgb.LGBMClassifier(**common)
    canopy_m.fit(x_train, y_canopy_train)
    canopy_pred = canopy_m.predict(x_test)

    coarse_acc = float(accuracy_score(y_coarse_test, coarse_pred))
    # Within-forest fine accuracy when coarse is correct
    forest_idx = np.where(y_coarse_test == "forest")[0]
    forest_fine_acc = None
    if len(forest_idx):
        forest_fine_acc = float(
            accuracy_score(y_cover_test[forest_idx], cover_pred[forest_idx].astype(str))
        )

    return evaluate_predictions(
        name,
        y_cover_test,
        cover_pred.astype(str),
        y_canopy_test,
        canopy_pred,
        train_rows=len(x_train),
        notes=(
            f"Hierarchical LightGBM coarse→fine; coarse_acc_on_test={coarse_acc:.4f}; "
            f"cover_acc_on_true_forest_rows={forest_fine_acc}; "
            f"groups={sorted(set(y_coarse_train))}"
        ),
        wall_seconds=time.perf_counter() - t0,
    )


def main() -> None:
    """Run locked-holdout algorithm and hierarchy experiments; write reports."""
    etl = Path(__file__).resolve().parents[1]
    data_path = etl / "data/processed/labeled_stands.csv"
    stratified_path = etl / "data/processed/labeled_stands_stratified_300k.csv"
    out_dir = etl / "models/experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path}...", flush=True)
    labeled = pd.read_csv(data_path)
    print(f"  {len(labeled):,} rows", flush=True)

    print(
        f"Building locked holdout: test_size={TEST_SIZE}, random_state={RANDOM_STATE}, "
        "stratify=cover_type (same as production train script)...",
        flush=True,
    )
    x_train, x_test, y_cover_train, y_cover_test, y_canopy_train, y_canopy_test = make_split(
        labeled
    )
    print(f"  train={len(x_train):,}  test={len(x_test):,}", flush=True)
    print(
        "  Leakage check: test indices never passed to .fit(); "
        "only x_train/y_*_train used for training.",
        flush=True,
    )

    results: list[ExperimentResult] = []

    # --- Existing committed artifacts on THIS holdout (must match their metadata OA) ---
    results.append(
        eval_existing_artifact(
            "existing_baseline_stand_geonb_v1",
            etl / "models/stand_geonb_v1",
            x_test,
            y_cover_test,
            y_canopy_test,
            train_rows=len(x_train),
            notes="Committed artifact; trained on full CSV with same split seed (no class_weight)",
        )
    )
    print(f"  existing baseline OA={results[-1].overall_accuracy_mean:.4f}", flush=True)

    results.append(
        eval_existing_artifact(
            "existing_balanced_stand_geonb_v1_balanced",
            etl / "models/stand_geonb_v1_balanced",
            x_test,
            y_cover_test,
            y_canopy_test,
            train_rows=len(x_train),
            notes=(
                "Committed artifact; trained on full CSV with same split seed "
                "(class_weight=balanced)"
            ),
        )
    )
    print(f"  existing balanced OA={results[-1].overall_accuracy_mean:.4f}", flush=True)

    # Stratified variant: different training distribution — evaluate on ITS own holdout,
    # and also on the full-data holdout (shows optimistic-smoke vs real distribution).
    print(f"Loading stratified sample {stratified_path}...", flush=True)
    stratified = pd.read_csv(stratified_path)
    xs_tr, xs_te, ys_c_tr, ys_c_te, ys_k_tr, ys_k_te = make_split(stratified)
    results.append(
        eval_existing_artifact(
            "existing_stratified_on_its_own_holdout",
            etl / "models/stand_geonb_v1_stratified",
            xs_te,
            ys_c_te,
            ys_k_te,
            train_rows=len(xs_tr),
            notes=(
                "Committed stratified smoke artifact evaluated on holdout from the SAME "
                "capped 358k sample (optimistic; majority forest classes capped)"
            ),
        )
    )
    print(f"  stratified-on-own OA={results[-1].overall_accuracy_mean:.4f}", flush=True)
    results.append(
        eval_existing_artifact(
            "existing_stratified_on_full_holdout",
            etl / "models/stand_geonb_v1_stratified",
            x_test,
            y_cover_test,
            y_canopy_test,
            train_rows=len(xs_tr),
            notes=(
                "Same stratified artifact scored on the FULL-province holdout "
                "(distribution shift vs its capped training sample)"
            ),
        )
    )
    print(f"  stratified-on-full OA={results[-1].overall_accuracy_mean:.4f}", flush=True)

    # --- Alternative algorithms (train ONLY on x_train) ---
    print("Training LightGBM (unweighted)...", flush=True)
    results.append(
        train_lightgbm(
            "lightgbm_unweighted",
            x_train,
            y_cover_train,
            y_canopy_train,
            x_test,
            y_cover_test,
            y_canopy_test,
            class_weight=None,
        )
    )
    print(
        f"  LGBM unweighted cover={results[-1].cover_accuracy:.4f} "
        f"OA={results[-1].overall_accuracy_mean:.4f} ({results[-1].wall_seconds:.1f}s)",
        flush=True,
    )

    print("Training LightGBM (balanced)...", flush=True)
    results.append(
        train_lightgbm(
            "lightgbm_balanced",
            x_train,
            y_cover_train,
            y_canopy_train,
            x_test,
            y_cover_test,
            y_canopy_test,
            class_weight="balanced",
        )
    )
    print(
        f"  LGBM balanced cover={results[-1].cover_accuracy:.4f} "
        f"OA={results[-1].overall_accuracy_mean:.4f} ({results[-1].wall_seconds:.1f}s)",
        flush=True,
    )

    print("Training XGBoost (unweighted)...", flush=True)
    results.append(
        train_xgboost(
            "xgboost_unweighted",
            x_train,
            y_cover_train,
            y_canopy_train,
            x_test,
            y_cover_test,
            y_canopy_test,
            balanced=False,
        )
    )
    print(
        f"  XGB unweighted cover={results[-1].cover_accuracy:.4f} "
        f"OA={results[-1].overall_accuracy_mean:.4f} ({results[-1].wall_seconds:.1f}s)",
        flush=True,
    )

    print("Training XGBoost (balanced sample_weight)...", flush=True)
    results.append(
        train_xgboost(
            "xgboost_balanced",
            x_train,
            y_cover_train,
            y_canopy_train,
            x_test,
            y_cover_test,
            y_canopy_test,
            balanced=True,
        )
    )
    print(
        f"  XGB balanced cover={results[-1].cover_accuracy:.4f} "
        f"OA={results[-1].overall_accuracy_mean:.4f} ({results[-1].wall_seconds:.1f}s)",
        flush=True,
    )

    print("Training hierarchical LightGBM...", flush=True)
    results.append(
        train_hierarchical_lgbm(
            "hierarchical_lgbm_unweighted",
            x_train,
            y_cover_train,
            y_canopy_train,
            x_test,
            y_cover_test,
            y_canopy_test,
        )
    )
    print(
        f"  Hierarchical cover={results[-1].cover_accuracy:.4f} "
        f"OA={results[-1].overall_accuracy_mean:.4f} ({results[-1].wall_seconds:.1f}s)",
        flush=True,
    )

    payload = {
        "protocol": {
            "features": FEATURE_COLUMNS,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": "cover_type",
            "data": str(data_path),
            "n_rows": len(labeled),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "evaluation": (
                "overall_accuracy_mean = mean(cover_accuracy, canopy_accuracy) "
                "via compute_object_classification_metrics (same as production)"
            ),
            "leakage_controls": [
                "Single train_test_split call; test partition only used in predict/metrics",
                "No hyperparameter search on test",
                "XGBoost label maps built from train classes only",
            ],
        },
        "results": [asdict(r) for r in results],
    }
    out_json = out_dir / "algorithm_hierarchy_experiments.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown summary
    lines = [
        "# Stand classifier algorithm / hierarchy experiments",
        "",
        "## Protocol (unchanged vs production train script)",
        "",
        f"- Data: `{data_path.name}` ({len(labeled):,} rows)",
        f"- Features: `{', '.join(FEATURE_COLUMNS)}`",
        f"- Split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=cover_type",
        f"- Holdout size: {len(x_test):,} rows (never used in `.fit()`)",
        "- Metrics: cover accuracy, canopy accuracy, overall = mean of the two "
        "(same as `compute_object_classification_metrics` / accuracy_report.md)",
        "",
        "## Results",
        "",
        (
            "| Experiment | Cover acc | Canopy acc | Overall (mean) | Cover macro F1 | "
            "Cover weighted F1 | Forest-only cover acc | Conifer recall | "
            "Deciduous recall | Mixed recall | Wall (s) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    row_fmt = (
        "| {name} | {c:.1%} | {k:.1%} | {o:.1%} | {mf:.1%} | {wf:.1%} | "
        "{fa} | {cr} | {dr} | {mr} | {w:.0f} |"
    )
    for r in results:
        lines.append(
            row_fmt.format(
                name=r.name,
                c=r.cover_accuracy,
                k=r.canopy_accuracy,
                o=r.overall_accuracy_mean,
                mf=r.cover_macro_f1,
                wf=r.cover_weighted_f1,
                fa=(
                    f"{r.forest_cover_accuracy:.1%}" if r.forest_cover_accuracy is not None else "—"
                ),
                cr=f"{r.conifer_recall:.1%}" if r.conifer_recall is not None else "—",
                dr=f"{r.deciduous_recall:.1%}" if r.deciduous_recall is not None else "—",
                mr=f"{r.mixed_recall:.1%}" if r.mixed_recall is not None else "—",
                w=r.wall_seconds,
            )
        )
    lines.extend(["", "## Notes per experiment", ""])
    for r in results:
        lines.append(f"- **{r.name}:** {r.notes}")
        lines.append(
            "  Top confusions: "
            + ", ".join(
                f"{c['true']}→{c['predicted']} ({c['count']})" for c in r.top_cover_confusions[:5]
            )
        )
    lines.extend(
        [
            "",
            "## Reading the numbers",
            "",
            "- `existing_stratified_on_its_own_holdout` is the known optimistic smoke test "
            "(majority forest classes capped in the training sample).",
            "- `existing_stratified_on_full_holdout` shows that artifact under the real "
            "province class mix.",
            "- Algorithm swaps (LightGBM / XGBoost / hierarchy) use the **same** full-data "
            "holdout as `existing_baseline_*` / `existing_balanced_*`.",
            "",
        ]
    )
    out_md = out_dir / "algorithm_hierarchy_experiments.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
