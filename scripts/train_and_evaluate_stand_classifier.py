"""Train stand classifier on harmonized GeoNB data and emit extended eval report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from terra_core.classification.evaluation import write_accuracy_report_markdown
from terra_core.classification.features import labeled_frame_to_features
from terra_core.classification.models import AccuracyReport
from terra_core.classification.registry import load_model_artifact
from terra_core.classification.training import (
    TrainingConfig,
    load_labeled_dataset,
    train_stand_classifier,
)

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

RARE_CLASSES = {
    "rocky_shore",
    "dune",
    "defense",
    "tidal_flat",
    "scrub",
    "wetland_unknown",
    "recreational",
    "aquatic_bed",
    "beach",
    "coastal_marsh",
}


def _top_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    top_n: int = 15,
) -> list[dict[str, object]]:
    """Return the largest off-diagonal confusion counts."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: list[tuple[int, str, str]] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), true_label, pred_label))
    pairs.sort(reverse=True)
    return [
        {"true": true_label, "predicted": pred_label, "count": count}
        for count, true_label, pred_label in pairs[:top_n]
    ]


def _feature_importances(model, feature_columns: list[str]) -> list[dict[str, float]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    ranked = sorted(
        zip(feature_columns, importances, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"feature": name, "importance": float(value)} for name, value in ranked]


def _mixed_confusion_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    """Summarize where true mixed stands are predicted."""
    mask = y_true == "mixed"
    if not mask.any():
        return {"support": 0, "predicted_distribution": {}}
    preds = y_pred[mask]
    counts = Counter(preds)
    total = int(mask.sum())
    return {
        "support": total,
        "recall": float((preds == "mixed").sum() / total),
        "predicted_distribution": dict(
            sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stand classifier and emit extended eval.")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="data/processed/labeled_stands.csv",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="models/stand_geonb_v1",
    )
    parser.add_argument("test_size", nargs="?", type=float, default=0.2)
    parser.add_argument("n_estimators", nargs="?", type=int, default=100)
    parser.add_argument(
        "--class-weight",
        choices=["balanced"],
        default=None,
        help="Use sklearn balanced sample weights during training.",
    )
    return parser.parse_args()


def main() -> None:
    """Train a stand classifier and write extended evaluation artifacts."""
    args = _parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    test_size = args.test_size
    n_estimators = args.n_estimators
    random_state = 42
    class_weight = args.class_weight

    print(f"Loading {data_path}...", flush=True)
    labeled = load_labeled_dataset(data_path)
    print(f"  {len(labeled):,} rows", flush=True)

    description = "GeoNB harmonized labeled stands v1 (shape + inventory attrs)"
    if class_weight:
        description += f"; class_weight={class_weight}"

    config = TrainingConfig(
        training_data_description=description,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        class_weight=class_weight,
    )
    print(
        f"  n_estimators={n_estimators}, test_size={test_size}, class_weight={class_weight}",
        flush=True,
    )

    print(f"Training with internal {1 - test_size:.0%}/{test_size:.0%} split...", flush=True)
    artifact = train_stand_classifier(
        labeled,
        config,
        output_dir=output_dir,
        feature_columns=FEATURE_COLUMNS,
    )

    x_matrix, y_cover, y_canopy, feature_columns = labeled_frame_to_features(
        labeled,
        feature_columns=FEATURE_COLUMNS,
    )
    split = train_test_split(
        x_matrix,
        y_cover,
        y_canopy,
        test_size=test_size,
        random_state=random_state,
        stratify=y_cover,
    )
    x_test, y_cover_test, y_canopy_test = split[1], split[3], split[5]

    loaded = load_model_artifact(output_dir)
    cover_pred = loaded.cover_type_model.predict(x_test)
    canopy_pred = loaded.canopy_closure_model.predict(x_test)

    cover_pairs = _top_confusion_pairs(y_cover_test, cover_pred)
    canopy_pairs = _top_confusion_pairs(y_canopy_test, canopy_pred)

    metrics = artifact.metadata.validation_metrics
    extended = {
        "model_id": artifact.metadata.model_id,
        "test_size": test_size,
        "train_size": 1 - test_size,
        "class_weight": class_weight,
        "feature_columns": list(artifact.metadata.feature_columns),
        "overall_accuracy": metrics["overall_accuracy"],
        "cover_type_metrics": metrics["cover_type_metrics"],
        "canopy_closure_metrics": metrics["canopy_closure_metrics"],
        "rare_cover_type_metrics": {
            k: v for k, v in metrics["cover_type_metrics"].items() if k in RARE_CLASSES
        },
        "mixed_confusion_breakdown": _mixed_confusion_breakdown(y_cover_test, cover_pred),
        "top_cover_confusions": cover_pairs,
        "top_canopy_confusions": canopy_pairs,
        "cover_feature_importances": _feature_importances(loaded.cover_type_model, feature_columns),
        "canopy_feature_importances": _feature_importances(
            loaded.canopy_closure_model, feature_columns
        ),
    }

    eval_path = output_dir / "extended_eval.json"
    eval_path.write_text(json.dumps(extended, indent=2), encoding="utf-8")

    report = AccuracyReport(
        overall_accuracy=float(metrics["overall_accuracy"]),
        cover_type_metrics=metrics["cover_type_metrics"],
        canopy_closure_metrics=metrics["canopy_closure_metrics"],
        mean_iou=0.0,
        per_class_iou={},
        support={str(k): int(v) for k, v in metrics.get("support", {}).items()},
    )
    write_accuracy_report_markdown(
        report,
        output_dir / "accuracy_report.md",
        model_id=artifact.metadata.model_id,
        training_data_description=config.training_data_description,
    )

    print(f"\nModel saved to {output_dir}")
    print(f"Extended eval: {eval_path}")
    print(f"metadata feature_columns: {list(artifact.metadata.feature_columns)}")


if __name__ == "__main__":
    main()
