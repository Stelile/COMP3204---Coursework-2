"""Quickly prepare, train, and evaluate the tomato YOLOv8 classifier."""

import argparse
from pathlib import Path

from PrepareYoloDataset import prepare_dataset
from YoloClassifier import YoloClassifier


def _print_metrics(metrics):
    print("\nThree-class test metrics")
    print("{:<12} {:>10} {:>10} {:>10} {:>10}".format(
        "Class", "Precision", "Recall", "F1-score", "Support"
    ))
    for class_name, scores in metrics["per_class"].items():
        print("{:<12} {:>10.3f} {:>10.3f} {:>10.3f} {:>10}".format(
            class_name,
            scores["precision"],
            scores["recall"],
            scores["f1"],
            scores["support"],
        ))
    print("Accuracy: {:.3f}".format(metrics["accuracy"]))
    print("Macro precision/recall/F1: {:.3f} / {:.3f} / {:.3f}".format(
        metrics["macro_precision"],
        metrics["macro_recall"],
        metrics["macro_f1"],
    ))
    print("Inference: {:.3f} ms/image".format(metrics["inference_ms_per_image"]))
    print("Confusion matrix: {}".format(metrics["confusion_matrix"]))
    if "binary_healthy_vs_diseased" in metrics:
        binary = metrics["binary_healthy_vs_diseased"]
        print("\nBinary Healthy vs Diseased")
        print("Accuracy: {:.3f}".format(binary["accuracy"]))
        print("Macro precision/recall/F1: {:.3f} / {:.3f} / {:.3f}".format(
            binary["macro_precision"],
            binary["macro_recall"],
            binary["macro_f1"],
        ))
        print("Confusion matrix: {}".format(binary["confusion_matrix"]))
    print("Metrics JSON: {}".format(metrics["metrics_file"]))


def main():
    repository_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(repository_dir / "PlantVillage.zip"),
        help="PlantVillage ZIP archive or extracted folder",
    )
    parser.add_argument("--data-dir", default=str(repository_dir / "yolo_dataset"))
    parser.add_argument("--split-manifest", help="Reuse a previous split_manifest.csv")
    parser.add_argument("--epochs", type=int, default=1, help="Defaults to one epoch for a quick smoke run")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or a CUDA device such as 0")
    parser.add_argument("--binary-metrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing prepared dataset")
    args = parser.parse_args()

    data_dir = prepare_dataset(
        args.source,
        args.data_dir,
        seed=args.seed,
        image_size=args.imgsz,
        split_manifest=args.split_manifest,
        overwrite=args.overwrite,
    )
    classifier = YoloClassifier(
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
    )
    best_weights = classifier.train(
        str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
    )
    print("Best weights: {}".format(best_weights))

    metrics = classifier.evaluate(
        str(data_dir),
        binary_metrics=args.binary_metrics,
        batch=args.batch,
        device=args.device,
    )
    _print_metrics(metrics)


if __name__ == "__main__":
    main()
