"""Train and evaluate an Ultralytics YOLOv8 image classifier."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PrepareYoloDataset import CLASS_NAMES


def resolve_device(device: str = "auto") -> str:
    """Resolve auto to CUDA, Apple MPS, or CPU, in that preference order."""
    if device.lower() != "auto":
        return device
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to select a device. Install the packages in requirements-yolo.txt."
        ) from error

    if torch.cuda.is_available():
        return "0"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _sync_device(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _report_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report

    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
    }


def _save_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    destination: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    figure, axis = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(labels))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(str(destination), dpi=160)
    plt.close(figure)


class YoloClassifier:
    def __init__(
        self,
        weights: str = "yolov8n-cls.pt",
        imgsz: int = 224,
        batch: int = 16,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        if imgsz <= 0 or batch <= 0:
            raise ValueError("imgsz and batch must be positive.")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required. Install the packages in requirements-yolo.txt."
            ) from error

        self.imgsz = imgsz
        self.batch = batch
        self.seed = seed
        self.device = resolve_device(device)
        self.model = YOLO(weights)

    def train(
        self,
        data_dir: str,
        epochs: int = 50,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        project: str = "runs/classify",
        name: str = "tomato-yolov8n-cls",
    ) -> Path:
        data_path = Path(data_dir).expanduser().resolve()
        if not (data_path / "train").is_dir() or not (data_path / "val").is_dir():
            raise ValueError("Dataset must contain train/ and val/ class folders: {}".format(data_path))
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        training_imgsz = self.imgsz if imgsz is None else imgsz
        training_batch = self.batch if batch is None else batch
        if training_imgsz <= 0 or training_batch <= 0:
            raise ValueError("imgsz and batch must be positive.")

        train_device = resolve_device(device) if device is not None else self.device
        self.model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=training_imgsz,
            batch=training_batch,
            seed=self.seed if seed is None else seed,
            device=train_device,
            project=project,
            name=name,
        )
        trainer = getattr(self.model, "trainer", None)
        save_dir = Path(trainer.save_dir) if trainer is not None else Path(project) / name
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.is_file():
            raise FileNotFoundError("Training completed without creating {}".format(best_weights))

        from ultralytics import YOLO

        self.model = YOLO(str(best_weights))
        self.device = train_device
        return best_weights

    def _class_names(self) -> List[str]:
        names = self.model.names
        if isinstance(names, dict):
            model_classes = [str(names[index]) for index in sorted(names)]
        else:
            model_classes = [str(name) for name in names]
        if set(model_classes) != set(CLASS_NAMES):
            raise ValueError(
                "Model classes {} do not match expected classes {}."
                .format(model_classes, list(CLASS_NAMES))
            )
        return model_classes

    def predict(self, image_path: str, device: Optional[str] = None) -> Dict[str, Any]:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("Image does not exist: {}".format(path))
        predict_device = resolve_device(device) if device is not None else self.device
        results = self.model.predict(
            source=str(path),
            imgsz=self.imgsz,
            device=predict_device,
            verbose=False,
        )
        result = results[0]
        class_names = self._class_names()
        index = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        probabilities = result.probs.data.tolist()
        return {
            "class": class_names[index],
            "confidence": confidence,
            "probabilities": {
                class_names[class_index]: float(probability)
                for class_index, probability in enumerate(probabilities)
            },
        }

    def evaluate(
        self,
        data_dir: str,
        output_dir: str = "runs/tomato_yolo_metrics",
        binary_metrics: bool = False,
        batch: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        test_dir = Path(data_dir).expanduser().resolve() / "test"
        if not test_dir.is_dir():
            raise ValueError("Test split directory does not exist: {}".format(test_dir))
        class_names = self._class_names()
        image_paths = []
        y_true = []
        for class_name in class_names:
            class_dir = test_dir / class_name
            if not class_dir.is_dir():
                raise ValueError("Missing test class directory: {}".format(class_dir))
            class_images = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in (".bmp", ".jpeg", ".jpg", ".png")
            )
            if not class_images:
                raise ValueError("No test images found in {}".format(class_dir))
            image_paths.extend(class_images)
            y_true.extend([class_name] * len(class_images))

        predict_device = resolve_device(device) if device is not None else self.device
        predictions = []
        elapsed = 0.0
        batch_size = batch or self.batch
        if batch_size <= 0:
            raise ValueError("batch must be positive.")

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            model_classes = self._class_names()
            _sync_device(predict_device)
            started = time.perf_counter()
            results = self.model.predict(
                source=[str(path) for path in batch_paths],
                imgsz=self.imgsz,
                batch=batch_size,
                device=predict_device,
                verbose=False,
            )
            _sync_device(predict_device)
            elapsed += time.perf_counter() - started
            predictions.extend(
                model_classes[int(result.probs.top1)]
                for result in results
            )
            if len(results) != len(batch_paths):
                raise RuntimeError("Model returned a different number of predictions than input images.")

        metrics = _report_metrics(y_true, predictions, class_names)
        metrics["inference_ms_per_image"] = elapsed * 1000.0 / len(image_paths)
        metrics["test_images"] = len(image_paths)
        metrics["classes"] = class_names

        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        confusion_path = output_path / "confusion_matrix.png"
        _save_confusion_matrix(
            y_true, predictions, class_names, confusion_path, "Tomato disease classification"
        )
        metrics["confusion_matrix"] = str(confusion_path.resolve())

        if binary_metrics:
            binary_true = ["Healthy" if label == "Healthy" else "Diseased" for label in y_true]
            binary_pred = ["Healthy" if label == "Healthy" else "Diseased" for label in predictions]
            binary_labels = ("Healthy", "Diseased")
            binary_report = _report_metrics(binary_true, binary_pred, binary_labels)
            binary_matrix_path = output_path / "binary_confusion_matrix.png"
            _save_confusion_matrix(
                binary_true,
                binary_pred,
                binary_labels,
                binary_matrix_path,
                "Healthy vs Diseased",
            )
            binary_report["confusion_matrix"] = str(binary_matrix_path.resolve())
            metrics["binary_healthy_vs_diseased"] = binary_report

        metrics_path = output_path / "metrics.json"
        metrics["metrics_file"] = str(metrics_path.resolve())
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
            file.write("\n")
        return metrics
