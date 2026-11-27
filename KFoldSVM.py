from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

from ImagePreprocessing import preprocess_image
from ConcatenateFeature import extract_feature

def load_samples(dataset_dir: Path, extensions=(".jpg",".jpeg",".png")):
    image_paths = []
    labels = []

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        folder_name = class_dir.name.lower()
        if folder_name.startswith("."):
            continue

        label = 0 if "healthy" in folder_name else 1

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in extensions:
                image_paths.append(img_path)
                labels.append(label)
    return image_paths, np.array(labels, dtype=np.int64)

def build_feature_matrix(image_paths, target_size=(224,224), sigma=0.8):
    feature_list = []
    total = len(image_paths)

    for idx, path in enumerate(image_paths, start=1):
        img_hsv = preprocess_image(str(path), target_size=target_size, sigma=sigma)
        vec = np.asarray(extract_feature(img_hsv, mask=None), dtype=np.float64).ravel()

        if not np.issubdtype(vec.dtype, np.number):
            bad_types = {type(item).__name__ for item in vec if not isinstance(item, (int, float, np.number))}
            raise TypeError(f"Non-numeric feature values in {path}: {sorted(bad_types)}")

        feature_list.append(vec)

    X = np.vstack(feature_list).astype(np.float64)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix X contains NaN or infinite values")
    
    return X

def plot_split_distribution(y_train_val, y_test, output_path: Path):
    tv_counts = [int(np.sum(y_train_val == 0)), int(np.sum(y_train_val == 1))]
    test_counts = [int(np.sum(y_test == 0)), int(np.sum(y_test == 1))]

    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7,4.5))
    bars_tv = ax.bar(x - width / 2, tv_counts, width, label="Train + Val (80%)")
    bars_test = ax.bar(x + width / 2, test_counts, width, label="Test Set (20%)")

    ax.set_xticks(x)
    ax.set_xticklabels(["Health (0)", "Diseased (1)"])
    ax.set_ylabel("Number of Images")
    ax.set_title("Stratified Class Distribution: Train+Val vs. Test")

    for bars in (bars_tv, bars_test):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_tuning_results(tuning_records, output_path: Path):
    labels = [f"{r['kernel']} (C={r['C']})" for r in tuning_records]
    mean_f1s = [r["mean_val_f1"] for r in tuning_records]
    mean_accs = [r["mean_val_acc"] for r in tuning_records]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5,4.8))
    ax.bar(x - width / 2, mean_accs, width, label="Mean Val Accuracy")
    ax.bar(x + width / 2, mean_f1s, width, label="Mean Val F1 (Diseased)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("5-Fold Cross-Validation Performance by SVM Configuration")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_val_vs_test(fold_accs, fold_f1s, test_acc, test_f1, best_cfg, output_path: Path):
    folds = [f"Fold {i}" for i in range(1, len(fold_accs) + 1)] + ["Mean Val", "Test"]
    acc_vals = list(fold_accs) + [float(np.mean(fold_accs)), float(test_acc)]
    f1_vals = list(fold_f1s) + [float(np.mean(fold_f1s)), float(test_f1)]

    x = np.arange(len(folds))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, acc_vals, width, label="Accuracy")
    ax.bar(x + width / 2, f1_vals, width, label="F1 (Diseased)")

    ax.axvline(len(fold_accs) - 0.5, color="grey", linestyle=":", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"Validation vs. Test Performance — Tuned SVM ({best_cfg['kernel']}, C={best_cfg['C']})"
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_test_confusion_matrix(y_test, y_test_pred, output_path: Path):
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy (0)", "Diseased (1)"],)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Test Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def run_train_val_test_pipeline(dataset_roots="PlantVillage", target_size=(224, 224), sigma=0.8, test_size=0.2, random_state=42):
    repo_dir = Path(__file__).resolve().parent
    dataset_dir = repo_dir / dataset_roots
    plots_dir = repo_dir / "visualisations"
    plots_dir.mkdir(exist_ok = True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_paths, y = load_samples(dataset_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No images in {dataset_dir}")

    X = build_feature_matrix(image_paths, target_size=target_size, sigma=sigma)
    print(f"Matrix X has shape {X.shape}")
    print(f"Class Healthy (0): {np.sum(y == 0)}, Diseased (1): {np.sum(y == 1)}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state,)
    print(f"Split sizes -> Train+Val: {X_train_val.shape[0]} samples, "f"Test: {X_test.shape[0]} samples")
    plot_split_distribution(y_train_val, y_test, plots_dir / "dataset_split_distributions.png")
    candidate_configs = [{"kernel": "linear", "C": 0.1, "gamma": "scale"},{"kernel": "linear", "C": 1.0, "gamma": "scale"},{"kernel": "rbf", "C": 1.0, "gamma": "scale"},{"kernel": "rbf", "C": 10.0, "gamma": "scale"},]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    tuning_records = []

    for cfg in candidate_configs:
        fold_accs = []
        fold_f1s = []

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_val, y_train_val), start=1):
            X_tr, X_va = X_train_val[tr_idx], X_train_val[va_idx]
            y_tr, y_va = y_train_val[tr_idx], y_train_val[va_idx]

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_va_scaled = scaler.transform(X_va)

            model = SVC(kernel=cfg["kernel"],C=cfg["C"],gamma=cfg["gamma"],class_weight="balanced",cache_size=1000,random_state=random_state,)
            model.fit(X_tr_scaled, y_tr)
            y_va_pred = model.predict(X_va_scaled)

            acc = accuracy_score(y_va, y_va_pred)
            f1 = f1_score(y_va, y_va_pred, pos_label=1)
            fold_accs.append(acc)
            fold_f1s.append(f1)

        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        std_f1 = float(np.std(fold_f1s))

        tuning_records.append({"kernel": cfg["kernel"],"C": cfg["C"],"gamma": cfg["gamma"],"fold_accs": fold_accs,"fold_f1s": fold_f1s,"mean_val_acc": mean_acc,"std_val_acc": std_acc,"mean_val_f1": mean_f1,"std_val_f1": std_f1,})

        print(f"Config [kernel={cfg['kernel']}, C={cfg['C']}] -> " f"Val Acc: {mean_acc:.4f} (+/- {std_acc:.4f}), " f"Val F1 (diseased): {mean_f1:.4f} (+/- {std_f1:.4f})")
    plot_tuning_results(tuning_records, plots_dir / "svm_validation_tuning.png")

    best_record = max(
        tuning_records,
        key=lambda r: (r["mean_val_f1"], r["mean_val_acc"]),
    )
    print(f"\nSelected best SVM -> kernel={best_record['kernel']}, C={best_record['C']} " f"(Val F1: {best_record['mean_val_f1']:.4f}, Val Acc: {best_record['mean_val_acc']:.4f})")

    final_scaler = StandardScaler()
    X_train_val_scaled = final_scaler.fit_transform(X_train_val)
    X_test_scaled = final_scaler.transform(X_test)

    final_svm = SVC(kernel=best_record["kernel"],C=best_record["C"],gamma=best_record["gamma"],class_weight="balanced",cache_size=1000,random_state=random_state,)
    final_svm.fit(X_train_val_scaled, y_train_val)
    y_test_pred = final_svm.predict(X_test_scaled)

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, pos_label=1)
    test_rec = recall_score(y_test, y_test_pred, pos_label=1)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=1)

    print("Test set verification")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test precision (diseased): {test_prec:.4f}")
    print(f"Test recall (diseased): {test_rec:.4f}")
    print(f"Test F1 (diseased): {test_f1:.4f}")

    plot_val_vs_test(best_record["fold_accs"],best_record["fold_f1s"],test_acc,test_f1,best_record,plots_dir / "validation_vs_test_performance.png",)
    plot_test_confusion_matrix(y_test, y_test_pred, plots_dir / "test_confusion_matrix.png")
    print(f"\n Saved visualisations to: {plots_dir}")

if __name__ == "__main__":
    run_train_val_test_pipeline(dataset_roots="PlantVillage")