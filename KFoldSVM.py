from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
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

def run_five_fold(dataset_roots="PlantVillage", target_size=(224,224), sigma=0.8):
    repo_dir = Path(__file__).resolve().parent
    dataset_dir = repo_dir / dataset_roots

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_paths,y = load_samples(dataset_dir)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_dir}")

    X = build_feature_matrix(image_paths, target_size=target_size, sigma=sigma)
    print(f"matrix X has shape {X.shape}")
    print(f"Class Health (0) : {np.sum(y == 0)}, Diseased (1) : {np.sum(y == 1)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_f1_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_trained_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42)
        svm_model.fit(X_trained_scaled, y_train)
        y_pred = svm_model.predict(X_val_scaled)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, pos_label=1)
        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        print(f"Fold {fold_idx} Accuracy: {acc:.3f}, F1 (diseased): {f1:.4f}")


        print(
            f"Fold {fold_idx} prepared:"
            f"train={X_trained_scaled.shape[0]} samples, val={X_val_scaled.shape[0]} samples"
            )

    print(f"\nMean Accuracy: {np.mean(fold_accuracies) :.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean F1 (diseased): {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")

if __name__ == "__main__":
    run_five_fold(dataset_roots="PlantVillage")