"""Prepare PlantVillage tomato images for Ultralytics classification."""

import argparse
import csv
import hashlib
import random
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


CLASS_NAMES = ("Blight", "Healthy", "Septoria")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
MANIFEST_FIELDS = ("source", "class", "split", "seed", "output")


@dataclass(frozen=True)
class ImageRecord:
    source: str
    class_name: str
    path: Optional[Path] = None
    archive_member: Optional[str] = None


def _class_from_folder(folder_name: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", folder_name.lower()).strip()
    if "early blight" in normalized or "late blight" in normalized:
        return "Blight"
    if "septoria" in normalized:
        return "Septoria"
    if "healthy" in normalized and ("tomato" in normalized or normalized == "healthy"):
        return "Healthy"
    return None


def _discover_directory(source: Path) -> List[ImageRecord]:
    records = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        class_folder = None
        class_name = None
        for parent in path.parents:
            if parent == source.parent:
                break
            class_name = _class_from_folder(parent.name)
            if class_name is not None:
                class_folder = parent
                break
            if parent == source:
                break

        if class_folder is not None and class_name is not None:
            relative_path = path.relative_to(class_folder).as_posix()
            source_id = "{}/{}".format(class_folder.name, relative_path)
            records.append(ImageRecord(source_id, class_name, path=path))
    return records


def _discover_archive(source: Path) -> List[ImageRecord]:
    try:
        with zipfile.ZipFile(str(source)) as archive:
            records = []
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                parts = Path(member.filename).parts
                selected = None
                for index in range(len(parts) - 2, -1, -1):
                    class_name = _class_from_folder(parts[index])
                    if class_name is not None:
                        selected = (index, class_name)
                        break
                if selected is None:
                    continue

                index, class_name = selected
                source_id = "/".join(parts[index:])
                records.append(
                    ImageRecord(
                        source_id,
                        class_name,
                        archive_member=member.filename,
                    )
                )
            return records
    except zipfile.BadZipFile as error:
        try:
            with source.open("rb") as file:
                header = file.read(128).decode("utf-8", errors="ignore")
        except OSError:
            header = ""
        if "git-lfs.github.com/spec" in header:
            raise ValueError(
                "{} is a Git LFS pointer, not the dataset archive. Run "
                "'git lfs pull' or pass the path to an extracted PlantVillage folder."
                .format(source)
            ) from error
        raise ValueError("{} is not a readable ZIP archive.".format(source)) from error


def load_split_manifest(path: str) -> Tuple[Dict[str, Tuple[str, str]], Optional[int]]:
    """Load source-to-class/split assignments and their recorded random seed."""
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"source", "class", "split"}.issubset(reader.fieldnames):
            raise ValueError("Split manifest must have source, class, and split columns.")

        assignments = {}
        seeds = set()
        for row in reader:
            source_id = row["source"]
            class_name = row["class"]
            split = row["split"]
            if class_name not in CLASS_NAMES or split not in ("train", "val", "test"):
                raise ValueError("Invalid class or split in manifest for {!r}.".format(source_id))
            if source_id in assignments:
                raise ValueError("Duplicate source entry in split manifest: {!r}".format(source_id))
            assignments[source_id] = (class_name, split)
            if "seed" in reader.fieldnames:
                try:
                    seeds.add(int(row["seed"]))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Invalid or missing seed in split manifest for {!r}.".format(source_id)
                    ) from error
    if len(seeds) > 1:
        raise ValueError("Split manifest contains inconsistent random seeds.")
    return assignments, next(iter(seeds)) if seeds else None


def _make_splits(
    records: List[ImageRecord],
    seed: int,
    val_fraction: float,
    test_fraction: float,
    manifest_path: Optional[Path],
) -> Dict[str, str]:
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must each be between 0 and 1.")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must sum to less than 1.")

    record_by_source = {record.source: record for record in records}
    if len(record_by_source) != len(records):
        raise ValueError("Source image identifiers are not unique.")

    if manifest_path is not None:
        assignments, manifest_seed = load_split_manifest(str(manifest_path))
        if manifest_seed is not None and manifest_seed != seed:
            raise ValueError(
                "Split manifest uses seed {}; requested seed is {}. Use the same seed."
                .format(manifest_seed, seed)
            )
        if set(assignments) != set(record_by_source):
            missing = len(set(record_by_source) - set(assignments))
            extra = len(set(assignments) - set(record_by_source))
            raise ValueError(
                "Manifest sources do not match this dataset ({} missing, {} extra)."
                .format(missing, extra)
            )
        for source_id, (class_name, _) in assignments.items():
            if record_by_source[source_id].class_name != class_name:
                raise ValueError("Manifest class does not match source {!r}.".format(source_id))
        for class_name in CLASS_NAMES:
            class_splits = {
                assignments[record.source][1]
                for record in records
                if record.class_name == class_name
            }
            if class_splits != {"train", "val", "test"}:
                raise ValueError(
                    "Manifest must assign {} images to train, val, and test.".format(class_name)
                )
        return {source_id: split for source_id, (_, split) in assignments.items()}

    rng = random.Random(seed)
    assignments = {}
    for class_name in CLASS_NAMES:
        class_records = sorted(
            (record for record in records if record.class_name == class_name),
            key=lambda record: record.source,
        )
        if len(class_records) < 3:
            raise ValueError(
                "{} needs at least 3 images to appear in train, val, and test; found {}."
                .format(class_name, len(class_records))
            )
        rng.shuffle(class_records)
        val_count = max(1, round(len(class_records) * val_fraction))
        test_count = max(1, round(len(class_records) * test_fraction))
        while val_count + test_count >= len(class_records):
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError("Split fractions leave no training images for {}.".format(class_name))

        for record in class_records[:test_count]:
            assignments[record.source] = "test"
        for record in class_records[test_count:test_count + val_count]:
            assignments[record.source] = "val"
        for record in class_records[test_count + val_count:]:
            assignments[record.source] = "train"
    return assignments


def _save_image(image: Image.Image, destination: Path, image_size: Optional[int]) -> None:
    image = image.convert("RGB")
    if image_size is not None:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize((image_size, image_size), resampling.LANCZOS)
    image.save(str(destination), format="JPEG", quality=95)


def prepare_dataset(
    source: str,
    output_dir: str,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    image_size: Optional[int] = 224,
    split_manifest: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Create train/val/test class folders and a reusable split_manifest.csv."""
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError("Dataset source does not exist: {}".format(source_path))
    if image_size is not None and image_size <= 0:
        raise ValueError("image_size must be positive or None.")

    if source_path == output_path or source_path in output_path.parents or output_path in source_path.parents:
        raise ValueError("Dataset source and output directory must not contain one another.")

    if source_path.is_dir():
        records = _discover_directory(source_path)
    elif source_path.is_file() and source_path.suffix.lower() == ".zip":
        records = _discover_archive(source_path)
    else:
        raise ValueError("Source must be a PlantVillage ZIP archive or extracted folder.")
    if not records:
        raise ValueError("No supported tomato image classes were found in {}.".format(source_path))

    found_classes = {record.class_name for record in records}
    missing_classes = set(CLASS_NAMES) - found_classes
    if missing_classes:
        raise ValueError("Dataset is missing tomato classes: {}.".format(", ".join(sorted(missing_classes))))

    manifest_path = Path(split_manifest).expanduser().resolve() if split_manifest else None
    assignments = _make_splits(records, seed, val_fraction, test_fraction, manifest_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError("Output path is not a directory: {}".format(output_path))
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            "Output directory is not empty: {}. Choose another path or pass overwrite=True."
            .format(output_path)
        )

    staging = Path(tempfile.mkdtemp(prefix=output_path.name + ".tmp-", dir=str(output_path.parent)))
    archive = None
    manifest_rows = []
    try:
        if source_path.is_file():
            archive = zipfile.ZipFile(str(source_path))
        for record in records:
            split = assignments[record.source]
            digest = hashlib.sha256(record.source.encode("utf-8")).hexdigest()[:12]
            stem = Path(record.source).stem
            relative_output = Path(split) / record.class_name / "{}_{}.jpg".format(stem, digest)
            destination = staging / relative_output
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                if archive is not None:
                    with archive.open(record.archive_member) as image_file:
                        with Image.open(image_file) as image:
                            _save_image(image, destination, image_size)
                else:
                    with Image.open(str(record.path)) as image:
                        _save_image(image, destination, image_size)
            except Exception as error:
                raise ValueError("Could not read image {!r}: {}".format(record.source, error)) from error

            manifest_rows.append(
                {
                    "source": record.source,
                    "class": record.class_name,
                    "split": split,
                    "seed": seed,
                    "output": relative_output.as_posix(),
                }
            )

        with (staging / "split_manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(sorted(manifest_rows, key=lambda row: row["source"]))

        if output_path.exists():
            shutil.rmtree(str(output_path))
        staging.rename(output_path)
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        raise
    finally:
        if archive is not None:
            archive.close()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="PlantVillage ZIP archive or extracted dataset folder")
    parser.add_argument("output_dir", help="Output directory for the Ultralytics dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Center-crop and resize to this square size (default: 224); use 0 to preserve size",
    )
    parser.add_argument("--split-manifest", help="Reuse a prior split_manifest.csv")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory")
    args = parser.parse_args()

    result = prepare_dataset(
        args.source,
        args.output_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        image_size=args.image_size or None,
        split_manifest=args.split_manifest,
        overwrite=args.overwrite,
    )
    print("Prepared YOLO classification dataset: {}".format(result))
    print("Use {} with the SVM pipeline to reproduce these splits.".format(result / "split_manifest.csv"))


if __name__ == "__main__":
    main()
