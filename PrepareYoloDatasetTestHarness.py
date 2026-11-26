"""Unit tests for deterministic PlantVillage split preparation."""

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from PrepareYoloDataset import load_split_manifest, prepare_dataset


class TestYoloDatasetSplits(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "PlantVillage"
        class_folders = (
            "Tomato___Early_blight",
            "Tomato___Tomato_healthy",
            "Tomato___Septoria_leaf_spot",
        )
        for class_index, class_folder in enumerate(class_folders):
            folder = self.source / class_folder
            folder.mkdir(parents=True)
            for image_index in range(8):
                image = Image.new(
                    "RGB",
                    (40 + image_index, 32),
                    (class_index * 60, image_index * 10, 100),
                )
                image.save(folder / "leaf_{}.jpg".format(image_index))

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _assignments(manifest_path):
        assignments, seed = load_split_manifest(str(manifest_path))
        return assignments, seed

    def test_seeded_splits_are_repeatable_and_cover_every_class(self):
        first = prepare_dataset(
            str(self.source), str(self.root / "first"), seed=37, image_size=224
        )
        second = prepare_dataset(
            str(self.source), str(self.root / "second"), seed=37, image_size=224
        )

        first_assignments, first_seed = self._assignments(first / "split_manifest.csv")
        second_assignments, second_seed = self._assignments(second / "split_manifest.csv")
        self.assertEqual(first_seed, 37)
        self.assertEqual(second_seed, 37)
        self.assertEqual(first_assignments, second_assignments)
        for class_name in ("Blight", "Healthy", "Septoria"):
            class_splits = {
                split
                for label, split in first_assignments.values()
                if label == class_name
            }
            self.assertEqual(class_splits, {"train", "val", "test"})
            for split in ("train", "val", "test"):
                self.assertTrue((first / split / class_name).is_dir())

    def test_manifest_replay_preserves_assignments_and_seed(self):
        original = prepare_dataset(
            str(self.source), str(self.root / "original"), seed=11
        )
        replayed = prepare_dataset(
            str(self.source),
            str(self.root / "replayed"),
            seed=11,
            split_manifest=str(original / "split_manifest.csv"),
        )
        original_assignments, _ = self._assignments(original / "split_manifest.csv")
        replayed_assignments, replayed_seed = self._assignments(replayed / "split_manifest.csv")
        self.assertEqual(original_assignments, replayed_assignments)
        self.assertEqual(replayed_seed, 11)

        with self.assertRaisesRegex(ValueError, "Use the same seed"):
            prepare_dataset(
                str(self.source),
                str(self.root / "wrong-seed"),
                seed=12,
                split_manifest=str(original / "split_manifest.csv"),
            )

    def test_manifest_has_unique_complete_source_assignments(self):
        output = prepare_dataset(str(self.source), str(self.root / "dataset"), seed=5)
        with (output / "split_manifest.csv").open(
            "r", newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))
        sources = [row["source"] for row in rows]
        self.assertEqual(len(sources), len(set(sources)))
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["seed"] == "5" for row in rows))
        for row in rows:
            output_path = output / row["output"]
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.parts[-3:-1], (row["split"], row["class"]))

        manifest_test_files = {
            output / row["output"] for row in rows if row["split"] == "test"
        }
        actual_test_files = set((output / "test").glob("*/*.jpg"))
        self.assertEqual(manifest_test_files, actual_test_files)

    def test_zip_and_extracted_sources_produce_identical_assignments(self):
        archive_path = self.root / "PlantVillage.zip"
        with zipfile.ZipFile(str(archive_path), "w") as archive:
            for image_path in self.source.rglob("*.jpg"):
                archive.write(
                    str(image_path),
                    image_path.relative_to(self.source).as_posix(),
                )

        extracted_output = prepare_dataset(
            str(self.source), str(self.root / "from-directory"), seed=19
        )
        archive_output = prepare_dataset(
            str(archive_path), str(self.root / "from-archive"), seed=19
        )
        extracted_assignments, _ = self._assignments(
            extracted_output / "split_manifest.csv"
        )
        archive_assignments, _ = self._assignments(
            archive_output / "split_manifest.csv"
        )
        self.assertEqual(extracted_assignments, archive_assignments)
        with (archive_output / "split_manifest.csv").open(
            "r", newline="", encoding="utf-8"
        ) as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(
            {
                archive_output / row["output"] for row in rows if row["split"] == "test"
            },
            set((archive_output / "test").glob("*/*.jpg")),
        )

    def test_invalid_fractions_and_existing_output_are_rejected(self):
        output = prepare_dataset(
            str(self.source), str(self.root / "protected-output"), seed=7
        )
        with self.assertRaises(FileExistsError):
            prepare_dataset(str(self.source), str(output), seed=7)
        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            prepare_dataset(
                str(self.source),
                str(self.root / "invalid-fractions"),
                val_fraction=0.6,
                test_fraction=0.4,
            )
        prepare_dataset(str(self.source), str(output), seed=7, overwrite=True)

    def test_missing_tomato_class_is_reported(self):
        incomplete_source = self.root / "incomplete"
        for class_folder in (
            "Tomato___Early_blight",
            "Tomato___Tomato_healthy",
        ):
            folder = incomplete_source / class_folder
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 32), (20, 40, 60)).save(folder / "leaf.jpg")

        with self.assertRaisesRegex(ValueError, "missing tomato classes: Septoria"):
            prepare_dataset(str(incomplete_source), str(self.root / "incomplete-out"))

    def test_corrupt_image_fails_with_source_context_and_cleans_staging(self):
        corrupt_image = self.source / "Tomato___Early_blight" / "bad.jpg"
        corrupt_image.write_bytes(b"not an image")
        output = self.root / "corrupt-image-out"

        with self.assertRaisesRegex(
            ValueError, "Could not read image 'Tomato___Early_blight/bad.jpg'"
        ):
            prepare_dataset(str(self.source), str(output), seed=3)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("corrupt-image-out.tmp-*")), [])

    def test_invalid_zip_archive_is_reported(self):
        archive_path = self.root / "broken.zip"
        archive_path.write_text("not a ZIP file", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "not a readable ZIP archive"):
            prepare_dataset(str(archive_path), str(self.root / "broken-zip-out"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
