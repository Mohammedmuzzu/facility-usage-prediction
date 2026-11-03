"""Immutable local model releases. Verify bytes BEFORE trusted Joblib loading.

The manifest detects corruption; it is not a signature against an attacker who
can replace both manifest and model. Distribute releases in trusted signed images.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import joblib
import pandas as pd
from src.utils import file_hash, save_json

FORMAT_VERSION = 2


class ArtifactError(ValueError):
    """A model release is absent, incompatible or fails its integrity checks."""


def code_hashes() -> dict[str, str]:
    source = Path(__file__).parent
    return {path.name: file_hash(path) for path in sorted(source.glob("*.py"))}


def publish_bundle(bundle: dict, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    release_id = uuid.uuid4().hex
    releases = root / "releases"
    releases.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=releases))
    try:
        bundle["metadata"]["artifact_format"] = FORMAT_VERSION
        bundle["metadata"]["source_hashes"] = code_hashes()
        joblib.dump(bundle, stage / "model_bundle.joblib", compress=3)
        for name, model in bundle["models"].items():
            (stage / name).mkdir()
            joblib.dump(model, stage / name / "model.joblib", compress=3)
        save_json(stage / "model_metadata.json", bundle["metadata"])
        manifest = {
            "format_version": FORMAT_VERSION,
            "release_id": release_id,
            "dependencies": bundle["metadata"]["dependencies"],
            "files": {
                str(path.relative_to(stage)): file_hash(path)
                for path in stage.rglob("*")
                if path.is_file()
            },
        }
        save_json(stage / "manifest.json", manifest)
        destination = releases / release_id
        os.replace(stage, destination)
        # Validate the complete immutable release before activation.
        load_bundle(destination / "model_bundle.joblib", follow_pointer=False)
        # Flat files remain assignment-compatible exports; serving follows current.json.
        for path in destination.rglob("*"):
            if path.is_file():
                target = root / path.relative_to(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".new")
                shutil.copyfile(path, temporary)
                os.replace(temporary, target)
        save_json(root / "current.json", {"release_id": release_id})
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def resolve_release(path: Path) -> Path:
    pointer = path.parent / "current.json"
    if pointer.exists():
        try:
            release_id = json.loads(pointer.read_text())["release_id"]
        except (ValueError, KeyError) as exc:
            raise ArtifactError("Invalid release pointer") from exc
        if (
            not isinstance(release_id, str)
            or len(release_id) != 32
            or any(c not in "0123456789abcdef" for c in release_id)
        ):
            raise ArtifactError("Invalid release ID")
        return path.parent / "releases" / release_id / "model_bundle.joblib"
    return path


def load_bundle(path: Path, follow_pointer: bool = True) -> dict:
    path = resolve_release(path) if follow_pointer else path
    try:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if (
            manifest["format_version"] != FORMAT_VERSION
            or "model_bundle.joblib" not in manifest["files"]
        ):
            raise ArtifactError("Unsupported artifact format")
        for relative, expected in manifest["files"].items():
            target = path.parent / relative
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or target.is_symlink()
            ):
                raise ArtifactError("Unsafe artifact path")
            if not target.is_file() or file_hash(target) != expected:
                raise ArtifactError("Artifact integrity verification failed")
        for package in ["scikit-learn", "numpy", "pandas", "joblib"]:
            if importlib.metadata.version(package) != manifest["dependencies"][package]:
                raise ArtifactError(f"Incompatible runtime dependency: {package}")
        # The deployment image/owner must establish trust in the manifest itself.
        bundle = joblib.load(path)
        if bundle["metadata"]["artifact_format"] != FORMAT_VERSION:
            raise ArtifactError("Unsupported bundle format")
        current = code_hashes()
        for name in [
            "feature_engineering.py",
            "config.py",
            "baselines.py",
            "predict.py",
            "utils.py",
        ]:
            if bundle["metadata"]["source_hashes"].get(name) != current[name]:
                raise ArtifactError(
                    "Inference code differs from the trained release; retrain explicitly"
                )
        expected_features = bundle["metadata"]["feature_names"]
        if any(
            list(model.feature_names_in_) != expected_features
            for model in bundle["models"].values()
        ):
            raise ArtifactError("Model feature schema mismatch")
        return bundle
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, EOFError, TypeError) as exc:
        raise ArtifactError("Model release is missing or invalid") from exc


def activate_release(root: Path, release_id: str) -> None:
    if len(release_id) != 32 or any(c not in "0123456789abcdef" for c in release_id):
        raise ArtifactError("Invalid release ID")
    path = root / "releases" / release_id / "model_bundle.joblib"
    load_bundle(path, follow_pointer=False)
    save_json(root / "current.json", {"release_id": release_id})


def history_contract(data: pd.DataFrame) -> dict:
    # Binding every original observation permits append-only live histories while
    # forbidding deletion, correction or earlier backfills without retraining.
    records = {}
    for row in data.itertuples(index=False):
        records[row.booking_id] = hashlib.sha256(
            "|".join(map(str, row)).encode()
        ).hexdigest()
    return {"records": records, "latest_booking": str(data.booking_timestamp.max())}


def validate_history(data: pd.DataFrame, contract: dict) -> None:
    actual = history_contract(data)
    expected = contract["records"]
    if any(
        actual["records"].get(reference) != digest
        for reference, digest in expected.items()
    ):
        raise ArtifactError(
            "History snapshot was changed or records are missing; retrain explicitly"
        )
    added = data[~data.booking_id.isin(expected)]
    if (
        len(added)
        and (added.booking_timestamp <= pd.Timestamp(contract["latest_booking"])).any()
    ):
        raise ArtifactError("Backfilled observations require retraining")
