from io import StringIO
from pathlib import Path
import json
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RTOL = 1e-8
ATOL = 1e-10


def _git_text(path):
    relative = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _tracked_outputs():
    result = subprocess.run(
        ["git", "ls-files", "data/processed"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def _compare_csv(path):
    committed = pd.read_csv(StringIO(_git_text(path)))
    rebuilt = pd.read_csv(path)
    if list(committed.columns) != list(rebuilt.columns):
        raise AssertionError(f"Column mismatch in {path.relative_to(ROOT)}")
    if committed.shape != rebuilt.shape:
        raise AssertionError(f"Shape mismatch in {path.relative_to(ROOT)}")

    for column in committed.columns:
        left = committed[column]
        right = rebuilt[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=RTOL,
                atol=ATOL,
                equal_nan=True,
            ):
                delta = np.nanmax(
                    np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))
                )
                raise AssertionError(
                    f"Numeric mismatch in {path.relative_to(ROOT)} column {column}; max abs diff={delta}"
                )
        else:
            if not left.fillna("<NA>").astype(str).equals(
                right.fillna("<NA>").astype(str)
            ):
                raise AssertionError(
                    f"Text mismatch in {path.relative_to(ROOT)} column {column}"
                )


def _compare_json_values(left, right, label):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"JSON key mismatch at {label}")
        for key in left:
            _compare_json_values(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(f"JSON length mismatch at {label}")
        for index, (a, b) in enumerate(zip(left, right)):
            _compare_json_values(a, b, f"{label}[{index}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isclose(float(left), float(right), rtol=RTOL, atol=ATOL, equal_nan=True):
            raise AssertionError(f"Numeric JSON mismatch at {label}: {left} != {right}")
        return
    if left != right:
        raise AssertionError(f"JSON mismatch at {label}: {left!r} != {right!r}")


def _compare_json(path):
    committed = json.loads(_git_text(path))
    rebuilt = json.loads(path.read_text())
    _compare_json_values(committed, rebuilt, path.relative_to(ROOT).as_posix())


def main():
    outputs = _tracked_outputs()
    if not outputs:
        raise RuntimeError("No tracked analytical outputs found.")

    for path in outputs:
        if not path.exists():
            raise FileNotFoundError(f"Rebuilt output is missing: {path.relative_to(ROOT)}")
        if path.suffix == ".csv":
            _compare_csv(path)
        elif path.suffix == ".json":
            _compare_json(path)
        else:
            if _git_text(path) != path.read_text():
                raise AssertionError(f"Text mismatch in {path.relative_to(ROOT)}")
        print(f"Reproduced {path.relative_to(ROOT)}")

    print(f"Reproducibility check passed for {len(outputs)} tracked analytical outputs.")


if __name__ == "__main__":
    main()
