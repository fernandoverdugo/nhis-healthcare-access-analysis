from pathlib import Path
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SOURCES = {
    "adult25.csv": {
        "url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2025/adult25csv.zip",
        "md5": "d8c359ca161e0bd3ef51345626308d72",
    },
    "adultinc25.csv": {
        "url": "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2025/adultinc25csv.zip",
        "md5": "abb6a970181e852a4673ec591451f765",
    },
}


def md5sum(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path, expected):
    actual = md5sum(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {Path(path).name}: expected {expected}, got {actual}")


def download_data(force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, spec in SOURCES.items():
        destination = RAW_DIR / filename
        if destination.exists() and not force:
            _verify(destination, spec["md5"])
            print(f"Verified existing {filename}")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / f"{filename}.zip"
            print(f"Downloading {filename} from CDC/NCHS...")
            request = urllib.request.Request(
                spec["url"],
                headers={"User-Agent": "nhis-healthcare-access-analysis/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            with zipfile.ZipFile(archive) as zf:
                members = [name for name in zf.namelist() if Path(name).name.lower() == filename.lower()]
                if len(members) != 1:
                    raise ValueError(f"Expected exactly one {filename} in {archive.name}")
                with zf.open(members[0]) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out)
        _verify(destination, spec["md5"])
        print(f"Downloaded and verified {filename}")


if __name__ == "__main__":
    download_data()
