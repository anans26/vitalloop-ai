"""Fetches the primary dataset (UCI Diabetes 130-US Hospitals, id=296) into datasets/raw/."""

from pathlib import Path

from ucimlrepo import fetch_ucirepo

RAW_DIR = Path(__file__).resolve().parents[2] / "datasets" / "raw"
UCI_DATASET_ID = 296


def download_diabetes_130(output_dir: Path = RAW_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "diabetic_data.csv"

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = dataset.data.ids.join(dataset.data.features)
    df["readmitted"] = dataset.data.targets["readmitted"]
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = download_diabetes_130()
    print(f"Saved raw dataset to {path}")
