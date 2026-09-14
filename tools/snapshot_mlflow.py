"""Create a consistent deployment snapshot without modifying the training DB."""
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "mlflow.db"
    target = ROOT / "deployment" / "mlflow.db"
    target.parent.mkdir(exist_ok=True)
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)
            result = dst.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError("Snapshot integrity check failed")
            count = dst.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    print(f"Snapshot ready: {count} runs; {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
