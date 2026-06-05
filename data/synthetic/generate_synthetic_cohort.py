"""
Generate a synthetic 79-patient SLE cohort that matches the STRUCTURE of public
schema (e.g. SDY997) without any real data. Writes a CSV and a SQLite metadata
table used by the text-to-SQL demo. No PHI is generated or used.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
N_PATIENTS = 79


def main() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "patient_id": [f"SYN-{i:03d}" for i in range(N_PATIENTS)],
            "age": rng.integers(18, 75, N_PATIENTS),
            "sex": rng.choice(["F", "M"], N_PATIENTS, p=[0.9, 0.1]),
            "sledai_2k": rng.integers(0, 20, N_PATIENTS),
            "albumin_g_dl": rng.normal(3.2, 0.7, N_PATIENTS).round(2),
            "complement_c3": rng.normal(90, 25, N_PATIENTS).round(1),
            "anti_dsdna_iu_ml": rng.integers(0, 400, N_PATIENTS),
            "hcq_responder": rng.choice([0, 1], N_PATIENTS, p=[0.45, 0.55]),
            "endotype": rng.choice(["ATNR", "NRR", "NRNR"], N_PATIENTS, p=[0.13, 0.54, 0.33]),
        }
    )
    csv_path = OUT_DIR / "synthetic_cohort.csv"
    df.to_csv(csv_path, index=False)

    sqlite_path = OUT_DIR / "cohort.sqlite"
    conn = sqlite3.connect(sqlite_path)
    df.to_sql("cohort", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Wrote {csv_path} and {sqlite_path} ({N_PATIENTS} synthetic patients, no PHI).")


if __name__ == "__main__":
    main()
