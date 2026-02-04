import re
import pandas as pd

# =========================
# CONFIG (ubah sesuai kebutuhan)
# =========================
INPUT_CSV  = "lcs_synthetic_20000.csv"     # atau "/mnt/data/lcs_synthetic_20000.csv"
OUTPUT_CSV = "lcs_clean.csv"

# kalau True: semua fitur biner yang nilainya {1,2} akan diubah jadi {0,1}
# (umumnya: 1=No, 2=Yes -> jadi 0=No, 1=Yes)
CONVERT_1_2_TO_0_1 = True

LABEL_COL = "LUNG_CANCER"  # kolom target
AGE_COL   = "AGE"          # kolom usia


def clean_column_name(col: str) -> str:
    """Bersihkan nama kolom: trim spasi, ganti spasi jadi _, buang karakter aneh."""
    col = col.strip()
    col = re.sub(r"\s+", "_", col)           # spasi beruntun -> _
    col = re.sub(r"[^0-9A-Za-z_]", "", col)  # buang karakter non-alfanumerik
    return col


def main():
    df = pd.read_csv(INPUT_CSV)

    print("== Sebelum cleaning ==")
    print("Shape:", df.shape)

    # 1) Rapikan nama kolom (hapus trailing space, ganti spasi jadi underscore, dll.)
    df.columns = [clean_column_name(c) for c in df.columns]

    # 2) Trim spasi untuk kolom bertipe object (mis. GENDER, LUNG_CANCER)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # 3) Hapus duplikat
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplikat dihapus: {dup_count}")

    # 4) Validasi missing value
    missing_total = int(df.isna().sum().sum())
    print("Total missing value:", missing_total)

    # 5) Encode label (YES/NO -> 1/0)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Kolom label '{LABEL_COL}' tidak ditemukan. Kolom ada: {list(df.columns)}")

    # Samakan huruf label (YES/NO)
    df[LABEL_COL] = df[LABEL_COL].str.upper().str.strip()
    df[LABEL_COL] = df[LABEL_COL].map({"YES": 1, "NO": 0})
    if df[LABEL_COL].isna().any():
        bad = df[df[LABEL_COL].isna()]
        raise ValueError(f"Ada label aneh selain YES/NO. Contoh baris:\n{bad.head()}")

    # 6) Encode gender (M/F -> 1/0)
    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].str.upper().str.strip().map({"M": 1, "F": 0})
        if df["GENDER"].isna().any():
            bad = df[df["GENDER"].isna()]
            raise ValueError(f"Ada nilai GENDER aneh selain M/F. Contoh:\n{bad.head()}")

    # 7) Opsional: ubah semua fitur {1,2} jadi {0,1} (kecuali AGE dan label)
    if CONVERT_1_2_TO_0_1:
        exclude = {LABEL_COL, AGE_COL}
        candidate_cols = [c for c in df.columns if c not in exclude]

        binary_12_cols = []
        for c in candidate_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                uniq = set(pd.Series(df[c].dropna().unique()).astype(int).tolist())
                if uniq.issubset({1, 2}):
                    binary_12_cols.append(c)

        if binary_12_cols:
            df[binary_12_cols] = df[binary_12_cols].replace({1: 0, 2: 1})
            print("Kolom {1,2}->{0,1}:", binary_12_cols)
        else:
            print("Tidak ada kolom biner {1,2} yang terdeteksi.")

    # 8) Ringkasan label (imbalance check)
    label_counts = df[LABEL_COL].value_counts()
    print("\n== Setelah cleaning ==")
    print("Shape:", df.shape)
    print("Distribusi label (0=NO, 1=YES):")
    print(label_counts)
    print("Proporsi YES:", round(label_counts.get(1, 0) / len(df), 4))

    # 9) Simpan hasil clean
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSukses! File hasil cleaning disimpan ke: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
