"""
Medis TXT export parser and overview generator.

Scans lumen_and_wall_eser/ and lumen_and_wall_robin/ directories,
extracts patient ID, vessel name, and maps to Study/Series UIDs.
"""

import re
from pathlib import Path
import pandas as pd


FLOW_ROOT = Path(r"C:\Users\lukass\Desktop\personal\data\flow")
ESER_DIR = FLOW_ROOT / "lumen_and_wall_eser"
ROBIN_DIR = FLOW_ROOT / "lumen_and_wall_robin"
UIDS_CSV = FLOW_ROOT / "study_series_uid" / "clean_uids.csv"


def parse_medis_header(filepath: Path) -> dict:
    """Parse GENERAL_INFO block from a Medis TXT export."""
    info = {
        "file_path": str(filepath),
        "patient_id": None,
        "vessel_name": None,
        "series_description": None,
        "study_description": None,
    }
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                break  # end of header
            # parse key : value
            if " : " in line:
                key, val = line.split(" : ", 1)
                key = key.lstrip("# ").strip()
                val = val.strip()
                if key == "patient_id":
                    info["patient_id"] = val
                elif key == "vessel_name":
                    info["vessel_name"] = val.upper()  # normalize LAD, LCX, RCA
                elif key == "series_description":
                    info["series_description"] = val if val != "None" else None
                elif key == "study_description":
                    info["study_description"] = val
    return info


def scan_reader_dir(reader_dir: Path, reader_name: str) -> list[dict]:
    """Scan all TXT files in a reader directory."""
    records = []
    for txt_file in reader_dir.glob("*.txt"):
        info = parse_medis_header(txt_file)
        info["reader"] = reader_name
        info["filename"] = txt_file.name
        records.append(info)
    return records


def load_uids() -> pd.DataFrame:
    """Load clean_uids.csv and return DataFrame."""
    df = pd.read_csv(UIDS_CSV, sep=";")
    # Prefer 'Session' source if duplicates exist
    df = df.sort_values("Source", ascending=False)  # Session > Button
    df = df.drop_duplicates(subset=["PatientID"], keep="first")
    return df


def build_overview(reader_dir: Path, reader_name: str) -> pd.DataFrame:
    """Build overview DataFrame with UID mapping for a single reader."""
    # Scan reader directory
    records = scan_reader_dir(reader_dir, reader_name)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Load UIDs
    uids = load_uids()

    # Merge on patient_id
    df = df.merge(
        uids[["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]],
        left_on="patient_id",
        right_on="PatientID",
        how="left",
    )
    df = df.drop(columns=["PatientID"])

    # Sort for readability
    df = df.sort_values(["patient_id", "vessel_name"])

    # Reorder columns
    cols = [
        "patient_id",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "vessel_name",
        "reader",
        "filename",
    ]
    df = df[[c for c in cols if c in df.columns]]

    return df


def main():
    print(f"Scanning Medis exports in {FLOW_ROOT}...")

    df_eser = build_overview(ESER_DIR, "Eser")
    df_robin = build_overview(ROBIN_DIR, "Robin")

    out_eser = FLOW_ROOT / "medis_overview_eser.csv"
    out_robin = FLOW_ROOT / "medis_overview_robin.csv"

    df_eser.to_csv(out_eser, index=False, sep=";")
    df_robin.to_csv(out_robin, index=False, sep=";")

    print(f"\nSaved ESER overview: {out_eser}")
    print(f"Rows: {len(df_eser)} | Unique patients: {df_eser['patient_id'].nunique()}")
    if not df_eser.empty:
        print("Vessels (Eser):")
        print(df_eser.groupby("vessel_name").size())

    print(f"\nSaved ROBIN overview: {out_robin}")
    print(f"Rows: {len(df_robin)} | Unique patients: {df_robin['patient_id'].nunique()}")
    if not df_robin.empty:
        print("Vessels (Robin):")
        print(df_robin.groupby("vessel_name").size())


def main2():
    """Build unified Medis overview from per-reader CSVs."""
    print("\nBuilding unified Medis overview from per-reader CSVs...")

    eser_path = FLOW_ROOT / "medis_overview_eser.csv"
    robin_path = FLOW_ROOT / "medis_overview_robin.csv"

    df_list: list[pd.DataFrame] = []

    if eser_path.exists():
        df_list.append(pd.read_csv(eser_path, sep=";"))
    else:
        print(f"Warning: {eser_path} not found")

    if robin_path.exists():
        df_list.append(pd.read_csv(robin_path, sep=";"))
    else:
        print(f"Warning: {robin_path} not found")

    if not df_list:
        print("No per-reader overview CSVs found. Run main() first.")
        return

    df = pd.concat(df_list, ignore_index=True)

    # Sort for readability if columns are present
    sort_cols = [c for c in ["patient_id", "reader", "vessel_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    out_unified = FLOW_ROOT / "medis_overview.csv"
    df.to_csv(out_unified, index=False, sep=";")

    print(f"\nSaved unified overview: {out_unified}")
    print(f"Rows: {len(df)} | Unique patients: {df['patient_id'].nunique()}")
    if "reader" in df.columns and "vessel_name" in df.columns:
        print("\nVessels per reader (unified):")
        print(df.groupby(["reader", "vessel_name"]).size().unstack(fill_value=0))


def main3():
    """Summarise vessel segments per reader and patient overlap using unified CSV."""
    overview_path = FLOW_ROOT / "medis_overview.csv"
    if not overview_path.exists():
        print(f"Unified overview not found: {overview_path}")
        print("Run main() and main2() first to generate it.")
        return

    df = pd.read_csv(overview_path, sep=";")

    required_cols = {"patient_id", "reader", "vessel_name"}
    if not required_cols.issubset(df.columns):
        print(f"medis_overview.csv is missing required columns: {required_cols - set(df.columns)}")
        return

    print(f"Loaded unified overview from {overview_path}")
    print(f"Total rows: {len(df)}")

    # Overall unique patients & segments per reader
    print("\nUnique patients and segments per reader (overall):")
    patients_per_reader = df.groupby("reader")["patient_id"].nunique()
    segments_per_reader = df.groupby("reader").size()

    for reader in sorted(df["reader"].unique()):
        n_pat = int(patients_per_reader.get(reader, 0))
        n_seg = int(segments_per_reader.get(reader, 0))
        avg_seg = n_seg / n_pat if n_pat else 0.0
        print(f"  {reader}: {n_pat} unique patients, {n_seg} segments, {avg_seg:.2f} segments/patient")

    # Overall totals
    total_patients = int(df["patient_id"].nunique())
    total_segments = int(len(df))
    total_avg = total_segments / total_patients if total_patients else 0.0
    print("\nOverall:")
    print(f"  {total_patients} unique patients, {total_segments} segments, {total_avg:.2f} segments/patient")

    # Patient ID sets per reader and overlap; write to clean text files
    reader_patient_sets = {
        reader: set(sub_df["patient_id"])
        for reader, sub_df in df.groupby("reader")
    }

    eser_set = reader_patient_sets.get("Eser", set())
    robin_set = reader_patient_sets.get("Robin", set())
    both_set = eser_set & robin_set if eser_set and robin_set else set()

    patients_eser_path = FLOW_ROOT / "patients_eser.txt"
    patients_robin_path = FLOW_ROOT / "patients_robin.txt"
    patients_both_path = FLOW_ROOT / "patients_both.txt"

    if eser_set:
        patients_eser_path.write_text("\n".join(sorted(eser_set)), encoding="utf-8")
        print(f"\nSaved patient IDs for Eser to {patients_eser_path} ({len(eser_set)} patients)")
    if robin_set:
        patients_robin_path.write_text("\n".join(sorted(robin_set)), encoding="utf-8")
        print(f"Saved patient IDs for Robin to {patients_robin_path} ({len(robin_set)} patients)")
    if both_set:
        patients_both_path.write_text("\n".join(sorted(both_set)), encoding="utf-8")
        print(f"Saved patient IDs present for both readers to {patients_both_path} ({len(both_set)} patients)")
    if not both_set:
        print("\nNo patients with data from more than one reader were found.")

    # Per‑vessel summary
    print("\nPer-vessel unique patients and overlap:")
    all_readers = sorted(df["reader"].unique())
    vessel_rows = []
    vessel_overlap_ids: dict[str, set[str]] = {}

    for vessel, g_vessel in df.groupby("vessel_name"):
        row: dict[str, object] = {"vessel_name": vessel}
        reader_sets: dict[str, set[str]] = {}

        for reader in all_readers:
            pats = set(g_vessel.loc[g_vessel["reader"] == reader, "patient_id"])
            reader_sets[reader] = pats
            row[f"patients_{reader}"] = len(pats)

        non_empty_sets = [s for s in reader_sets.values() if s]
        if len(non_empty_sets) >= 2:
            overlap_set = set.intersection(*non_empty_sets)
        else:
            overlap_set = set()

        row["overlap_patients"] = len(overlap_set)
        vessel_rows.append(row)
        vessel_overlap_ids[vessel] = overlap_set

    if vessel_rows:
        vessel_df = pd.DataFrame(vessel_rows)
        vessel_df = vessel_df.sort_values("vessel_name")
        print("\nPer-vessel summary table (unique patients and overlap counts):")
        print(vessel_df.to_string(index=False))


if __name__ == "__main__":
    #main()
    #main2()
    main3()
    pass
