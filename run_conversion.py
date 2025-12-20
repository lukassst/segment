#!/usr/bin/env python3
"""Run MEDIS to JSON and GIfTI conversions for all example files."""

import os
import sys

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from medis_to_json import convert_medis_to_json
from medis_to_gii import convert_medis_to_gii

# Base paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CTA_FILE = os.path.join(DATA_DIR, '01-BER-0088.nii.gz')

# MEDIS export files
MEDIS_FILES = [
    '01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_lad.txt',
    '01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_lcx.txt',
    '01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_rca.txt',
    '01-BER-0088_20181010_Session_Session 16_04_2021 14_46 _ecrf_d1.txt',
]

def main():
    print("=" * 60)
    print("MEDIS Export Conversion")
    print("=" * 60)
    
    all_json_files = []
    all_gii_files = []
    
    for medis_file in MEDIS_FILES:
        medis_path = os.path.join(DATA_DIR, medis_file)
        
        if not os.path.exists(medis_path):
            print(f"\nSkipping (not found): {medis_file}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Processing: {medis_file}")
        print("=" * 60)
        
        # Convert to JSON
        print("\n--- JSON Point Cloud ---")
        try:
            json_files = convert_medis_to_json(medis_path, CTA_FILE, DATA_DIR)
            all_json_files.extend(json_files)
        except Exception as e:
            print(f"  Error: {e}")
        
        # Convert to GIfTI
        print("\n--- GIfTI Mesh ---")
        try:
            gii_files = convert_medis_to_gii(medis_path, CTA_FILE, DATA_DIR)
            all_gii_files.extend(gii_files)
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"JSON files created: {len(all_json_files)}")
    for f in all_json_files:
        print(f"  - {os.path.basename(f)}")
    
    print(f"\nGIfTI files created: {len(all_gii_files)}")
    for f in all_gii_files:
        print(f"  - {os.path.basename(f)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
