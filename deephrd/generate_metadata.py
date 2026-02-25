#!/usr/bin/env python3
"""
generate_metadata.py

Automatically generates a metadata.tsv file for DeepHRD prediction
by scanning a directory for slide images.

Usage:
    python3 generate_metadata.py --slideDir /path/to/slides/ --output metadata.tsv
"""

import os
import argparse
import pandas as pd
from pathlib import Path


def extract_patient_id(filename):
    """
    Attempts to extract a patient ID from the filename.
    For TCGA format: TCGA-A1-A0SE-01A-01-BS1.xxx.svs -> TCGA-A1-A0SE
    Otherwise uses filename without extension as patient ID.
    """
    name = Path(filename).stem  # Remove extension
    
    # Try TCGA format: TCGA-XX-XXXX-XX-XX-XXX
    parts = name.split('-')
    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '-'.join(parts[:3])
    
    # Try other common formats: extract first 3 hyphen-separated parts
    if len(parts) >= 3:
        return '-'.join(parts[:3])
    
    # Fallback: use filename without extension, limit length
    return name[:50]  # Limit to reasonable length


def find_slide_files(directory, extensions=None):
    """
    Find all slide image files in the directory.
    
    Args:
        directory: Path to directory containing slides
        extensions: List of file extensions to look for (default: ['.svs', '.ndpi'])
    
    Returns:
        List of slide filenames (just the filename, not full path)
    """
    if extensions is None:
        extensions = ['.svs', '.ndpi']
    
    slide_files = []
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for ext in extensions:
        # Find files with this extension (case-insensitive)
        for file_path in directory.glob(f"*{ext}"):
            slide_files.append(file_path.name)
        for file_path in directory.glob(f"*{ext.upper()}"):
            if file_path.name not in slide_files:
                slide_files.append(file_path.name)
    
    return sorted(slide_files)


def generate_metadata(slide_dir, output_file, label_value=0.0, softLabel_value=0.0):
    """
    Generate a metadata.tsv file for DeepHRD prediction.
    
    Args:
        slide_dir: Directory containing slide images
        output_file: Path where metadata.tsv will be saved
        label_value: Placeholder label value (default: 0.0)
        softLabel_value: Placeholder softLabel value (default: 0.0, not used for prediction-only)
    """
    print(f"Scanning directory: {slide_dir}")
    slide_files = find_slide_files(slide_dir)
    
    if not slide_files:
        print(f"Warning: No slide files found in {slide_dir}")
        print("Looking for files with extensions: .svs, .ndpi")
        return
    
    print(f"Found {len(slide_files)} slide file(s)")
    
    # Create metadata DataFrame
    metadata_rows = []
    for slide_file in slide_files:
        metadata_rows.append({
            'slide': slide_file,  # Full filename - will be index
            'label': label_value,
            'softLabel': softLabel_value,
            'partition': 'test'  # All set to 'test' for prediction
        })
    
    # Create DataFrame with slide filename as index
    df = pd.DataFrame(metadata_rows)
    df.set_index('slide', inplace=True)
    
    # Ensure columns are in the correct order: label, softLabel, partition
    df = df[['label', 'softLabel', 'partition']]
    
    # Save as tab-separated file with index=True so 'slide' appears as first column
    df.to_csv(output_file, sep='\t', index=True)
    
    print(f"\nMetadata file created: {output_file}")
    print(f"Total samples: {len(df)}")
    print(f"\nFirst few entries:")
    print(df.head())
    print(f"\nNote: All samples are set to partition='test' for prediction.")
    print(f"Label and softLabel values are placeholders and not used during prediction.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate metadata.tsv file for DeepHRD prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate metadata for all slides in a directory
  python3 generate_metadata.py --slideDir /path/to/slides/ --output metadata.tsv
  
  # Generate metadata with custom label and softLabel values
  python3 generate_metadata.py --slideDir /path/to/slides/ --output metadata.tsv --label 0.0 --softLabel 0.0
        """
    )
    
    required_group = parser.add_argument_group('required arguments')
    optional_group = parser.add_argument_group('optional arguments')

    required_group.add_argument(
        '--slideDir',
        type=str,
        required=True,
        help='Directory containing slide image files (.svs, .ndpi)'
    )

    required_group.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for metadata.tsv file'
    )

    optional_group.add_argument(
        '--label',
        type=float,
        default=0.0,
        help='Placeholder label value (default: 0.0)'
    )

    optional_group.add_argument(
        '--softLabel',
        type=float,
        default=0.0,
        help='Placeholder softLabel value (default: 0.0, not used for prediction-only)'
    )
    
    args = parser.parse_args()
    
    generate_metadata(args.slideDir, args.output, args.label, args.softLabel)


if __name__ == '__main__':
    main()
