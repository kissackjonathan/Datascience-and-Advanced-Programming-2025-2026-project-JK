"""
Script to convert CSV files from UTF-16 to UTF-8 encoding
Handles the encoding issues with FSI data files
"""
import pandas as pd
from pathlib import Path


def convert_csv_encoding(input_path, output_path, source_encoding="utf-16"):
    """
    Convert CSV file from source encoding to UTF-8

    Parameters:
    -----------
    input_path : str or Path
        Path to input CSV file
    output_path : str or Path
        Path to save converted CSV file
    source_encoding : str, default='utf-16'
        Source file encoding
    """
    try:
        # Read with source encoding
        df = pd.read_csv(input_path, encoding=source_encoding, sep="\t")
        print(f"✓ Read {input_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")

        # Save as UTF-8
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✓ Saved to {output_path.name} (UTF-8)")

        return df

    except Exception as e:
        print(f"✗ Error converting {input_path.name}: {e}")
        return None


def main():
    # Paths
    downloads_dir = Path("/Users/kissack.jonathan/Downloads")
    raw_data_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CSV ENCODING CONVERSION - UTF-16 to UTF-8")
    print("=" * 60)
    print()

    # Find all Rankings CSV files
    csv_files = list(downloads_dir.glob("Rankings*.csv"))

    if not csv_files:
        print("No Rankings CSV files found in Downloads")
        return

    print(f"Found {len(csv_files)} file(s) to convert:\n")

    # Convert each file
    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. Converting {csv_file.name}...")

        # Create output filename
        if csv_file.name == "Rankings.csv":
            output_name = "fsi_rankings.csv"
        else:
            # Rankings(1).csv -> fsi_rankings_2.csv
            num = csv_file.stem.split("(")[1].rstrip(")")
            output_name = f"fsi_rankings_{int(num)+1}.csv"

        output_path = raw_data_dir / output_name

        # Convert
        df = convert_csv_encoding(csv_file, output_path)

        if df is not None:
            print(f"   Preview: {list(df.columns[:5])}")
            print()

    print("=" * 60)
    print("CONVERSION COMPLETE")
    print(f"Files saved to: {raw_data_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
