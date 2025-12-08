#!/usr/bin/env python3
"""
Script to convert SEER cancer data CSV into LLM query format for survival prediction.
Excludes survival-related columns since we want the model to predict survival.
"""

import csv
import json
import argparse
from pathlib import Path


# Columns to exclude (survival-related and target variables)
EXCLUDE_COLUMNS = {
    'survival_months_raw',
    'cause_death',
    'survival_months',
    'censored_status',
    'survival_years',
    'is_alive',
    'died_from_cancer',
}

# Query template for survival prediction
QUERY_TEMPLATE = """Given the following patient information, predict if the patient will die from cancer or survive:

Patient ID: {patient_id}
Primary Site: {primary_site_labeled}
Age Group: {age_group}
Age (numeric): {age_numeric}
Age Category: {age_category}
Race: {race}
Sex: {sex}
Year of Diagnosis: {year_diagnosis}
Treatment Era: {treatment_era}
Site Recode: {site_recode}
Cancer System: {cancer_system}
Histology Code: {histology_code}
Behavior: {behavior}

Based on this information, what is the predicted survival outcome for this patient? Return a 1 for survival and 0 for death."""


def load_csv(filepath: str) -> list[dict]:
    """Load CSV file and return list of row dictionaries."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def filter_columns(row: dict) -> dict:
    """Remove survival-related columns from a row."""
    return {k: v for k, v in row.items() if k not in EXCLUDE_COLUMNS}


def row_to_query(row: dict) -> str:
    """Convert a data row to an LLM query using the template."""
    filtered = filter_columns(row)
    return QUERY_TEMPLATE.format(**filtered)


def convert_to_queries(data: list[dict]) -> list[dict]:
    """Convert all rows to query format with metadata."""
    queries = []
    for row in data:
        query = row_to_query(row)
        # Store original survival data as ground truth for evaluation
        ground_truth = {k: row[k] for k in EXCLUDE_COLUMNS if k in row}
        queries.append({
            'patient_id': row.get('patient_id', ''),
            'query': query,
            'ground_truth': ground_truth,
            'features': filter_columns(row)
        })
    return queries


def save_queries(queries: list[dict], output_path: str, format: str = 'json'):
    """Save queries to file in specified format."""
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(queries, f, indent=2)
    elif format == 'jsonl':
        with open(output_path, 'w', encoding='utf-8') as f:
            for query in queries:
                f.write(json.dumps(query) + '\n')
    elif format == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, query in enumerate(queries):
                f.write(f"=== Query {i+1} (Patient: {query['patient_id']}) ===\n")
                f.write(query['query'])
                f.write('\n\n')
    else:
        raise ValueError(f"Unsupported format: {format}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SEER cancer data CSV to LLM queries for survival prediction'
    )
    parser.add_argument(
        'input_csv',
        nargs='?',
        default='seer_data_test.csv',
        help='Input CSV file path (default: seer_data_test.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        default='queries.json',
        help='Output file path (default: queries.json)'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'jsonl', 'txt'],
        default='json',
        help='Output format: json, jsonl, or txt (default: json)'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=None,
        help='Number of samples to convert (default: all)'
    )
    parser.add_argument(
        '--print-sample',
        action='store_true',
        help='Print a sample query to stdout'
    )
    
    args = parser.parse_args()
    
    # Resolve input path
    input_path = Path(args.input_csv)
    if not input_path.is_absolute():
        input_path = Path(__file__).parent / input_path
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Load data
    print(f"Loading data from: {input_path}")
    data = load_csv(str(input_path))
    print(f"Loaded {len(data)} rows")
    
    # Limit samples if requested
    if args.num_samples is not None:
        data = data[:args.num_samples]
        print(f"Using first {len(data)} samples")
    
    # Convert to queries
    queries = convert_to_queries(data)
    
    # Print sample if requested
    if args.print_sample and queries:
        print("\n=== Sample Query ===")
        print(queries[0]['query'])
        print("=" * 40)
    
    # Resolve output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    
    # Save queries
    save_queries(queries, str(output_path), args.format)
    print(f"\nSaved {len(queries)} queries to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
