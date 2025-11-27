#!/usr/bin/env python3
"""
Memory-safe reservoir sampling for large CSV.
Usage:
  python sample_csv.py --input /mnt/data/Airbnb_Data.csv --output sampled_airbnb_50k.csv --n 50000
"""
import argparse, csv, random

def reservoir_sample(infile, outfile, n):
    with open(infile, newline='', encoding='utf-8', errors='replace') as inf:
        reader = csv.reader(inf)
        header = next(reader)
        reservoir = []
        for i, row in enumerate(reader, start=1):
            if i <= n:
                reservoir.append(row)
            else:
                j = random.randint(1, i)
                if j <= n:
                    reservoir[j-1] = row
    with open(outfile, "w", newline='', encoding='utf-8') as outf:
        writer = csv.writer(outf)
        writer.writerow(header)
        writer.writerows(reservoir)
    print(f"Sampled {len(reservoir)} rows to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="./sampled_airbnb.csv")
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()
    reservoir_sample(args.input, args.output, args.n)
