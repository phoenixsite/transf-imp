#!/usr/bin/env bash

# Use: ./merge_summaries.sh /path/to/directory output.csv

INPUT_DIR="$1"
OUTPUT_FILE="$2"

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_FILE" ]]; then
    echo "Use: $0 <directory> <output_file>"
    exit 1
fi

> "$OUTPUT_FILE"

HEADER_WRITTEN=false

for d in "$INPUT_DIR"/*/ ; do
    SUMMARY_FILE="${d}summary.csv"

    if [[ -f "$SUMMARY_FILE" ]]; then
        echo "Processing: $SUMMARY_FILE"

        if [[ "$HEADER_WRITTEN" = false ]]; then
            head -n 1 "$SUMMARY_FILE" >> "$OUTPUT_FILE"
            HEADER_WRITTEN=true
        fi

        sed -n '2p' "$SUMMARY_FILE" >> "$OUTPUT_FILE"
    else
        echo "summary.csv in $d does not exist"
    fi
done