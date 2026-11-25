#!/bin/bash
work_dir="result_filtered"

if [ ! -d "$work_dir" ]; then
    echo "Error: Working directory '$work_dir' not found!"
    exit 1
fi

subdirs=("$work_dir"/*/)

echo "Starting processing in $work_dir..."
echo "========================================"

for dir in "${subdirs[@]}"; do
    dir=${dir%/}
    
    echo "Processing directory: $dir"

    if [ ! -d "$dir/extracted_figures" ]; then
        echo "Warning: $dir/extracted_figures not found, skipping..."
        echo "----------------------------------------"
        continue
    fi

    mkdir -p "$dir/separated_subfigures"

    python sep.py \
        --img "$dir/extracted_figures" \
        --output_dir "$dir/separated_subfigures"
    
    echo "Finished processing $dir"
    echo "----------------------------------------"
done

echo "========================================"
echo "All directories processed!"