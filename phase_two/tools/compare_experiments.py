#!/usr/bin/env python3
"""
Experiment Comparison Tool
==========================
Generate comparison tables from training logs.

Usage:
    # Compare 2 experiments (first is reference)
    python tools/compare_experiments.py --log output/ref/log.txt output/exp1/log.txt \
        --names Reference Experiment1 --output comparison.csv
    
    # Compare 3+ experiments
    python tools/compare_experiments.py --log output/ref/log.txt output/exp1/log.txt output/exp2/log.txt \
        --names Reference Uniform Systematic --output comparison.csv
    
    # Use last model instead of best
    python tools/compare_experiments.py --log output/ref/log.txt output/exp1/log.txt \
        --names Reference Experiment1 --use-last --output comparison.csv
"""

import re
import argparse
import csv
from pathlib import Path
from collections import OrderedDict
from typing import List, Dict, Optional


def parse_log_for_comparison(log_path: str, use_last: bool = False) -> Dict:
    """
    Parse a log file and extract IoU values at best (or last) checkpoint.
    
    Returns:
        Dict with keys:
        - miou: best/last mIoU value
        - iteration: iteration of best/last evaluation
        - per_class_iou: dict of {class_name: iou_value}
    """
    data = {
        'miou': 0.0,
        'iteration': 0,
        'per_class_iou': OrderedDict(),
        'config': {}
    }
    
    # Regex patterns
    eval_pattern = re.compile(
        r"'mIoU':\s*([\d.]+).*?'fwIoU':\s*([\d.]+)"
    )
    
    per_class_pattern = re.compile(
        r"'IoU-(\w+)':\s*([\d.]+)"
    )
    
    config_eval_period_pattern = re.compile(r'EVAL_PERIOD:\s*(\d+)')
    
    eval_period = 2000  # Default
    
    # First pass: get config
    with open(log_path, 'r') as f:
        content = f.read()
        eval_period_match = config_eval_period_pattern.search(content)
        if eval_period_match:
            eval_period = int(eval_period_match.group(1))
    
    # Second pass: collect all evaluations
    evaluations = []
    eval_count = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            eval_match = eval_pattern.search(line)
            if eval_match:
                eval_count += 1
                miou = float(eval_match.group(1))
                iteration = eval_count * eval_period
                
                per_class = OrderedDict()
                for match in per_class_pattern.finditer(line):
                    class_name = match.group(1)
                    iou_value = float(match.group(2))
                    per_class[class_name] = iou_value
                
                evaluations.append({
                    'miou': miou,
                    'iteration': iteration,
                    'per_class_iou': per_class
                })
    
    if not evaluations:
        print(f"Warning: No evaluation results found in {log_path}")
        return data
    
    # Select best or last
    if use_last:
        selected = evaluations[-1]
    else:
        selected = max(evaluations, key=lambda x: x['miou'])
    
    data['miou'] = selected['miou']
    data['iteration'] = selected['iteration']
    data['per_class_iou'] = selected['per_class_iou']
    
    return data


def generate_comparison_csv(
    data_list: List[Dict],
    names: List[str],
    output_path: str,
    include_deltas: bool = True
):
    """
    Generate a comparison CSV from parsed log data.
    
    Args:
        data_list: List of parsed data dicts
        names: List of experiment names
        output_path: Path to save CSV
        include_deltas: Whether to include delta columns (vs Reference, vs previous)
    """
    if not data_list:
        print("No data to compare")
        return
    
    # Get class names from first experiment
    classes = list(data_list[0]['per_class_iou'].keys())
    
    # Build header
    header = ['Class']
    for i, name in enumerate(names):
        header.append(f'{name}_IoU')
        if include_deltas and i > 0:
            # Delta vs Reference (first experiment)
            header.append(f'{name[:3]}_vs_Ref')
            # Delta vs previous experiment (if more than 2 experiments)
            if i > 1:
                prev_name = names[i-1]
                header.append(f'{name[:3]}_vs_{prev_name[:3]}')
    
    # Build rows
    rows = []
    
    # Per-class rows
    for class_name in classes:
        # Convert to Title_Case (e.g., impervious_surface -> Impervious_Surface)
        display_name = '_'.join(word.capitalize() for word in class_name.split('_'))
        row = [display_name]
        
        ref_iou = data_list[0]['per_class_iou'].get(class_name, 0.0)
        prev_iou = ref_iou
        
        for i, data in enumerate(data_list):
            iou = data['per_class_iou'].get(class_name, 0.0)
            row.append(round(iou, 2))
            
            if include_deltas and i > 0:
                # Delta vs Reference
                delta_ref = iou - ref_iou
                row.append(round(delta_ref, 2))
                
                # Delta vs previous
                if i > 1:
                    delta_prev = iou - prev_iou
                    row.append(round(delta_prev, 2))
            
            prev_iou = iou
        
        rows.append(row)
    
    # mIoU row
    miou_row = ['mIoU']
    ref_miou = data_list[0]['miou']
    prev_miou = ref_miou
    
    for i, data in enumerate(data_list):
        miou = data['miou']
        miou_row.append(round(miou, 2))
        
        if include_deltas and i > 0:
            delta_ref = miou - ref_miou
            miou_row.append(round(delta_ref, 2))
            
            if i > 1:
                delta_prev = miou - prev_miou
                miou_row.append(round(delta_prev, 2))
        
        prev_miou = miou
    
    rows.append(miou_row)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"Comparison saved to: {output_path}")
    
    # Also print to console
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    # Print as formatted table
    col_widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    
    # Header
    header_line = " | ".join(str(h).ljust(w) for h, w in zip(header, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Rows
    for row in rows:
        row_line = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(row_line)
    
    print("="*80)
    
    # Print summary
    print("\nSUMMARY:")
    for name, data in zip(names, data_list):
        print(f"  {name}: {data['miou']:.2f}% mIoU @ iteration {data['iteration']}")


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results from training logs')
    parser.add_argument('--log', nargs='+', required=True, 
                       help='Log file paths (first is reference)')
    parser.add_argument('--names', nargs='+', default=None,
                       help='Experiment names (default: folder names)')
    parser.add_argument('--output', '-o', default='comparison.csv',
                       help='Output CSV path')
    parser.add_argument('--use-last', action='store_true',
                       help='Use last evaluation instead of best')
    parser.add_argument('--no-deltas', action='store_true',
                       help='Do not include delta columns')
    
    args = parser.parse_args()
    
    # Set default names
    if args.names is None:
        args.names = [Path(log).parent.name for log in args.log]
    
    if len(args.names) != len(args.log):
        print(f"Error: Number of names ({len(args.names)}) must match number of logs ({len(args.log)})")
        return
    
    # Parse all logs
    print("Parsing log files...")
    data_list = []
    for log_path, name in zip(args.log, args.names):
        print(f"  - {name}: {log_path}")
        data = parse_log_for_comparison(log_path, use_last=args.use_last)
        data_list.append(data)
    
    # Generate comparison
    generate_comparison_csv(
        data_list,
        args.names,
        args.output,
        include_deltas=not args.no_deltas
    )


if __name__ == '__main__':
    main()
