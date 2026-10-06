#!/usr/bin/env python3
"""
Mask2Former Training Visualization Tool
========================================
Generate comprehensive plots from training logs.

Usage:
    # Single experiment
    python tools/plot_training.py --log output/vaihingen_systematic_weighted_40k/log.txt
    
    # Compare multiple experiments
    python tools/plot_training.py --log output/exp1/log.txt output/exp2/log.txt --names "Batch2" "Batch8"
    
    # Save to specific directory
    python tools/plot_training.py --log output/exp1/log.txt --save-dir output/plots/
    
    # Specific plots only
    python tools/plot_training.py --log output/exp1/log.txt --plots loss miou
"""

import re
import argparse
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def parse_log_file(log_path: str) -> Dict:
    """
    Parse a detectron2/mask2former log file and extract training metrics.
    
    Returns:
        Dict with keys:
        - iterations: list of iteration numbers
        - losses: dict of {loss_name: [values]}
        - lr: list of learning rates
        - eval_iterations: list of evaluation iterations
        - miou: list of mIoU values
        - per_class_iou: dict of {class_name: [values]}
        - config: dict of config values
    """
    data = {
        'iterations': [],
        'losses': defaultdict(list),
        'lr': [],
        'eval_iterations': [],
        'miou': [],
        'fwiou': [],
        'macc': [],
        'pacc': [],
        'per_class_iou': defaultdict(list),
        'boundary_iou': defaultdict(list),
        'config': {},
        'best_miou': 0,
        'best_iter': 0,
    }
    
    # Regex patterns
    # Make loss_boundary optional to support both reference (no boundary) and boundary loss models
    loss_pattern = re.compile(
        r'iter:\s*(\d+)\s+total_loss:\s*([\d.-]+).*?'
        r'loss_ce:\s*([\d.-]+).*?loss_mask:\s*([\d.-]+).*?loss_dice:\s*([\d.-]+)'
        r'(?:.*?loss_boundary:\s*([\d.-]+))?.*?lr:\s*([\d.e-]+)'
    )
    
    eval_pattern = re.compile(
        r"'mIoU':\s*([\d.]+).*?'fwIoU':\s*([\d.]+)"
    )
    
    per_class_pattern = re.compile(
        r"'IoU-(\w+)':\s*([\d.]+)"
    )
    
    boundary_iou_pattern = re.compile(
        r"'BoundaryIoU-(\w+)':\s*([\d.]+)"
    )
    
    config_batch_pattern = re.compile(r'IMS_PER_BATCH:\s*(\d+)')
    config_lr_pattern = re.compile(r'BASE_LR:\s*([\d.e-]+)')
    config_iter_pattern = re.compile(r'MAX_ITER:\s*(\d+)')
    config_eval_period_pattern = re.compile(r'EVAL_PERIOD:\s*(\d+)')
    
    eval_period = 2000  # Default
    
    with open(log_path, 'r') as f:
        content = f.read()
        
        # Parse config
        batch_match = config_batch_pattern.search(content)
        if batch_match:
            data['config']['batch_size'] = int(batch_match.group(1))
        
        lr_match = config_lr_pattern.search(content)
        if lr_match:
            data['config']['base_lr'] = float(lr_match.group(1))
            
        iter_match = config_iter_pattern.search(content)
        if iter_match:
            data['config']['max_iter'] = int(iter_match.group(1))
            
        eval_period_match = config_eval_period_pattern.search(content)
        if eval_period_match:
            eval_period = int(eval_period_match.group(1))
            data['config']['eval_period'] = eval_period
    
    # Parse line by line for losses and evaluations
    eval_count = 0  # Counter for sequential evaluations
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse training losses
            loss_match = loss_pattern.search(line)
            if loss_match:
                iteration = int(loss_match.group(1))
                total_loss = float(loss_match.group(2))
                loss_ce = float(loss_match.group(3))
                loss_mask = float(loss_match.group(4))
                loss_dice = float(loss_match.group(5))
                loss_boundary = float(loss_match.group(6)) if loss_match.group(6) else 0.0
                lr = float(loss_match.group(7))
                
                data['iterations'].append(iteration)
                data['losses']['total_loss'].append(total_loss)
                data['losses']['loss_ce'].append(loss_ce)
                data['losses']['loss_mask'].append(loss_mask)
                data['losses']['loss_dice'].append(loss_dice)
                data['losses']['loss_boundary'].append(loss_boundary)
                data['lr'].append(lr)
            
            # Parse evaluation results
            # Use sequential counting: 1st eval = eval_period, 2nd = 2*eval_period, etc.
            eval_match = eval_pattern.search(line)
            if eval_match:
                miou = float(eval_match.group(1))
                fwiou = float(eval_match.group(2))
                
                eval_count += 1
                eval_iter = eval_count * eval_period
                
                data['eval_iterations'].append(eval_iter)
                data['miou'].append(miou)
                data['fwiou'].append(fwiou)
                
                # Parse per-class IoU
                for match in per_class_pattern.finditer(line):
                    class_name = match.group(1)
                    iou_value = float(match.group(2))
                    data['per_class_iou'][class_name].append(iou_value)
                
                # Parse boundary IoU
                for match in boundary_iou_pattern.finditer(line):
                    class_name = match.group(1)
                    biou_value = float(match.group(2))
                    data['boundary_iou'][class_name].append(biou_value)
    
    # Find best mIoU from parsed data
    if data['miou']:
        data['best_miou'] = max(data['miou'])
        best_idx = data['miou'].index(data['best_miou'])
        data['best_iter'] = data['eval_iterations'][best_idx]
    
    return data


def smooth_curve(values: List[float], window: int = 50) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_losses(data_list: List[Dict], names: List[str], save_dir: str, show: bool = True):
    """Plot training losses over iterations - combined grid view."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Losses Over Iterations', fontsize=16, fontweight='bold')
    
    loss_types = ['total_loss', 'loss_ce', 'loss_mask', 'loss_dice', 'loss_boundary']
    titles = ['Total Loss', 'Cross-Entropy Loss', 'Mask Loss', 'Dice Loss', 'Boundary Loss']
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for idx, (loss_type, title) in enumerate(zip(loss_types, titles)):
        ax = axes[idx // 3, idx % 3]
        
        for data, name, color in zip(data_list, names, colors):
            if loss_type in data['losses'] and data['losses'][loss_type]:
                iters = data['iterations'][:len(data['losses'][loss_type])]
                values = data['losses'][loss_type]
                
                # Plot raw values with low alpha
                ax.plot(iters, values, alpha=0.2, color=color)
                
                # Plot smoothed values
                if len(values) > 50:
                    smoothed = smooth_curve(values, window=50)
                    smooth_iters = iters[49:][:len(smoothed)]
                    ax.plot(smooth_iters, smoothed, label=name, color=color, linewidth=2)
                else:
                    ax.plot(iters, values, label=name, color=color, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    # Hide the 6th subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'losses.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_individual_losses(data_list: List[Dict], names: List[str], save_dir: str, show: bool = True):
    """Plot each loss type as a separate high-resolution image."""
    loss_types = ['total_loss', 'loss_ce', 'loss_mask', 'loss_dice', 'loss_boundary']
    titles = ['Total Loss', 'Cross-Entropy Loss', 'Mask Loss', 'Dice Loss', 'Boundary Loss']
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for loss_type, title in zip(loss_types, titles):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        has_data = False
        for data, name, color in zip(data_list, names, colors):
            if loss_type in data['losses'] and data['losses'][loss_type]:
                has_data = True
                iters = data['iterations'][:len(data['losses'][loss_type])]
                values = data['losses'][loss_type]
                
                # Plot raw values with low alpha
                ax.plot(iters, values, alpha=0.15, color=color)
                
                # Plot smoothed values
                if len(values) > 50:
                    smoothed = smooth_curve(values, window=50)
                    smooth_iters = iters[49:][:len(smoothed)]
                    ax.plot(smooth_iters, smoothed, label=name, color=color, linewidth=2.5)
                else:
                    ax.plot(iters, values, label=name, color=color, linewidth=2.5)
        
        if not has_data:
            plt.close()
            continue
            
        ax.set_xlabel('Iteration', fontsize=13)
        ax.set_ylabel('Loss', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        filename = f'loss_{loss_type.replace("loss_", "")}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


def plot_miou(data_list: List[Dict], names: List[str], save_dir: str, show: bool = True):
    """Plot mIoU over training iterations."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for data, name, color in zip(data_list, names, colors):
        if data['eval_iterations'] and data['miou']:
            iters = data['eval_iterations']
            miou = data['miou']
            
            ax.plot(iters, miou, marker='o', markersize=4, label=name, color=color, linewidth=2)
            
            # Mark best point
            if data['best_miou'] > 0:
                best_idx = miou.index(max(miou))
                ax.scatter([iters[best_idx]], [max(miou)], s=100, color=color, 
                          marker='*', zorder=5, edgecolors='black', linewidths=1)
                ax.annotate(f'{max(miou):.2f}%', (iters[best_idx], max(miou)),
                           textcoords="offset points", xytext=(5, 10), fontsize=9,
                           color=color, fontweight='bold')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('mIoU (%)', fontsize=12)
    ax.set_title('Validation mIoU Over Training', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'miou.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_loss_miou_combined(data_list: List[Dict], names: List[str], save_dir: str, show: bool = True):
    """Plot losses and mIoU on the same figure with dual y-axes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    # Left plot: Total Loss
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    for data, name, color in zip(data_list, names, colors):
        # Plot loss on primary axis
        if data['iterations'] and data['losses']['total_loss']:
            iters = data['iterations']
            values = data['losses']['total_loss']
            
            if len(values) > 50:
                smoothed = smooth_curve(values, window=50)
                smooth_iters = iters[49:][:len(smoothed)]
                line1, = ax1.plot(smooth_iters, smoothed, color=color, linewidth=2, 
                                  linestyle='-', label=f'{name} (Loss)')
            else:
                line1, = ax1.plot(iters, values, color=color, linewidth=2,
                                  linestyle='-', label=f'{name} (Loss)')
        
        # Plot mIoU on secondary axis
        if data['eval_iterations'] and data['miou']:
            line2, = ax1_twin.plot(data['eval_iterations'], data['miou'], 
                                   color=color, linewidth=2, linestyle='--',
                                   marker='o', markersize=4, label=f'{name} (mIoU)')
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1_twin.set_ylabel('mIoU (%)', fontsize=12)
    ax1.set_title('Training Loss & Validation mIoU', fontsize=14, fontweight='bold')
    
    # Combine legends - place outside plot area
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
               bbox_to_anchor=(1.0, 1.0), fontsize=9, framealpha=0.9)
    
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    # Right plot: Boundary Loss specifically
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    for data, name, color in zip(data_list, names, colors):
        if data['iterations'] and data['losses']['loss_boundary']:
            iters = data['iterations']
            values = data['losses']['loss_boundary']
            
            if len(values) > 50:
                smoothed = smooth_curve(values, window=50)
                smooth_iters = iters[49:][:len(smoothed)]
                ax2.plot(smooth_iters, smoothed, color=color, linewidth=2, 
                        linestyle='-', label=f'{name} (Boundary)')
            else:
                ax2.plot(iters, values, color=color, linewidth=2,
                        linestyle='-', label=f'{name} (Boundary)')
        
        if data['eval_iterations'] and data['miou']:
            ax2_twin.plot(data['eval_iterations'], data['miou'], 
                         color=color, linewidth=2, linestyle='--',
                         marker='o', markersize=4, label=f'{name} (mIoU)')
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Boundary Loss', fontsize=12)
    ax2_twin.set_ylabel('mIoU (%)', fontsize=12)
    ax2.set_title('Boundary Loss & Validation mIoU', fontsize=14, fontweight='bold')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               bbox_to_anchor=(1.0, 1.0), fontsize=9, framealpha=0.9)
    
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_miou_combined.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_per_class_iou(data_list: List[Dict], names: List[str], save_dir: str, show: bool = True):
    """Plot per-class IoU progression."""
    if not data_list or not data_list[0]['per_class_iou']:
        print("No per-class IoU data found.")
        return
    
    classes = list(data_list[0]['per_class_iou'].keys())
    n_classes = len(classes)
    
    # Determine grid size
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Per-Class IoU Progression', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for idx, class_name in enumerate(classes):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        for data, name, color in zip(data_list, names, colors):
            if class_name in data['per_class_iou']:
                iters = data['eval_iterations'][:len(data['per_class_iou'][class_name])]
                values = data['per_class_iou'][class_name]
                ax.plot(iters, values, marker='o', markersize=3, 
                       label=name, color=color, linewidth=1.5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('IoU (%)')
        ax.set_title(class_name.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    # Hide unused subplots
    for idx in range(n_classes, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_iou.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_learning_rate(data_list: List[Dict], names: List[str], save_dir: str, show: bool = True):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for data, name, color in zip(data_list, names, colors):
        if data['iterations'] and data['lr']:
            ax.plot(data['iterations'], data['lr'], label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_final_comparison_bar(data_list: List[Dict], names: List[str], save_dir: str, show: bool = True):
    """Plot final per-class IoU comparison as bar chart."""
    if not data_list or not data_list[0]['per_class_iou']:
        print("No per-class IoU data found.")
        return
    
    classes = list(data_list[0]['per_class_iou'].keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(classes))
    width = 0.8 / len(data_list)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for i, (data, name, color) in enumerate(zip(data_list, names, colors)):
        final_values = []
        for class_name in classes:
            if class_name in data['per_class_iou'] and data['per_class_iou'][class_name]:
                # Get value at best mIoU iteration
                best_idx = data['miou'].index(max(data['miou'])) if data['miou'] else -1
                if best_idx >= 0 and best_idx < len(data['per_class_iou'][class_name]):
                    final_values.append(data['per_class_iou'][class_name][best_idx])
                else:
                    final_values.append(data['per_class_iou'][class_name][-1])
            else:
                final_values.append(0)
        
        bars = ax.bar(x + i * width - 0.4 + width/2, final_values, width, 
                     label=f'{name} (mIoU: {max(data["miou"]):.2f}%)', color=color)
        
        # Add value labels on bars
        for bar, val in zip(bars, final_values):
            ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('IoU (%)', fontsize=12)
    ax.set_title('Final Per-Class IoU Comparison (at Best mIoU)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in classes], fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_comparison.png'), dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def print_summary(data_list: List[Dict], names: List[str]):
    """Print summary statistics for all experiments."""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for data, name in zip(data_list, names):
        print(f"\n{name}")
        print("-"*40)
        
        if data['config']:
            print(f"  Batch Size: {data['config'].get('batch_size', 'N/A')}")
            print(f"  Base LR: {data['config'].get('base_lr', 'N/A')}")
            print(f"  Max Iterations: {data['config'].get('max_iter', 'N/A')}")
        
        if data['miou']:
            print(f"  Best mIoU: {max(data['miou']):.2f}%")
            best_idx = data['miou'].index(max(data['miou']))
            print(f"  Best Iteration: {data['eval_iterations'][best_idx]}")
            print(f"  Final mIoU: {data['miou'][-1]:.2f}%")
        
        if data['per_class_iou']:
            print(f"  Per-Class IoU at Best:")
            best_idx = data['miou'].index(max(data['miou'])) if data['miou'] else -1
            for class_name, values in data['per_class_iou'].items():
                if best_idx >= 0 and best_idx < len(values):
                    print(f"    {class_name}: {values[best_idx]:.2f}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot Mask2Former training logs')
    parser.add_argument('--log', nargs='+', required=True, help='Path(s) to log file(s)')
    parser.add_argument('--names', nargs='+', default=None, help='Names for each experiment')
    parser.add_argument('--save-dir', default='./output/plots', help='Directory to save plots')
    parser.add_argument('--plots', nargs='+', default=['all'], 
                       choices=['all', 'loss', 'loss_individual', 'miou', 'combined', 'per_class', 'lr', 'bar'],
                       help='Which plots to generate')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set default names if not provided
    if args.names is None:
        args.names = [Path(log).parent.name for log in args.log]
    
    # Parse all log files
    print("Parsing log files...")
    data_list = []
    for log_path in args.log:
        print(f"  - {log_path}")
        data = parse_log_file(log_path)
        data_list.append(data)
    
    show = not args.no_show
    
    # Generate plots
    if 'all' in args.plots or 'loss' in args.plots:
        print("Generating loss plots (grid)...")
        plot_losses(data_list, args.names, args.save_dir, show)
    
    if 'all' in args.plots or 'loss_individual' in args.plots:
        print("Generating individual loss plots...")
        plot_individual_losses(data_list, args.names, args.save_dir, show)
    
    if 'all' in args.plots or 'miou' in args.plots:
        print("Generating mIoU plot...")
        plot_miou(data_list, args.names, args.save_dir, show)
    
    if 'all' in args.plots or 'combined' in args.plots:
        print("Generating combined loss/mIoU plot...")
        plot_loss_miou_combined(data_list, args.names, args.save_dir, show)
    
    if 'all' in args.plots or 'per_class' in args.plots:
        print("Generating per-class IoU plots...")
        plot_per_class_iou(data_list, args.names, args.save_dir, show)
    
    if 'all' in args.plots or 'lr' in args.plots:
        print("Generating learning rate plot...")
        plot_learning_rate(data_list, args.names, args.save_dir, show)
    
    if 'all' in args.plots or 'bar' in args.plots:
        print("Generating final comparison bar chart...")
        plot_final_comparison_bar(data_list, args.names, args.save_dir, show)
    
    # Print summary
    print_summary(data_list, args.names)
    
    print(f"\nPlots saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
