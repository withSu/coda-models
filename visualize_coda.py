import argparse
import numpy as np
import torch
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as mpatches

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def visualize_bev(points, pred_boxes, gt_boxes, save_path=None, sample_idx=0):
    """
    Visualize Bird's Eye View with predictions and GT
    Args:
        points: (N, 3+) point cloud
        pred_boxes: (M, 7) predicted boxes [x, y, z, dx, dy, dz, heading]
        gt_boxes: (K, 7) ground truth boxes
        save_path: path to save the visualization
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot point cloud
    if points is not None:
        # Filter points for BEV (remove height)
        pc_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
        mask = (points[:, 0] > pc_range[0]) & (points[:, 0] < pc_range[3]) & \
               (points[:, 1] > pc_range[1]) & (points[:, 1] < pc_range[4])
        points_filtered = points[mask]
        
        # Plot points
        ax.scatter(points_filtered[:, 0], points_filtered[:, 1], 
                  c='gray', s=0.1, alpha=0.5)
    
    # Color map for classes
    color_map = {
        'Car': 'red',
        'Pedestrian': 'blue', 
        'Cyclist': 'green',
        'Other': 'yellow'
    }
    
    # Plot GT boxes
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            if len(box) < 7:
                continue
            x, y, z, dx, dy, dz, heading = box[:7]
            
            # Create rectangle corners
            corners = np.array([
                [-dx/2, -dy/2],
                [dx/2, -dy/2],
                [dx/2, dy/2],
                [-dx/2, dy/2]
            ])
            
            # Rotate corners
            rot_matrix = np.array([
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)]
            ])
            corners = corners @ rot_matrix.T
            corners[:, 0] += x
            corners[:, 1] += y
            
            # Draw box
            polygon = Polygon(corners, fill=False, edgecolor='green', 
                            linewidth=2, linestyle='--', label='GT')
            ax.add_patch(polygon)
            
            # Draw direction
            ax.arrow(x, y, 3*np.cos(heading), 3*np.sin(heading),
                    head_width=0.5, head_length=0.5, fc='green', ec='green')
    
    # Plot predicted boxes
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            if len(box) < 7:
                continue
            x, y, z, dx, dy, dz, heading = box[:7]
            
            # Get confidence score if available
            score = box[7] if len(box) > 7 else 1.0
            
            # Create rectangle corners
            corners = np.array([
                [-dx/2, -dy/2],
                [dx/2, -dy/2],
                [dx/2, dy/2],
                [-dx/2, dy/2]
            ])
            
            # Rotate corners
            rot_matrix = np.array([
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)]
            ])
            corners = corners @ rot_matrix.T
            corners[:, 0] += x
            corners[:, 1] += y
            
            # Draw box with confidence-based alpha
            polygon = Polygon(corners, fill=False, edgecolor='red',
                            linewidth=2, alpha=max(0.3, score), label='Pred')
            ax.add_patch(polygon)
            
            # Draw direction
            ax.arrow(x, y, 3*np.cos(heading), 3*np.sin(heading),
                    head_width=0.5, head_length=0.5, fc='red', ec='red', alpha=max(0.3, score))
            
            # Add score text
            if score < 1.0:
                ax.text(x, y, f'{score:.2f}', fontsize=8, color='red')
    
    # Set axis properties
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'BEV Detection Results - Sample {sample_idx}')
    
    # Add legend
    handles = [
        mpatches.Patch(color='green', label='Ground Truth'),
        mpatches.Patch(color='red', label='Predictions')
    ]
    ax.legend(handles=handles)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize CODa detection results')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/da_models/second_coda32_oracle_3class.yaml',
                        help='Config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='visualization_results', 
                        help='Directory to save visualizations')
    parser.add_argument('--score_thresh', type=float, default=0.3, 
                        help='Score threshold for predictions')
    args = parser.parse_args()
    
    # Setup config
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build dataloader
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=4,
        logger=None,
        training=False,
        total_epochs=1,
        seed=666
    )
    
    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=None, to_cpu=False)
    model.cuda()
    model.eval()
    
    print(f"Visualizing {args.num_samples} samples...")
    
    with torch.no_grad():
        for idx, batch_dict in enumerate(test_loader):
            if idx >= args.num_samples:
                break
                
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)
            
            # Get point cloud
            points = batch_dict['points'][:, 1:4].cpu().numpy()  # Remove batch index
            
            # Get predictions
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            
            # Filter by score
            mask = pred_scores > args.score_thresh
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
            
            # Add scores to boxes for visualization
            if len(pred_boxes) > 0:
                pred_boxes = np.concatenate([pred_boxes, pred_scores[:, None]], axis=1)
            
            # Get GT boxes if available
            gt_boxes = None
            if 'gt_boxes' in batch_dict:
                gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
                # Remove invalid boxes
                mask = np.any(gt_boxes != 0, axis=1)
                gt_boxes = gt_boxes[mask]
            
            # Save visualization
            save_path = save_dir / f'sample_{idx:04d}.png'
            visualize_bev(points, pred_boxes, gt_boxes, save_path, idx)
            
            print(f"Sample {idx}: {len(pred_boxes)} predictions, {len(gt_boxes) if gt_boxes is not None else 0} GT boxes")
    
    print(f"\nVisualization complete! Results saved to {save_dir}")

if __name__ == '__main__':
    main()