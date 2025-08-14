import argparse
import numpy as np
import torch
from pathlib import Path
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def save_visualization_data(points, pred_boxes, gt_boxes, pred_labels, gt_labels, save_path):
    """Save visualization data as numpy arrays for external visualization"""
    
    data = {
        'points': points,
        'pred_boxes': pred_boxes,
        'gt_boxes': gt_boxes,
        'pred_labels': pred_labels,
        'gt_labels': gt_labels
    }
    
    np.savez(save_path, **data)
    print(f"Saved data to {save_path}")

def print_detection_summary(pred_boxes, pred_scores, pred_labels, gt_boxes, class_names):
    """Print detection summary"""
    print("\n" + "="*50)
    print("Detection Summary:")
    print("-"*50)
    
    if len(pred_boxes) > 0:
        print(f"Predictions: {len(pred_boxes)} boxes")
        for cls_id in np.unique(pred_labels):
            mask = pred_labels == cls_id
            cls_name = class_names[cls_id-1] if cls_id <= len(class_names) else f"Class_{cls_id}"
            count = mask.sum()
            avg_score = pred_scores[mask].mean()
            print(f"  {cls_name}: {count} boxes (avg score: {avg_score:.3f})")
    else:
        print("No predictions")
    
    if gt_boxes is not None and len(gt_boxes) > 0:
        print(f"\nGround Truth: {len(gt_boxes)} boxes")
    else:
        print("\nNo ground truth boxes")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Simple CODa detection visualization')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/da_models/second_coda32_oracle_3class.yaml',
                        help='Config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process')
    parser.add_argument('--save_dir', type=str, default='vis_data', 
                        help='Directory to save visualization data')
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
        total_epochs=1
    )
    
    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    # Create a simple logger
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    print(f"\nProcessing {args.num_samples} samples...")
    print(f"Classes: {cfg.CLASS_NAMES}")
    print(f"Score threshold: {args.score_thresh}")
    
    with torch.no_grad():
        for idx, batch_dict in enumerate(test_loader):
            if idx >= args.num_samples:
                break
            
            print(f"\n--- Sample {idx} ---")
            
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)
            
            # Get point cloud (remove batch index)
            points = batch_dict['points'][:, 1:4].cpu().numpy()
            print(f"Point cloud: {points.shape[0]} points")
            
            # Get predictions
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
            
            # Filter by score
            mask = pred_scores > args.score_thresh
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
            
            # Get GT boxes if available
            gt_boxes = None
            gt_labels = None
            if 'gt_boxes' in batch_dict:
                gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
                # Remove invalid boxes
                mask = np.any(gt_boxes != 0, axis=1)
                gt_boxes = gt_boxes[mask]
                
                if 'gt_labels' in batch_dict:
                    gt_labels = batch_dict['gt_labels'][0].cpu().numpy()
                    gt_labels = gt_labels[mask]
            
            # Print summary
            print_detection_summary(pred_boxes, pred_scores, pred_labels, gt_boxes, cfg.CLASS_NAMES)
            
            # Save data for external visualization
            save_path = save_dir / f'sample_{idx:04d}.npz'
            save_visualization_data(points, pred_boxes, gt_boxes, pred_labels, gt_labels, save_path)
            
            # Also save as text for easy inspection
            txt_path = save_dir / f'sample_{idx:04d}_summary.txt'
            with open(txt_path, 'w') as f:
                f.write(f"Sample {idx} Detection Results\n")
                f.write("="*50 + "\n")
                f.write(f"Points: {points.shape[0]}\n")
                f.write(f"Predictions: {len(pred_boxes)}\n")
                f.write(f"Ground Truth: {len(gt_boxes) if gt_boxes is not None else 0}\n")
                f.write("\nPrediction Details:\n")
                for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                    cls_name = cfg.CLASS_NAMES[label-1] if label <= len(cfg.CLASS_NAMES) else f"Class_{label}"
                    f.write(f"  [{i}] {cls_name}: score={score:.3f}, "
                           f"pos=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}), "
                           f"size=({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f})\n")
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Data saved to: {save_dir}")
    print(f"You can load the .npz files in Python to visualize with your preferred tool")
    print(f"Example: data = np.load('{save_dir}/sample_0000.npz')")

if __name__ == '__main__':
    main()