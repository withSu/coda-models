import argparse
import numpy as np
import torch
from pathlib import Path
import time
from PIL import Image
import cv2

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def create_bev_image(points, pred_boxes, gt_boxes, img_size=800, pixel_per_meter=8):
    """Create a BEV image for video frames"""
    
    # Create empty image
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Convert points to pixel coordinates
    pc_range = 50  # meters
    if points is not None and len(points) > 0:
        # Filter points in range
        mask = (points[:, 0] > -pc_range) & (points[:, 0] < pc_range) & \
               (points[:, 1] > -pc_range) & (points[:, 1] < pc_range)
        points_filtered = points[mask]
        
        # Convert to pixel coordinates
        px = ((points_filtered[:, 0] + pc_range) * pixel_per_meter).astype(int)
        py = ((points_filtered[:, 1] + pc_range) * pixel_per_meter).astype(int)
        
        # Clip to image bounds
        px = np.clip(px, 0, img_size-1)
        py = np.clip(py, 0, img_size-1)
        
        # Draw points (gray)
        img[py, px] = [100, 100, 100]
    
    # Draw GT boxes (green)
    if gt_boxes is not None:
        for box in gt_boxes:
            if len(box) < 7:
                continue
            x, y, z, dx, dy, dz, heading = box[:7]
            
            # Convert to pixel coordinates
            cx = int((x + pc_range) * pixel_per_meter)
            cy = int((y + pc_range) * pixel_per_meter)
            w = int(dx * pixel_per_meter)
            h = int(dy * pixel_per_meter)
            
            # Draw rotated rectangle
            rect = ((cx, cy), (w, h), np.degrees(heading))
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)
    
    # Draw predicted boxes (red)
    if pred_boxes is not None:
        for box in pred_boxes:
            if len(box) < 7:
                continue
            x, y, z, dx, dy, dz, heading = box[:7]
            score = box[7] if len(box) > 7 else 1.0
            
            # Convert to pixel coordinates
            cx = int((x + pc_range) * pixel_per_meter)
            cy = int((y + pc_range) * pixel_per_meter)
            w = int(dx * pixel_per_meter)
            h = int(dy * pixel_per_meter)
            
            # Draw rotated rectangle
            rect = ((cx, cy), (w, h), np.degrees(heading))
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            color = (0, 0, int(255 * score))  # Red with confidence
            cv2.drawContours(img, [box_points], 0, color, 2)
    
    # Flip Y axis for correct orientation
    img = cv2.flip(img, 0)
    
    # Add legend
    cv2.putText(img, "Green: GT, Red: Pred", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Create video from detection sequence')
    parser.add_argument('--cfg_file', type=str, 
                        default='tools/cfgs/da_models/second_coda32_oracle_3class.yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--num_frames', type=int, default=30, 
                        help='Number of frames for video')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--output', type=str, default='detection_video.mp4',
                        help='Output video file')
    parser.add_argument('--score_thresh', type=float, default=0.3)
    args = parser.parse_args()
    
    # Setup config
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    
    # Build dataloader
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=4,
        logger=None,
        training=False,
        total_epochs=1
    )
    
    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    import logging
    logger = logging.getLogger(__name__)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (800, 800))
    
    print(f"Creating video with {args.num_frames} frames...")
    
    frame_count = 0
    with torch.no_grad():
        for idx, batch_dict in enumerate(test_loader):
            if frame_count >= args.num_frames:
                break
                
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)
            
            # Get data
            points = batch_dict['points'][:, 1:4].cpu().numpy()
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            
            # Filter by score
            mask = pred_scores > args.score_thresh
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            
            # Add scores to boxes
            if len(pred_boxes) > 0:
                pred_boxes = np.concatenate([pred_boxes, pred_scores[:, None]], axis=1)
            
            # Get GT
            gt_boxes = None
            if 'gt_boxes' in batch_dict:
                gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
                mask = np.any(gt_boxes != 0, axis=1)
                gt_boxes = gt_boxes[mask]
            
            # Create BEV image
            img = create_bev_image(points, pred_boxes, gt_boxes)
            
            # Add frame number
            cv2.putText(img, f"Frame {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(img)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count}/{args.num_frames} frames")
    
    # Release video writer
    out.release()
    print(f"\nVideo saved to: {args.output}")
    print(f"Play with: ffplay {args.output}")

if __name__ == '__main__':
    main()