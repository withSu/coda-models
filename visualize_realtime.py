import argparse
import numpy as np
import torch
from pathlib import Path
import cv2
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def create_bev_image(points, pred_boxes, gt_boxes, img_size=800, pixel_per_meter=8):
    """Create a BEV image for display"""
    
    # Create empty image (black background)
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
        
        # Draw points (gray/white based on height)
        heights = points_filtered[:, 2]
        colors = np.clip((heights + 2) * 30, 50, 150).astype(int)
        for i in range(len(px)):
            img[py[i], px[i]] = [colors[i], colors[i], colors[i]]
    
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
            cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # Green
    
    # Draw predicted boxes (red/blue based on class)
    class_colors = {
        1: (0, 0, 255),    # Car - Red
        2: (255, 0, 0),    # Pedestrian - Blue
        3: (0, 255, 255),  # Cyclist - Yellow
    }
    
    if pred_boxes is not None:
        for i, box in enumerate(pred_boxes):
            if len(box) < 7:
                continue
            x, y, z, dx, dy, dz, heading = box[:7]
            score = box[7] if len(box) > 7 else 1.0
            label = int(box[8]) if len(box) > 8 else 1
            
            # Convert to pixel coordinates
            cx = int((x + pc_range) * pixel_per_meter)
            cy = int((y + pc_range) * pixel_per_meter)
            w = int(dx * pixel_per_meter)
            h = int(dy * pixel_per_meter)
            
            # Get color based on class
            color = class_colors.get(label, (255, 255, 255))
            # Adjust brightness based on confidence
            color = tuple(int(c * max(0.3, score)) for c in color)
            
            # Draw rotated rectangle
            rect = ((cx, cy), (w, h), np.degrees(heading))
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            cv2.drawContours(img, [box_points], 0, color, 2)
            
            # Draw score text
            if score < 1.0:
                text = f"{score:.2f}"
                cv2.putText(img, text, (cx-10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Flip Y axis for correct orientation
    img = cv2.flip(img, 0)
    
    # Add grid lines
    grid_color = (30, 30, 30)
    for i in range(0, img_size, img_size//10):
        cv2.line(img, (i, 0), (i, img_size), grid_color, 1)
        cv2.line(img, (0, i), (img_size, i), grid_color, 1)
    
    # Add center crosshair
    center = img_size // 2
    cv2.line(img, (center-20, center), (center+20, center), (100, 100, 100), 1)
    cv2.line(img, (center, center-20), (center, center+20), (100, 100, 100), 1)
    
    return img

def main():
    parser = argparse.ArgumentParser(description='Real-time detection visualization')
    parser.add_argument('--cfg_file', type=str, 
                        default='tools/cfgs/da_models/second_coda32_oracle_3class.yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--delay', type=int, default=100, 
                        help='Delay between frames in ms (lower=faster)')
    parser.add_argument('--score_thresh', type=float, default=0.3)
    parser.add_argument('--loop', action='store_true', 
                        help='Loop through dataset continuously')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting sample index')
    parser.add_argument('--sorted', action='store_true', default=True,
                        help='Use sorted sequence order for CODa dataset')
    args = parser.parse_args()
    
    # Setup config
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    
    # Build dataset with sorted option for CODa
    from pcdet.datasets.coda.coda_dataset import CODataset
    
    test_set = CODataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH),  # Convert to Path object
        logger=None,
        use_sorted_imageset=False  # Don't use sorted imageset to avoid KeyError
    )
    
    # Sort the dataset infos by frame ID to ensure sequence order
    if args.sorted:
        # Sort coda_infos by lidar_idx to maintain sequence order
        test_set.coda_infos = sorted(test_set.coda_infos, 
                                     key=lambda x: int(x['point_cloud']['lidar_idx']))
    
    # Build dataloader without shuffle
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_set, 
        batch_size=1,
        pin_memory=True,
        num_workers=4,
        shuffle=False,  # Important: no shuffle for sequence order
        collate_fn=test_set.collate_batch,
        drop_last=False,
        sampler=None,
        timeout=0
    )
    
    sampler = None
    
    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    
    import logging
    logger = logging.getLogger(__name__)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    # Create window
    window_name = 'CODa Detection Visualization (Press Q to quit, SPACE to pause)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 900)
    
    print("="*60)
    print("Real-time Detection Visualization")
    print("-"*60)
    print("Controls:")
    print("  Q/ESC: Quit")
    print("  SPACE: Pause/Resume")
    print("  S: Slower playback")
    print("  F: Faster playback")
    print("  R: Reset to beginning")
    print("  N: Next frame (when paused)")
    print("-"*60)
    print("Legend:")
    print("  Green boxes: Ground Truth")
    print("  Red boxes: Predicted Cars")
    print("  Blue boxes: Predicted Pedestrians")
    print("  Yellow boxes: Predicted Cyclists")
    print("="*60)
    
    paused = False
    frame_idx = args.start_idx
    delay = args.delay
    
    with torch.no_grad():
        while True:
            # Reset iterator if needed
            if args.loop and frame_idx >= len(test_loader):
                frame_idx = 0
                test_loader_iter = iter(test_loader)
            elif frame_idx >= len(test_loader):
                print("\nReached end of dataset. Press R to restart or Q to quit.")
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    frame_idx = 0
                    test_loader_iter = iter(test_loader)
                continue
            
            # Get batch
            if frame_idx == 0:
                test_loader_iter = iter(test_loader)
            
            # Skip to current frame if needed
            if frame_idx > 0 and frame_idx == args.start_idx:
                for _ in range(frame_idx):
                    next(test_loader_iter)
            
            batch_dict = next(test_loader_iter)
            
            if not paused or frame_idx == args.start_idx:
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model(batch_dict)
                
                # Get data
                points = batch_dict['points'][:, 1:4].cpu().numpy()
                pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
                pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
                pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()
                
                # Filter by score
                mask = pred_scores > args.score_thresh
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
                pred_labels = pred_labels[mask]
                
                # Add scores and labels to boxes
                if len(pred_boxes) > 0:
                    pred_boxes = np.concatenate([
                        pred_boxes, 
                        pred_scores[:, None],
                        pred_labels[:, None]
                    ], axis=1)
                
                # Get GT
                gt_boxes = None
                if 'gt_boxes' in batch_dict:
                    gt_boxes = batch_dict['gt_boxes'][0].cpu().numpy()
                    mask = np.any(gt_boxes != 0, axis=1)
                    gt_boxes = gt_boxes[mask]
                
                # Create BEV image
                img = create_bev_image(points, pred_boxes, gt_boxes)
                
                # Add info text with frame ID
                frame_id = batch_dict.get('frame_id', [None])[0]
                info_text = f"Frame: {frame_idx}/{len(test_loader)} | "
                if frame_id is not None:
                    info_text += f"ID: {frame_id} | "
                info_text += f"Points: {len(points)} | "
                info_text += f"Pred: {len(pred_boxes)} | "
                info_text += f"GT: {len(gt_boxes) if gt_boxes is not None else 0} | "
                info_text += f"Delay: {delay}ms"
                if paused:
                    info_text += " [PAUSED]"
                
                cv2.putText(img, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Class counts
                class_names = ['Car', 'Pedestrian', 'Cyclist']
                pred_text = "Predictions: "
                for i, name in enumerate(class_names):
                    count = np.sum(pred_labels == i+1)
                    if count > 0:
                        pred_text += f"{name}:{count} "
                cv2.putText(img, pred_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display image
                cv2.imshow(window_name, img)
                
                if not paused:
                    frame_idx += 1
            
            # Handle keyboard input
            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE - pause/resume
                paused = not paused
            elif key == ord('s'):  # S - slower
                delay = min(delay + 50, 2000)
            elif key == ord('f'):  # F - faster
                delay = max(delay - 50, 1)
            elif key == ord('r'):  # R - reset
                frame_idx = 0
            elif key == ord('n') and paused:  # N - next frame when paused
                frame_idx += 1
                paused = False
                delay = 0
                # Process one frame then pause again
                continue
    
    cv2.destroyAllWindows()
    print("\nVisualization ended.")

if __name__ == '__main__':
    main()