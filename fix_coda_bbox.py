import pickle
import numpy as np
from pcdet.utils import box_utils, calibration_kitti
from pathlib import Path

def boxes3d_kitti_camera_to_imageboxes(boxes3d_camera, calib_info, image_shape=None):
    """
    :param boxes3d_camera: (N, 7) [x, y, z, l, h, w, ry] in camera coords
    :param calib_info: calibration dict with P2, R0_rect, Tr_velo_to_cam
    :return: boxes_2d: (N, 4) [x1, y1, x2, y2] in image coords
    """
    boxes3d_camera = boxes3d_camera.copy()
    
    # Get corners of 3D boxes
    corners_3d = box_utils.boxes_to_corners_3d(boxes3d_camera)
    
    # N x 8 x 3
    pts_3d = corners_3d.reshape(-1, 3)
    
    # Add homogeneous coordinate
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    
    # Project to image plane
    P2 = calib_info['P2']
    pts_2d = np.dot(pts_3d_homo, P2.T)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    # Reshape back to N x 8 x 2
    pts_2d = pts_2d[:, :2].reshape(-1, 8, 2)
    
    # Get min/max for each box
    x1 = np.min(pts_2d[:, :, 0], axis=1)
    y1 = np.min(pts_2d[:, :, 1], axis=1)
    x2 = np.max(pts_2d[:, :, 0], axis=1)
    y2 = np.max(pts_2d[:, :, 1], axis=1)
    
    boxes_2d = np.stack([x1, y1, x2, y2], axis=1)
    
    # Clip to image boundaries if image_shape is provided
    if image_shape is not None:
        boxes_2d[:, 0] = np.clip(boxes_2d[:, 0], 0, image_shape[1])
        boxes_2d[:, 1] = np.clip(boxes_2d[:, 1], 0, image_shape[0])
        boxes_2d[:, 2] = np.clip(boxes_2d[:, 2], 0, image_shape[1])
        boxes_2d[:, 3] = np.clip(boxes_2d[:, 3], 0, image_shape[0])
    
    return boxes_2d

def fix_coda_bbox_in_infos(info_path, output_path=None):
    """Fix bbox in CODa info files by projecting 3D boxes to 2D"""
    
    print(f"Loading {info_path}...")
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    
    print(f"Processing {len(infos)} samples...")
    fixed_count = 0
    
    for i, info in enumerate(infos):
        if i % 500 == 0:
            print(f"  Processing sample {i}/{len(infos)}...")
            
        if 'annos' not in info:
            continue
            
        annos = info['annos']
        
        # Check if bbox needs fixing
        if 'bbox' in annos and len(annos['bbox']) > 0:
            if np.all(annos['bbox'] == 0):
                # Need to fix - compute from 3D boxes
                if 'location' in annos and 'dimensions' in annos and 'rotation_y' in annos:
                    # Get 3D boxes in camera coordinates
                    loc = annos['location']
                    dims = annos['dimensions']
                    rots = annos['rotation_y']
                    
                    # Convert to [x, y, z, l, h, w, ry] format
                    boxes_3d = np.concatenate([loc, dims, rots[:, np.newaxis]], axis=1)
                    
                    # Get calibration info
                    calib_info = info['calib']
                    
                    # Get image shape if available
                    image_shape = info['image']['image_shape'] if 'image' in info else None
                    
                    # Project to 2D
                    boxes_2d = boxes3d_kitti_camera_to_imageboxes(boxes_3d, calib_info, image_shape)
                    
                    # Update bbox
                    annos['bbox'] = boxes_2d.astype(np.float32)
                    fixed_count += 1
    
    print(f"Fixed {fixed_count} samples with zero bbox")
    
    # Save fixed version
    if output_path is None:
        output_path = info_path.replace('.pkl', '_fixed.pkl')
    
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(infos, f)
    
    print("Done!")
    return output_path

if __name__ == "__main__":
    # Fix validation set
    val_info_path = "/home/a/ros2_ws/coda-models/data/coda32_allclass_full/coda_infos_val.pkl"
    val_output = fix_coda_bbox_in_infos(val_info_path)
    
    # Fix training set
    train_info_path = "/home/a/ros2_ws/coda-models/data/coda32_allclass_full/coda_infos_train.pkl"
    train_output = fix_coda_bbox_in_infos(train_info_path)
    
    print(f"\nFixed files saved as:")
    print(f"  Validation: {val_output}")
    print(f"  Training: {train_output}")