import numpy as np
import argparse
from pathlib import Path

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not installed. Please install with: pip install open3d")

def create_bbox_lines(box):
    """
    Create line set for a 3D bounding box
    box: [x, y, z, dx, dy, dz, heading]
    """
    x, y, z, dx, dy, dz, heading = box[:7]
    
    # Create 8 corners of the box
    corners = np.array([
        [-dx/2, -dy/2, -dz/2],
        [dx/2, -dy/2, -dz/2],
        [dx/2, dy/2, -dz/2],
        [-dx/2, dy/2, -dz/2],
        [-dx/2, -dy/2, dz/2],
        [dx/2, -dy/2, dz/2],
        [dx/2, dy/2, dz/2],
        [-dx/2, dy/2, dz/2]
    ])
    
    # Rotate corners
    rot_matrix = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])
    corners = corners @ rot_matrix.T
    
    # Translate to position
    corners[:, 0] += x
    corners[:, 1] += y
    corners[:, 2] += z
    
    # Define lines connecting corners
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
    ]
    
    # Add direction arrow (front of the box)
    front_center = (corners[1] + corners[2]) / 2
    front_top = (corners[5] + corners[6]) / 2
    arrow_tip = front_center + np.array([np.cos(heading), np.sin(heading), 0]) * dx/2
    
    return corners, lines, arrow_tip

def visualize_sample(npz_file, point_size=1.0, show_grid=True):
    """Visualize a single sample with Open3D"""
    
    if not HAS_OPEN3D:
        print("Open3D is required for 3D visualization")
        return
    
    print(f"Loading {npz_file}...")
    data = np.load(npz_file)
    
    points = data['points']
    pred_boxes = data['pred_boxes'] if 'pred_boxes' in data else np.array([])
    gt_boxes = data['gt_boxes'] if 'gt_boxes' in data else np.array([])
    
    print(f"Points: {len(points)}")
    print(f"Predictions: {len(pred_boxes)}")
    print(f"Ground Truth: {len(gt_boxes)}")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Detection - {npz_file.name}")
    
    # Add point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray color
    vis.add_geometry(pcd)
    
    # Add prediction boxes (RED)
    for i, box in enumerate(pred_boxes):
        if len(box) < 7:
            continue
        corners, lines, arrow_tip = create_bbox_lines(box)
        
        # Create line set for box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0])  # Red for predictions
        vis.add_geometry(line_set)
        
        # Add arrow for direction
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.1, cone_radius=0.2, 
            cylinder_height=1.0, cone_height=0.5
        )
        arrow.paint_uniform_color([1, 0, 0])
        # Position arrow at box center
        arrow.translate([box[0], box[1], box[2]])
        # Rotate arrow to point in heading direction
        R = arrow.get_rotation_matrix_from_xyz([0, 0, box[6]])
        arrow.rotate(R, center=[box[0], box[1], box[2]])
        vis.add_geometry(arrow)
    
    # Add GT boxes (GREEN)
    for box in gt_boxes:
        if len(box) < 7:
            continue
        corners, lines, _ = create_bbox_lines(box)
        
        # Create line set for box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0, 1, 0])  # Green for GT
        vis.add_geometry(line_set)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)
    
    # Add grid if requested
    if show_grid:
        # Create a simple grid on the ground plane
        grid_lines = []
        grid_points = []
        grid_range = 50
        grid_step = 10
        
        for i in range(-grid_range, grid_range + 1, grid_step):
            # Lines parallel to X axis
            grid_points.append([i, -grid_range, 0])
            grid_points.append([i, grid_range, 0])
            grid_lines.append([len(grid_points)-2, len(grid_points)-1])
            
            # Lines parallel to Y axis  
            grid_points.append([-grid_range, i, 0])
            grid_points.append([grid_range, i, 0])
            grid_lines.append([len(grid_points)-2, len(grid_points)-1])
        
        grid = o3d.geometry.LineSet()
        grid.points = o3d.utility.Vector3dVector(grid_points)
        grid.lines = o3d.utility.Vector2iVector(grid_lines)
        grid.paint_uniform_color([0.3, 0.3, 0.3])
        vis.add_geometry(grid)
    
    # Set viewpoint
    view_control = vis.get_view_control()
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_front([1, 1, 1])
    view_control.set_zoom(0.3)
    
    # Instructions
    print("\n" + "="*50)
    print("3D Visualization Controls:")
    print("-"*50)
    print("Mouse:")
    print("  Left button + drag: Rotate view")
    print("  Scroll: Zoom in/out")
    print("  Right button + drag: Pan view")
    print("\nKeyboard:")
    print("  R: Reset viewpoint")
    print("  Q/Esc: Close window")
    print("  +/-: Increase/decrease point size")
    print("  G: Toggle geometry")
    print("  N: Show normals")
    print("\nColors:")
    print("  Gray: Point cloud")
    print("  Red: Predicted boxes")
    print("  Green: Ground truth boxes")
    print("="*50)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='3D visualization of detection results')
    parser.add_argument('--data_dir', type=str, default='vis_data',
                        help='Directory containing .npz files')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--point_size', type=float, default=1.0,
                        help='Point size for visualization')
    parser.add_argument('--no_grid', action='store_true',
                        help='Disable grid visualization')
    args = parser.parse_args()
    
    # Find npz file
    data_dir = Path(args.data_dir)
    npz_file = data_dir / f'sample_{args.sample_idx:04d}.npz'
    
    if not npz_file.exists():
        print(f"File not found: {npz_file}")
        print(f"Available files:")
        for f in sorted(data_dir.glob('*.npz')):
            print(f"  {f.name}")
        return
    
    # Visualize
    visualize_sample(npz_file, args.point_size, not args.no_grid)

if __name__ == '__main__':
    main()