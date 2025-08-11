# Copyright (c) AMRL. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

from glob import glob
import os
import json
from os.path import join, isfile
import shutil

import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from PIL import Image

import tqdm
from multiprocessing import Pool

class CODa2KITTI(object):
    """CODa to KITTI converter.
    This class serves as the converter to change the CODa raw data to KITTI
    format.
    Args:
        load_dir (str): Directory to load CODa raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 workers=64,
                 split="training",
                 test_mode=False,
                 channels=128):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.lidar_list = [
            'os1'
        ]
        self.type_list = [
            'CAR', 
            'PEDESTRIAN',
            'BIKE', 
            'MOTORCYCLE',
            'SCOOTER',
            'TREE',
            'TRAFFIC SIGN',
            'CANOPY',
            'TRAFFIC LIGHT',
            'BIKE RACK',
            'BOLLARD',
            'PARKING KIOSK',
            'MAILBOX',
            'FIRE HYDRANT',
            'FREESTANDING PLANT',
            'POLE',
            'INFORMATIONAL SIGN',
            'DOOR',
            'FENCE',
            'RAILING',
            'CONE',
            'CHAIR',
            'BENCH',
            'TABLE',
            'TRASH CAN',
            'NEWSPAPER DISPENSER',
            'ROOM LABEL',
            'STANCHION',
            'SANITIZER DISPENSER',
            'CONDIMENT DISPENSER',
            'VENDING MACHINE',
            'EMERGENCY AID KIT',
            'FIRE EXTINGUISHER',
            'COMPUTER',
            'OTHER',
            'HORSE',
            'PICKUP TRUCK',  
            'DELIVERY TRUCK', 
            'SERVICE VEHICLE', 
            'UTILITY VEHICLE', 
            'FIRE ALARM',
            'ATM',
            'CART',
            'COUCH',
            'TRAFFIC ARM',
            'WALL SIGN',
            'FLOOR SIGN',
            'DOOR SWITCH',
            'EMERGENCY PHONE',
            'DUMPSTER',
            'SEGWAY',
            'BUS',
            'SKATEBOARD',
            'WATER FOUNTAIN'
            # Classes below have not been annotated       
            # 'GOLF CART'
            # 'TRUCK'
            # 'CONSTRUCTION BARRIER'
            # 'TELEVISION',
            # 'VACUUM CLEANER',       
        ]
        self.coda_to_kitti_class_map = {
            # Full Class List
            'CAR': 'Car',
            'PEDESTRIAN': 'Pedestrian',
            'BIKE': 'Cyclist',
            'MOTORCYCLE': 'Motorcycle',
            'SCOOTER': 'Scooter',
            'TREE': 'Tree',
            'TRAFFIC SIGN': 'TrafficSign',
            'CANOPY': 'Canopy',
            'TRAFFIC LIGHT': 'TrafficLight',
            'BIKE RACK': 'BikeRack',
            'BOLLARD': 'Bollard',
            'CONSTRUCTION BARRIER': 'ConstructionBarrier',
            'PARKING KIOSK': 'ParkingKiosk',
            'MAILBOX': 'Mailbox',
            'FIRE HYDRANT': 'FireHydrant',
            'FREESTANDING PLANT': 'FreestandingPlant',
            'POLE': 'Pole',
            'INFORMATIONAL SIGN': 'InformationalSign',
            'DOOR': 'Door',
            'FENCE': 'Fence',
            'RAILING': 'Railing',
            'CONE': 'Cone',
            'CHAIR': 'Chair',
            'BENCH': 'Bench',
            'TABLE': 'Table',
            'TRASH CAN': 'TrashCan',
            'NEWSPAPER DISPENSER': 'NewspaperDispenser',
            'ROOM LABEL': 'RoomLabel',
            'STANCHION': 'Stanchion',
            'SANITIZER DISPENSER': 'SanitizerDispenser',
            'CONDIMENT DISPENSER': 'CondimentDispenser',
            'VENDING MACHINE': 'VendingMachine',
            'EMERGENCY AID KIT': 'EmergencyAidKit',
            'FIRE EXTINGUISHER': 'FireExtinguisher',
            'COMPUTER': 'Computer',
            'TELEVISION': 'Television',
            'OTHER': 'Other',
            'HORSE': 'Other',
            'PICKUP TRUCK': 'PickupTruck',  
            'DELIVERY TRUCK': 'DeliveryTruck', 
            'SERVICE VEHICLE': 'ServiceVehicle', 
            'UTILITY VEHICLE': 'UtilityVehicle',
            'FIRE ALARM': 'FireAlarm',
            'ATM': 'ATM',
            'CART': 'Cart',
            'COUCH': 'Couch',
            'TRAFFIC ARM': 'TrafficArm',
            'WALL SIGN': 'WallSign',
            'FLOOR SIGN': 'FloorSign',
            'DOOR SWITCH': 'DoorSwitch',
            'EMERGENCY PHONE': 'EmergencyPhone',
            'DUMPSTER': 'Dumpster',
            'VACUUM CLEANER': 'VacuumCleaner',
            'SEGWAY': 'Segway',
            'BUS': 'Bus',
            'SKATEBOARD': 'Skateboard',
            'WATER FOUNTAIN': 'WaterFountain'
        }
        #MAP Classes not found in KITTI to DontCare
        for class_type in self.type_list:
            class_name = class_type.upper()
            if class_name not in self.coda_to_kitti_class_map.keys():
                self.coda_to_kitti_class_map[class_name] = 'DontCare'

        self.coda_to_kitti_occlusion = {
            "None":     0,
            "unknown":  0, 
            "Unknown":  0,
            "Light":    1,
            "Medium":   1,
            "Heavy":    2,
            "Full":     2
        }
        # ★ 여기 추가
        self.load_dir = load_dir
        
        # save_dir는 coda 변환 데이터셋 루트 (예: data/coda32_allclass_full)
        self.save_dir = save_dir
        self.split = split
        self.workers = int(workers)
        self.test_mode = test_mode
        
        
        self.bbox_label_files = []
        self.image_files = []   # cam0 경로 저장 용
        self.lidar_files = []
        self.cam_ids = [0]      # cam0만 사용
        
        
        kitti_split = 'testing' if self.test_mode else 'training'
        self.kitti_image2_dir = os.path.join(self.save_dir, self.split, 'image_2')
        self.kitti_label2_dir = os.path.join(self.save_dir, self.split, 'label_2')
        self.point_cloud_save_dir = os.path.join(self.save_dir, self.split, 'velodyne')
        self.calib_save_dir = os.path.join(self.save_dir, 'calib')
        self.pose_save_dir = os.path.join(self.save_dir, self.split, 'pose')
        self.timestamp_save_dir = os.path.join(self.save_dir, self.split, 'timestamp')
        self.imageset_save_dir = os.path.join(self.save_dir, 'ImageSets')

        os.makedirs(self.kitti_image2_dir, exist_ok=True)
        os.makedirs(self.kitti_label2_dir, exist_ok=True)
        os.makedirs(self.point_cloud_save_dir, exist_ok=True)
        os.makedirs(self.calib_save_dir, exist_ok=True)
        os.makedirs(self.pose_save_dir, exist_ok=True)
        os.makedirs(self.timestamp_save_dir, exist_ok=True)
        os.makedirs(self.imageset_save_dir, exist_ok=True)
  
        # Used to downsample lidar vertical channels
        self.channels = channels

        self.process_metadata()
        self.create_folder()
        self.create_imagesets()

    def process_metadata(self):
        metadata_path = join(self.load_dir, "metadata")
        assert os.path.exists(metadata_path), "Metadata directory %s does not exist" % metadata_path

        metadata_files = glob("%s/*.json" % metadata_path)
        metadata_files = sorted(metadata_files, key=lambda fname: int(fname.split('/')[-1].split('.')[0]) )

        for mfile in metadata_files:
            assert os.path.isfile(mfile), '%s does not exist' % mfile
            meta_json = json.load(open(mfile, "r"))

            label_list = meta_json["ObjectTracking"][self.split]
            self.bbox_label_files.extend(label_list)

            lidar_list = [label_path.replace('3d_label', '3d_raw').replace('.json', '.bin') 
                for label_path in label_list]
            self.lidar_files.extend(lidar_list)

            image_list = [label_path.replace('3d_label', '2d_rect')
                .replace('os1', 'cam0').replace('.json', '.png') for label_path in label_list]
            self.image_files.extend(image_list)

    def create_imagesets(self):
        if self.split=="testing":
            imageset_file = "test.txt"
        elif self.split=="training":
            imageset_file = "train.txt"
        elif self.split=="validation":
            imageset_file = "val.txt"

        imageset_path = join(self.imageset_save_dir, imageset_file)
        imageset_fp = open(imageset_path, 'w+')
        
        for lidar_path in self.lidar_files:
            lidar_file = lidar_path.split('/')[-1]
            _, _, traj, frame_idx = self.get_filename_info(lidar_file)
            frame_name = f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}'
            imageset_fp.write(frame_name+'\n')

        imageset_fp.close()

    def convert(self):
        """Convert action."""
        print('Start converting ...')


        pool = Pool(processes=self.workers)

        file_list = list(range(len(self)))
        for _ in tqdm.tqdm(pool.imap_unordered(self.convert_one, [(self, i) for i in file_list]), total=len(file_list)):
            pass
        print('\nFinished ...')
    
    @staticmethod
    def get_filename_info(filename):
        filename_prefix  = filename.split('.')[0]
        filename_prefix  = filename_prefix.split('_')
        
        modality        = filename_prefix[0]+"_"+filename_prefix[1]
        sensor_name     = filename_prefix[2]
        trajectory      = filename_prefix[3]
        frame           = filename_prefix[4]
        return (modality, sensor_name, trajectory, frame)

    @staticmethod
    def set_filename_by_prefix(modality, sensor_name, trajectory, frame):
        if "2d_rect"==modality:
            filetype = "jpg" # change to jpg later
        elif "2d_bbox"==modality:
            filetype = "txt"
        elif "3d_raw"==modality:
            filetype = "bin"
        elif "3d_bbox"==modality:
            filetype = "json"
        sensor_filename = "%s_%s_%s_%s.%s" % (
            modality, 
            sensor_name, 
            trajectory,
            frame,
            filetype
            )
        return sensor_filename

    @staticmethod
    def get_calibration_info(filepath):
        filename = filepath.split('/')[-1]
        filename_prefix = filename.split('.')[0]
        filename_split = filename_prefix.split('_')

        calibration_info = None
        src, tar = filename_split[1], filename_split[-1]
        if len(filename_split) > 3:
            #Sensor to Sensor transform
            extrinsic = yaml.safe_load(open(filepath, 'r'))
            calibration_info = extrinsic
        else:
            #Intrinsic transform
            intrinsic = yaml.safe_load(open(filepath, 'r'))
            calibration_info = intrinsic
        
        return calibration_info, src, tar

    def load_calibrations(self, outdir, trajectory):
        calibrations_path = os.path.join(outdir, "calibrations", str(trajectory))
        calibration_fps = [os.path.join(calibrations_path, file) for file in os.listdir(calibrations_path) if file.endswith(".yaml")]

        calibrations = {}
        for calibration_fp in calibration_fps:
            cal, src, tar = self.get_calibration_info(calibration_fp)
            cal_id = "%s_%s"%(src, tar)

            if cal_id not in calibrations.keys():
                calibrations[cal_id] = {}

            calibrations[cal_id].update(cal)
        
        return calibrations

    def convert_one(self, args):
        _, file_idx = args
        relpath = self.bbox_label_files[file_idx]
        filename = relpath.split('/')[-1]
        _, _, traj, frame_idx = self.get_filename_info(filename)

        # 이미지: cam0만 시도, 없으면 더미 생성
        cam_id = 0
        cam = "cam0"
        img_file = self.set_filename_by_prefix("2d_rect", cam, traj, frame_idx)
        img_path = join(self.load_dir, '2d_rect', cam, str(traj), img_file)
        self.save_image_or_dummy(traj, img_path, frame_idx)

        if not self.test_mode:
            self.save_label_kitti_single(traj, frame_idx)  # label_2에 1개 파일만 쓴다

        self.save_calib_cam0_only(traj, frame_idx)  # cam0 기반 P2 작성
        ok_lidar = self.save_lidar(traj, frame_idx, file_idx, self.channels)
        if not ok_lidar:
            return None

        self.save_pose(traj, frame_idx, file_idx)
        self.save_timestamp(traj, frame_idx, file_idx)
        
        
    def __len__(self):
        """Length of the filename list."""
        return len(self.bbox_label_files)

    def save_image_or_dummy(self, traj, src_img_path, frame_idx):
        base, _ = os.path.splitext(src_img_path)
        found = None
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
            p = base + ext
            if os.path.isfile(p):
                found = p
                break

        out_path = os.path.join(self.kitti_image2_dir, f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.jpg')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if found:
            shutil.copyfile(found, out_path)
        else:
            img = Image.new('RGB', (1242, 375))
            img.save(out_path, format='JPEG')

    def save_label_kitti_single(self, traj, frame_idx):
        anno_file = self.set_filename_by_prefix("3d_bbox", "os1", traj, frame_idx)
        anno_path = join(self.load_dir, "3d_bbox", "os1", traj, anno_file)
        anno_dict = json.load(open(anno_path))

        calibrations = self.load_calibrations(self.load_dir, traj)
        Tr_os1_to_cam0 = np.array(calibrations['os1_cam0']['extrinsic_matrix']['data']).reshape(4, 4)

        out_path = os.path.join(self.kitti_label2_dir, f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt')
        with open(out_path, 'w') as fp:
            for obj in anno_dict["3dbbox"]:
                name = obj['classId'].upper()
                if name not in self.type_list:
                    continue
                if name not in self.coda_to_kitti_class_map:
                    continue
                cls = self.coda_to_kitti_class_map[name]

                h = obj['h']; l = obj['l']; w = obj['w']
                x = obj['cX']; y = obj['cY']; z = obj['cZ'] - h / 2
                pt = Tr_os1_to_cam0 @ np.array([x, y, z, 1]).reshape(4, 1)
                x, y, z, _ = pt.flatten().tolist()
                rotation_y = -obj['y'] - np.pi / 2

                truncated = 0.0
                occluded = 0
                alpha = -10.0
                bbox = [0.0, 0.0, 0.0, 0.0]

                line = f"{cls} {truncated} {occluded} {alpha} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {h} {w} {l} {x} {y} {z} {rotation_y}\n"
                fp.write(line)

    def save_calib_cam0_only(self, traj, frame_idx):
        calibrations = self.load_calibrations(self.load_dir, traj)

        # P2: 12개 숫자(3x4) 그대로
        P2_vals = calibrations['cam0_intrinsics']['projection_matrix']['data']
        P2 = [f'{v:e}' for v in P2_vals]

        # R0: 3x3 항등행렬 → 숫자 9개만 보장해서 기록
        R0_vals = np.eye(3).flatten()               # 길이 9
        R0 = [f'{v:e}' for v in R0_vals][:9]        # 혹시 모를 꼬임 방지로 [:9]

        # Tr_velo_to_cam: 3x4(12개)만 기록
        Tr_os1_to_cam0 = np.array(
            calibrations['os1_cam0']['extrinsic_matrix']['data']
        ).reshape(4, 4)
        Tr = [f'{v:e}' for v in Tr_os1_to_cam0[:3, :].reshape(12,)]

        lines = []
        for k in range(4):
            lines.append(f"P{k}: " + ' '.join(P2))
        lines.append('R0: ' + ' '.join(R0))                # ← R0는 9개
        lines.append('Tr_velo_to_cam: ' + ' '.join(Tr))    # ← 12개

        out_path = os.path.join(
            self.calib_save_dir, f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt'
        )
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))


    def save_pose(self, traj, frame_idx, file_idx):
        pose_dir = join(self.load_dir, "poses", "imu")
        pose_path = join(pose_dir, f"{traj}.txt")
        assert isfile(pose_path), f"Pose file for traj {traj} does not exist: {pose_path}"
        pose_np = np.loadtxt(pose_path, skiprows=int(frame_idx), max_rows=1)

        pose_T = pose_np[1:4].reshape(3, -1)
        pose_quat_xyzw = np.append(pose_np[5:8], pose_np[4])
        pose_R = R.from_quat(pose_quat_xyzw).as_matrix()

        pose_kitti = np.eye(4)
        pose_kitti[:3, :] = np.hstack((pose_R, pose_T))
        np.savetxt(join(self.pose_save_dir, f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt'), pose_kitti)

    # 파일: tools/data_converter/coda_converter.py  (save_lidar)

    def save_lidar(self, traj, frame_idx, file_idx, channels):
        from os.path import isfile, join
        import os, shutil

        base_dir = join(self.load_dir, "3d_raw", "os1", str(traj))

        # 후보 파일명에서 file_idx → frame_idx 로 교체
        candidates = [
            join(base_dir, f"3d_raw_os1_{traj}_{frame_idx}.bin"),
            join(base_dir, f"3d_raw_os1_{traj}_{str(frame_idx).zfill(4)}.bin"),
            join(base_dir, f"3d_raw_os1_{traj}_{str(frame_idx).zfill(5)}.bin"),
            join(base_dir, f"3d_comp_os1_{traj}_{frame_idx}.bin"),
            join(base_dir, f"3d_comp_os1_{traj}_{str(frame_idx).zfill(4)}.bin"),
            join(base_dir, f"3d_comp_os1_{traj}_{str(frame_idx).zfill(5)}.bin"),
        ]

        src_path = None
        for cand in candidates:
            if isfile(cand):
                src_path = cand
                break

        if src_path is None:
            print(f"[WARN] lidar missing, skip traj={traj} frame={frame_idx}")
            return False

        out_bin = os.path.join(self.point_cloud_save_dir, f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.bin')
        os.makedirs(os.path.dirname(out_bin), exist_ok=True)
        shutil.copyfile(src_path, out_bin)
        return True

    def downsample_lidar(self, pc, channels):
        # Downsamples by selecting vertical channels with different step sizes
        vert_ds = 128 // int(channels)
        pc = pc[:, :4].reshape(128, 1024, -1)
        ds_pc = pc[np.arange(0, 128, vert_ds), :, :]
        return ds_pc.reshape(-1, 4)

    def save_label(self, traj, cam_id, frame_idx, file_idx):
        """Parse and save the label data in txt format.
        The relation between coda and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (coda) -> l, h, w (kitti)
        2. x-y-z: front-left-up (coda) -> right-down-front(kitti)
        3. bbox origin at volumetric center (coda) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (coda)
        Args:
            traj (str): Current trajectory index.
            frame_idx (str): Current frame index.
        """
        import os
        import json
        import numpy as np
        from os.path import join

        # 3D 라벨은 필수라서 기존대로 읽는다.
        anno_file = self.set_filename_by_prefix("3d_bbox", "os1", traj, frame_idx)
        anno_path = join(self.load_dir, "3d_bbox", "os1", traj, anno_file)
        anno_dict = json.load(open(anno_path))

        # [MOD] 2D bbox는 CODa 전체에 항상 존재하지 않으므로, 없을 때를 대비한 예외처리를 한다.
        twod_anno_file = self.set_filename_by_prefix("2d_bbox", "cam0", traj, frame_idx)
        twod_anno_path = join(self.load_dir, "2d_bbox", "cam0", traj, twod_anno_file)

        twod_anno_dict = None
        if os.path.exists(twod_anno_path):
            try:
                # CODa devkit 2D 포맷이 id,xmin,ymin,xmax,ymax,count 등일 수 있어 2:6만 사용한다.
                twod_anno_dict = np.loadtxt(twod_anno_path, dtype=float).reshape(-1, 6)
            except Exception as e:
                print(f"[WARN] 2D bbox read error at {twod_anno_path}: {e}; using zeros.")
                twod_anno_dict = None  # [MOD] 실패 시 None으로 처리한다.

        # 캘리브레이션
        calibrations = self.load_calibrations(self.load_dir, traj)
        Tr_os1_to_camx = np.array(calibrations['os1_cam0']['extrinsic_matrix']['data']).reshape(4, 4)

        if cam_id == 1:
            R_cam0_to_cam1 = np.array(calibrations['cam0_cam1']['extrinsic_matrix']['R']['data']).reshape(3, 3)
            T_cam0_to_cam1 = np.array(calibrations['cam0_cam1']['extrinsic_matrix']['T']).reshape(3, 1)
            Tr_cam0_to_cam1 = np.eye(4)
            Tr_cam0_to_cam1[:3, :] = np.hstack((R_cam0_to_cam1, T_cam0_to_cam1))
            Tr_os1_to_camx = Tr_cam0_to_cam1 @ Tr_os1_to_camx

        # 전체 라벨 파일
        fp_label_all = open(
            f'{self.label_all_save_dir}/' +
            f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt', 'w+')

        # 개별 카메라 라벨 파일 초기화
        fp_label = open(
            f'{self.label_save_dir}{cam_id}/' +
            f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt', 'w')
        fp_label.close()

        for anno_idx, tred_anno in enumerate(anno_dict["3dbbox"]):
            name = tred_anno['classId'].upper()
            if name not in self.type_list:
                continue
            if name not in self.coda_to_kitti_class_map:
                continue
            my_type = self.coda_to_kitti_class_map[name]

            # [MOD] 2D bbox가 없거나 인덱스 범위를 넘어가면 0,0,0,0으로 대체한다.
            if twod_anno_dict is None:
                bbox = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                try:
                    # 열 2:6이 xmin,ymin,xmax,ymax라고 가정한다.
                    bbox = twod_anno_dict[anno_idx, 2:6].astype(float)
                except Exception:
                    bbox = np.array([0.0, 0.0, 0.0, 0.0])

            # 크기 변환(CODa l,w,h -> KITTI l,h,w 로 쓰기 위해 매핑)
            height = tred_anno['h']
            length = tred_anno['l']
            width  = tred_anno['w']

            # 중심 좌표, KITTI는 바닥 중심을 쓰므로 z에서 h/2 보정
            x = tred_anno['cX']
            y = tred_anno['cY']
            z = tred_anno['cZ'] - height / 2

            truncated = 0
            alpha = -10  # [MOD] 이미지 기반 각도 부재 시 고정값
            coda_occluded = tred_anno.get('labelAttributes', {}).get('isOccluded', 'Unknown')
            occluded = self.coda_to_kitti_occlusion.get(coda_occluded, 0)

            # LiDAR → 카메라 변환
            pt_ref = Tr_os1_to_camx @ np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            # 회전 변환(yaw 축 정의 차이 보정)
            rotation_y = -tred_anno['y'] - np.pi / 2

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bbox[0], 2), round(bbox[1], 2), round(bbox[2], 2), round(bbox[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            line_all = line[:-1] + '\n'

            # 카메라별 라벨 파일 append
            with open(
                f'{self.label_save_dir}{cam_id}/' +
                f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt', 'a') as fp_label:
                fp_label.write(line)

            fp_label_all.write(line_all)

        fp_label_all.close()



    def save_timestamp(self, traj, frame_idx, file_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.
        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.
        Args:
            traj (str): Current trajectory
            frame_idx (str): Current frame index.
        """
        ts_dir = join(self.load_dir, "timestamps")
        frame_to_ts_file = "%s.txt" % traj
        frame_to_ts_path = join(ts_dir, frame_to_ts_file)
        ts_s_np = np.loadtxt(frame_to_ts_path, skiprows=int(frame_idx), max_rows=1)
        ts_us_np = int(ts_s_np * 1e6)

        with open(
                join(f'{self.timestamp_save_dir}/' +
                     f'{str(traj).zfill(2)}{str(frame_idx).zfill(5)}.txt'),
                'w') as f:
            f.write(str(ts_us_np))


    def create_folder(self):
        # LiDAR-only: __init__에서 필요한 디렉터리를 모두 생성했으므로 여기서는 아무 것도 하지 않는다.
        return
    
    def create_folder(self):
        return
