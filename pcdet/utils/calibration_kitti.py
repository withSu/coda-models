import numpy as np


def get_calib_from_file(calib_file, use_coda=False):
    import numpy as np

    with open(calib_file, 'r') as f:
        lines = f.read().strip().splitlines()

    def has_prefix(p):
        return any(ln.startswith(p + ':') for ln in lines)

    def parse_line(prefix, expected_len):
        ln = next((ln for ln in lines if ln.startswith(prefix + ':')), None)
        assert ln is not None, f'{prefix}: line missing in {calib_file}'
        # 콜론 뒤만 파싱 (정규식이 'R0'의 0까지 집계하는 문제 방지)
        vals = np.fromstring(ln.split(':', 1)[1], sep=' ')
        # 길이 보정(넘치면 자르고, 모자라면 0으로 패딩)
        if vals.size > expected_len:
            vals = vals[:expected_len]
        elif vals.size < expected_len:
            vals = np.pad(vals, (0, expected_len - vals.size))
        return vals

    # P2, P3
    P2 = parse_line('P2', 12)
    if has_prefix('P3'):
        P3 = parse_line('P3', 12)
    else:
        # P3 없으면 P2를 복제 (일부 변환 파이프라인은 P3 미생성)
        P3 = P2.copy()

    # R0 또는 R0_rect 둘 다 허용
    if has_prefix('R0'):
        R0 = parse_line('R0', 9)
    elif has_prefix('R0_rect'):
        R0 = parse_line('R0_rect', 9)
    else:
        # 없으면 단위행렬
        R0 = np.eye(3, dtype=np.float32).reshape(-1)

    # Tr_velo_to_cam (이름이 Tr_velo2cam으로 저장된 코드도 있어 호환)
    if has_prefix('Tr_velo_to_cam'):
        Tr = parse_line('Tr_velo_to_cam', 12)
    elif has_prefix('Tr_velo2cam'):
        Tr = parse_line('Tr_velo2cam', 12)
    else:
        raise AssertionError(f'Tr_velo_to_cam line missing in {calib_file}')

    return {
        'P2': P2.reshape(3, 4),
        'P3': P3.reshape(3, 4),
        'R0': R0.reshape(3, 3),
        'Tr_velo2cam': Tr.reshape(3, 4),  # Calibration 클래스가 이 키를 기대하는 경우가 많음
    }


class Calibration(object):
    def __init__(self, calib_file, use_coda=False):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file, use_coda)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner
