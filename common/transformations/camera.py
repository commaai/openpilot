from __future__ import annotations
import itertools
from typing import Tuple, Dict, Iterator, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt

import openpilot.common.transformations.orientation as orient

@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    focal_length: float

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def intrinsics(self) -> npt.NDArray[np.float64]:
        return np.array([
            [self.focal_length, 0.0, self.width / 2],
            [0.0, self.focal_length, self.height / 2],
            [0.0, 0.0, 1.0]
        ])

    @property
    def intrinsics_inv(self) -> npt.NDArray[np.float64]:
        return np.linalg.inv(self.intrinsics)

@dataclass(frozen=True)
class _NoneCameraConfig(CameraConfig):
    width: int = 0
    height: int = 0
    focal_length: float = 0.0

@dataclass(frozen=True)
class DeviceCameraConfig:
    fcam: CameraConfig
    dcam: CameraConfig
    ecam: CameraConfig

    def all_cams(self) -> Iterator[Tuple[str, CameraConfig]]:
        for cam in ['fcam', 'dcam', 'ecam']:
            if not isinstance(getattr(self, cam), _NoneCameraConfig):
                yield cam, getattr(self, cam)

# Camera configurations
_ar_ox_fisheye = CameraConfig(1928, 1208, 567.0)
_os_fisheye = CameraConfig(2688 // 2, 1520 // 2, 567.0 / 4 * 3)
_ar_ox_config = DeviceCameraConfig(CameraConfig(1928, 1208, 2648.0), _ar_ox_fisheye, _ar_ox_fisheye)
_os_config = DeviceCameraConfig(CameraConfig(2688 // 2, 1520 // 2, 1522.0 * 3 / 4), _os_fisheye, _os_fisheye)
_neo_config = DeviceCameraConfig(CameraConfig(1164, 874, 910.0), CameraConfig(816, 612, 650.0), _NoneCameraConfig())

DEVICE_CAMERAS: Dict[Tuple[str, str], DeviceCameraConfig] = {
    ("neo", "unknown"): _neo_config,
    ("tici", "unknown"): _ar_ox_config,
    ("unknown", "ar0231"): _ar_ox_config,
    ("unknown", "ox03c10"): _ar_ox_config,
    ("pc", "unknown"): _ar_ox_config,
}

# Update DEVICE_CAMERAS with product combinations
DEVICE_CAMERAS.update({
    (device, camera): config
    for device, (camera, config) in itertools.product(
        ('tici', 'tizi', 'mici'),
        (('ar0231', _ar_ox_config), ('ox03c10', _ar_ox_config), ('os04c10', _os_config))
    )
})

# Constants
DEVICE_FRAME_FROM_VIEW_FRAME: npt.NDArray[np.float64] = np.array([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.]
])
VIEW_FRAME_FROM_DEVICE_FRAME: npt.NDArray[np.float64] = DEVICE_FRAME_FROM_VIEW_FRAME.T

def get_view_frame_from_road_frame(roll: float, pitch: float, yaw: float, height: float) -> npt.NDArray[np.float64]:
    device_from_road = orient.rot_from_euler([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
    view_from_road = VIEW_FRAME_FROM_DEVICE_FRAME.dot(device_from_road)
    return np.hstack((view_from_road, [[0], [height], [0]]))

def get_view_frame_from_calib_frame(roll: float, pitch: float, yaw: float, height: float) -> npt.NDArray[np.float64]:
    device_from_calib = orient.rot_from_euler([roll, pitch, yaw])
    view_from_calib = VIEW_FRAME_FROM_DEVICE_FRAME.dot(device_from_calib)
    return np.hstack((view_from_calib, [[0], [height], [0]]))

def vp_from_ke(m: npt.NDArray[np.float64]) -> Tuple[float, float]:
    return (m[0, 0] / m[2, 0], m[1, 0] / m[2, 0])

def roll_from_ke(m: npt.NDArray[np.float64]) -> float:
    return np.arctan2(-(m[1, 0] - m[1, 1] * m[2, 0] / m[2, 1]),
                      -(m[0, 0] - m[0, 1] * m[2, 0] / m[2, 1]))

def normalize(img_pts: npt.NDArray[np.float64], intrinsics: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    intrinsics_inv = np.linalg.inv(intrinsics)
    img_pts = np.atleast_2d(img_pts)
    img_pts_homogeneous = np.hstack((img_pts, np.ones((img_pts.shape[0], 1))))
    img_pts_normalized = img_pts_homogeneous.dot(intrinsics_inv.T)
    img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
    return img_pts_normalized[:, :2].reshape(img_pts.shape)

def denormalize(img_pts: npt.NDArray[np.float64], intrinsics: npt.NDArray[np.float64],
                width: float = np.inf, height: float = np.inf) -> npt.NDArray[np.float64]:
    img_pts = np.atleast_2d(img_pts)
    img_pts_homogeneous = np.hstack((img_pts, np.ones((img_pts.shape[0], 1), dtype=img_pts.dtype)))
    img_pts_denormalized = img_pts_homogeneous.dot(intrinsics.T)

    if np.isfinite(width):
        img_pts_denormalized[np.logical_or(img_pts_denormalized[:, 0] > width, img_pts_denormalized[:, 0] < 0)] = np.nan
    if np.isfinite(height):
        img_pts_denormalized[np.logical_or(img_pts_denormalized[:, 1] > height, img_pts_denormalized[:, 1] < 0)] = np.nan

    return img_pts_denormalized[:, :2].reshape(img_pts.shape)

def get_calib_from_vp(vp: npt.NDArray[np.float64], intrinsics: npt.NDArray[np.float64]) -> Tuple[float, float, float]:
    vp_norm = normalize(vp, intrinsics)
    yaw_calib = np.arctan(vp_norm[0])
    pitch_calib = -np.arctan(vp_norm[1] * np.cos(yaw_calib))
    roll_calib = 0
    return roll_calib, pitch_calib, yaw_calib

def device_from_ecef(pos_ecef: npt.NDArray[np.float64], orientation_ecef: npt.NDArray[np.float64],
                     pt_ecef: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    pt_ecef = np.atleast_2d(pt_ecef)
    ecef_from_device_rot = orient.rotations_from_quats(orientation_ecef)
    device_from_ecef_rot = ecef_from_device_rot.T
    pt_ecef_rel = pt_ecef - pos_ecef
    pt_device = np.einsum('jk,ik->ij', device_from_ecef_rot, pt_ecef_rel)
    return pt_device.reshape(pt_ecef.shape)

def img_from_device(pt_device: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    pt_device = np.atleast_2d(pt_device)
    pt_view = np.einsum('jk,ik->ij', VIEW_FRAME_FROM_DEVICE_FRAME, pt_device)
    pt_view[pt_view[:, 2] < 0] = np.nan
    pt_img = pt_view / pt_view[:, 2:3]
    return pt_img.reshape(pt_device.shape)[:, :2]