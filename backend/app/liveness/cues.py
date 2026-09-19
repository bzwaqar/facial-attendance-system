"""Lightweight cue extractors for head pose, eye dynamics, and image texture."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class HeadPoseCues:
    """Estimated geometric head pose proxies from 5 facial landmarks."""

    yaw_proxy: float       # In [-1.0, 1.0], ~0.0 = frontal, >0 = turned left, <0 = turned right
    pitch_proxy: float     # In [-1.0, 1.0], vertical tilt proxy
    roll_degrees: float    # In-plane rotation angle in degrees
    eye_distance: float    # Pixel distance between left and right eyes
    center_x: float        # Normalized center X of face
    center_y: float        # Normalized center Y of face


def estimate_head_pose_cues(
    landmarks: List[List[float]],
    frame_shape: Optional[Tuple[int, int]] = None,
) -> Optional[HeadPoseCues]:
    """Compute normalized head pose and geometric proxies from 5 SCRFD facial landmarks.
    
    Landmark indexing (SCRFD convention):
        0: Left eye [x, y]
        1: Right eye [x, y]
        2: Nose tip [x, y]
        3: Left mouth corner [x, y]
        4: Right mouth corner [x, y]
        
    Args:
        landmarks: 5-point landmark coordinates [[x, y], ...].
        frame_shape: Optional (height, width) for coordinate normalization.
        
    Returns:
        HeadPoseCues dataclass or None if landmarks are invalid.
    """
    if not landmarks or len(landmarks) < 5:
        return None

    try:
        pts = np.array(landmarks, dtype=np.float32)
        le = pts[0]  # Left eye
        re = pts[1]  # Right eye
        nose = pts[2]  # Nose tip
        lm = pts[3]  # Left mouth
        rm = pts[4]  # Right mouth

        # 1. Inter-ocular distance & Roll angle
        dx_eyes = float(re[0] - le[0])
        dy_eyes = float(re[1] - le[1])
        eye_distance = float(math.hypot(dx_eyes, dy_eyes))
        if eye_distance < 1.0:
            return None

        roll_rad = math.atan2(dy_eyes, dx_eyes)
        roll_deg = float(math.degrees(roll_rad))

        # 2. Normalized Yaw Proxy
        # Horizontal distance from left eye to nose vs nose to right eye
        dx_le_nose = float(nose[0] - le[0])
        dx_nose_re = float(re[0] - nose[0])
        denom_yaw = dx_le_nose + dx_nose_re + 1e-6
        yaw_proxy = float(np.clip((dx_le_nose - dx_nose_re) / denom_yaw, -1.0, 1.0))

        # 3. Normalized Pitch Proxy
        # Vertical distance from eye midpoint to nose vs nose to mouth midpoint
        eye_mid_y = float((le[1] + re[1]) * 0.5)
        mouth_mid_y = float((lm[1] + rm[1]) * 0.5)
        dy_eye_nose = float(nose[1] - eye_mid_y)
        dy_nose_mouth = float(mouth_mid_y - nose[1])
        denom_pitch = dy_eye_nose + dy_nose_mouth + 1e-6
        pitch_proxy = float(np.clip((dy_eye_nose - dy_nose_mouth) / denom_pitch, -1.0, 1.0))

        # 4. Normalized face center coordinates
        h, w = frame_shape if frame_shape else (1, 1)
        center_x = float(nose[0] / max(1.0, float(w)))
        center_y = float(nose[1] / max(1.0, float(h)))

        return HeadPoseCues(
            yaw_proxy=yaw_proxy,
            pitch_proxy=pitch_proxy,
            roll_degrees=roll_deg,
            eye_distance=eye_distance,
            center_x=center_x,
            center_y=center_y,
        )
    except Exception:
        return None


def extract_eye_patches(
    frame: np.ndarray,
    landmarks: List[List[float]],
    target_size: Tuple[int, int] = (24, 16),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Crop and normalize grayscale eye patches from frame using landmark positions.
    
    Args:
        frame: Input BGR image array.
        landmarks: 5-point landmark coordinates.
        target_size: (width, height) to resize eye patches to.
        
    Returns:
        Tuple of (left_eye_patch, right_eye_patch) as uint8 grayscale arrays, or None.
    """
    if landmarks is None or len(landmarks) < 5 or frame is None or frame.size == 0:
        return None

    try:
        pts = np.array(landmarks, dtype=np.float32)
        le = pts[0]
        re = pts[1]
        eye_dist = math.hypot(float(re[0] - le[0]), float(re[1] - le[1]))
        if eye_dist < 8.0:
            return None

        patch_w = int(max(8, eye_dist * 0.35))
        patch_h = int(max(6, eye_dist * 0.25))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]

        def _crop_patch(center: np.ndarray) -> Optional[np.ndarray]:
            cx, cy = int(round(center[0])), int(round(center[1]))
            x1 = max(0, cx - patch_w // 2)
            y1 = max(0, cy - patch_h // 2)
            x2 = min(w, cx + patch_w // 2)
            y2 = min(h, cy + patch_h // 2)
            if x2 <= x1 or y2 <= y1:
                return None
            patch = gray[y1:y2, x1:x2]
            return cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)

        left_patch = _crop_patch(le)
        right_patch = _crop_patch(re)
        if left_patch is not None and right_patch is not None:
            return (left_patch, right_patch)
        return None
    except Exception:
        return None


def compute_patch_difference(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Compute normalized mean absolute difference between two consecutive eye patches.
    
    Returns:
        Difference value normalized in [0.0, 1.0].
    """
    if patch_a is None or patch_b is None or patch_a.shape != patch_b.shape:
        return 0.0
    diff = np.mean(np.abs(patch_a.astype(np.float32) - patch_b.astype(np.float32)))
    return float(diff / 255.0)


def extract_texture_cues(face_crop: np.ndarray) -> Dict[str, float]:
    """Compute passive image texture, sharpness, and glare cues on a cropped face.
    
    Args:
        face_crop: BGR or grayscale face region crop.
        
    Returns:
        Dictionary of texture metrics:
            - laplacian_variance: Measure of high-frequency sharpness (low = blurred photo)
            - specular_ratio: Ratio of near-saturated pixels (>250, screen reflection cue)
            - contrast: Standard deviation of pixel intensities
    """
    if face_crop is None or face_crop.size == 0:
        return {"laplacian_variance": 0.0, "specular_ratio": 0.0, "contrast": 0.0}

    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        specular_pixels = int(np.count_nonzero(gray >= 250))
        specular_ratio = float(specular_pixels / max(1, gray.size))
        contrast = float(np.std(gray))

        return {
            "laplacian_variance": laplacian_var,
            "specular_ratio": specular_ratio,
            "contrast": contrast,
        }
    except Exception:
        return {"laplacian_variance": 0.0, "specular_ratio": 0.0, "contrast": 0.0}
