"""
DMS 行为指标计算:
  - EAR  (Eye Aspect Ratio)   : 眼睛纵横比, 用于判断闭眼
  - MAR  (Mouth Aspect Ratio) : 嘴巴纵横比, 用于判断打哈欠
  - PERCLOS                   : 单位时间内眼睛闭合比例, 疲劳驾驶核心指标
"""

import numpy as np
from collections import deque


def compute_ear(eye_points: np.ndarray) -> float:
    """
    计算单只眼睛的 EAR (Eye Aspect Ratio).

    eye_points: shape (6, 2), 按如下顺序排列:
        p0(外眼角) - p1(上眼睑左) - p2(上眼睑右)
        p3(内眼角) - p4(下眼睑右) - p5(下眼睑左)

    EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    """
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_mar(mouth_upper: np.ndarray, mouth_lower: np.ndarray) -> float:
    """
    计算 MAR (Mouth Aspect Ratio).
    mouth_upper / mouth_lower: shape (2,), 上唇中心 / 下唇中心坐标
    MAR = |upper - lower| / baseline (简化版本)
    """
    return float(np.linalg.norm(mouth_upper - mouth_lower))


class PERCLOSTracker:
    """
    PERCLOS (Percentage of Eye Closure) 追踪器.
    在滑动窗口内统计眼睛闭合帧占比.

    使用方式:
        tracker = PERCLOSTracker(window_sec=60, fps=30)
        for each_frame:
            ear = compute_ear(landmarks)
            tracker.update(ear)
            if tracker.is_fatigued():
                alert()
    """

    def __init__(self, window_sec: float = 60.0, fps: float = 30.0,
                 ear_threshold: float = 0.20, fatigue_threshold: float = 0.40):
        """
        Args:
            window_sec:        滑动窗口秒数
            fps:               帧率
            ear_threshold:     EAR 低于此值认为闭眼
            fatigue_threshold: PERCLOS 高于此值认为疲劳
        """
        self.max_frames = int(window_sec * fps)
        self.ear_threshold = ear_threshold
        self.fatigue_threshold = fatigue_threshold
        self._history = deque(maxlen=self.max_frames)

    def update(self, ear: float):
        """输入当前帧双眼平均 EAR 值."""
        self._history.append(1 if ear < self.ear_threshold else 0)

    @property
    def perclos(self) -> float:
        """当前 PERCLOS 值 (闭眼帧占比)."""
        if len(self._history) == 0:
            return 0.0
        return sum(self._history) / len(self._history)

    def is_fatigued(self) -> bool:
        """判断当前是否疲劳."""
        return self.perclos >= self.fatigue_threshold

    def reset(self):
        self._history.clear()


class YawnDetector:
    """
    连续打哈欠检测器.
    当 MAR 超过阈值且持续帧数足够时, 判定为一次打哈欠.
    """

    def __init__(self, mar_threshold: float = 30.0,
                 min_frames: int = 15, fps: float = 30.0):
        self.mar_threshold = mar_threshold
        self.min_frames = min_frames
        self._open_count = 0
        self._yawn_total = 0

    def update(self, mar: float) -> bool:
        """返回当前帧是否判定为打哈欠结束 (刚刚完成一次)."""
        if mar > self.mar_threshold:
            self._open_count += 1
            return False
        else:
            triggered = self._open_count >= self.min_frames
            if triggered:
                self._yawn_total += 1
            self._open_count = 0
            return triggered

    @property
    def total_yawns(self) -> int:
        return self._yawn_total
