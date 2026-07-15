#!/usr/bin/env python3
"""
ROS sensor_msgs/Image <-> numpy/PIL 변환 유틸.

cv_bridge 에 의존하지 않고 직접 변환합니다.
(cv_bridge 는 OpenCV 버전 충돌이 잦아서, 픽셀 버퍼를 직접 numpy 로 해석)

지원 인코딩: rgb8, bgr8, mono8  (시뮬 카메라는 보통 rgb8/bgr8)
필요하면 인코딩 케이스를 추가하세요.
"""

import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage

def grab_camera_image(node, ctx):
    # 최신 ROS 이미지를 PIL로 변환하고 캡처 시각(stamp)을 같은 스냅샷에서 함께 꺼낸다.
    # (콜백이 latest_image를 계속 갱신하므로 이미지와 stamp를 따로 읽으면 어긋난다.)
    image_msg = node.latest_image
    ctx.image = ros_image_to_pil(image_msg)
    ctx.image_stamp = image_msg.header.stamp

def ros_image_to_numpy(msg: RosImage) -> np.ndarray:
    """sensor_msgs/Image -> HxWx3 (RGB) numpy uint8 배열."""
    height = msg.height
    width = msg.width
    encoding = msg.encoding.lower()

    # 원시 바이트 -> numpy
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        img = data.reshape((height, width, 3))
        if encoding == "bgr8":
            img = img[:, :, ::-1]  # BGR -> RGB
        return np.ascontiguousarray(img)

    elif encoding == "mono8":
        gray = data.reshape((height, width))
        return np.ascontiguousarray(np.stack([gray] * 3, axis=-1))

    else:
        raise ValueError(
            f"지원하지 않는 이미지 인코딩: '{msg.encoding}'. "
            f"image_utils.py 에 케이스를 추가하세요.")


def ros_image_to_pil(msg: RosImage) -> PILImage.Image:
    # GroundingDINO가 사용할 수 있는 RGB 형식의 PIL 이미지로 바꾸는 함수
    """sensor_msgs/Image -> PIL.Image (RGB). GroundingDINO 입력용."""
    arr = ros_image_to_numpy(msg)
    return PILImage.fromarray(arr, mode="RGB")
