#!/usr/bin/env python3
"""
Solver별 파이프라인 컨텍스트(ctx) 생성 함수 모음.

각 solver(t1/t2/t3)의 process 함수는 질문 1건을 처리하는 동안 필요한 모든
"작업 변수"를 ctx 하나에 담아 들고 다닌다. 스텝 함수들은 이 ctx를 인자로 받아
자기 담당 필드를 **채우기만** 하고, 다음 스텝 함수가 그 필드를 읽어 이어서 쓴다.
이렇게 하면 process 함수 본문이 `함수(node, ctx)` 호출의 나열로만 남아, 위에서
아래로 읽으면 파이프라인 순서가 그대로 보인다.

ctx는 클래스가 아니라 SimpleNamespace(속성으로 접근하는 단순 구조체)다. 아래 함수들이
필요한 필드를 미리 다 선언해 만들어 주고, 각 스텝 함수가 `ctx.image = ...`처럼 채운다.
인스턴스는 각 process 진입부에서 질문마다 새로 만든다(질문 간 값이 섞이지 않도록).
"""

from types import SimpleNamespace


def make_instruction_context(question):
    """t1: "그 외" 명령형 질문 처리용 (아직 stub — 앞으로 1m 직진 waypoint)."""
    return SimpleNamespace(
        question=question,
        robot_pose={},   # {"x", "y", "z", "yaw"} 스냅샷
        waypoint={},     # {"x", "y", "heading"}
    )


def make_numerical_context(question):
    """t2: "how many / count ..." 개수 세기 처리용 (현재 시야 기준)."""
    return SimpleNamespace(
        question=question,
        image=None,              # PIL.Image
        image_stamp=None,        # 카메라 캡처 시각(TF/ray lookup용)
        detect_prompt="",        # GroundingDINO에 넣을 검출어
        detections=[],           # 2D 후보 박스들
        scan_points_map=None,    # map frame으로 변환된 point cloud (N,3)
        candidate_indices=[],    # 공간관계로 걸러진 후보 index
        count=0,                 # 최종 개수
    )


def make_object_ref_context(question):
    """t3: "find ..." 물체 지시 처리용 (검출 → 후보 좁히기 → 선택 → 3D 위치 → 발행)."""
    return SimpleNamespace(
        question=question,
        image=None,              # PIL.Image
        image_stamp=None,        # 카메라 캡처 시각(TF/ray lookup용)
        detect_prompt="",        # GroundingDINO에 넣을 검출어
        detections=[],           # 2D 후보 박스들
        scan_points_map=None,    # map frame으로 변환된 point cloud (N,3)
        candidate_indices=[],    # 공간관계로 걸러진 후보 index
        selected_index=-1,       # 최종 선택된 후보 index
        selected=None,           # detections[selected_index]
        segmentation_mask=None,  # 선택 물체의 SAM 마스크(실패 시 None)
        result={},               # 3D 위치/크기 추정 결과
        robot_pose={},           # {"x", "y", "z", "yaw"} 스냅샷
        waypoint={},             # 접근 waypoint {"x", "y", "heading"}
    )
