# tmah_vlm pipeline structure

이 패키지는 C++ ROS node처럼 구조가 보이도록 정리했다.

## vlm_node.py

`vlm_node.py`는 전체 시스템의 지휘자 역할만 한다.

```text
initialize_state()
initialize_modules()
initialize_subscribers()
initialize_publishers()
initialize_timers()

Callback
  question_callback()
  pose_callback()
  image_callback()
  scan_callback()

Main control loop
  main_control_loop()
    -> dispatch_question()
       -> handlers/object_reference.py
       -> handlers/numerical.py
       -> handlers/instruction.py
```

## object_reference.py

`Find ...` 질문이 들어왔을 때 실제 object finding pipeline을 단계적으로 실행한다.

```text
Stage 0. input check
Stage 1. prepare image
Stage 2. parse question
Stage 3. detect 2D candidates
Stage 4. select candidate
Stage 5. transform scan to map frame
Stage 6. estimate 3D target
Stage 7. make approach waypoint
Stage 8. publish result
Stage 9. save debug files
```

## tf/coordinate_transform.py

좌표 변환 전용 파일이다.

```text
camera ray -> map ray
sensor scan points -> map points
camera origin -> map origin
```

기본 fallback TF tree는 `config.py`의 `STATIC_TF_FALLBACKS`에 둔다.

```text
map
 └── sensor
      └── camera
```
