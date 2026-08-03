# /home/kante/CMU-VLN-Challenge-2026/ai_module/src/sysnav_ros2_mvp/sysnav/reasoning/gemini_selector.py <-
# 여기서 GeminiSelector.select() 함수는 Gemini API를 호출하여 후보 객체 중 하나를 선택하는 역할을 한다.
# 또한, 물체들의 속성
# 두 물체 간 관계 (on the,, over,,이런 관계)

# 를 잘 잡는지 파악하기 위함

# 이미지는 로봇에서 추출한 이미지인 /home/kante/CMU-VLN-Challenge-2026/ai_module/debug/sysnav_detect_1784607496.591.jpg 을 사용

# 인풋 문장을 생성(이 파일 안에서 " find tv on the table" 이런거)
# 여기서 목표를 추출
# 1. 각 물체의 속성
# 2. 두 물체들의 관계
# 3. 두 그래프의 엣지 생성 (object-object edge)

"""
GeminiSelector.select()는 후보 object_id 하나만 고르는 API라 속성/관계 추출 실험에는 그대로
못 쓴다. 대신 같은 google-genai 클라이언트 호출 방식(config.GEMINI_MODEL, 구조화된
response_schema 출력)을 재사용한 수동 실행 스크립트.

QUESTION은 "find A on B, and find C in front of D"처럼 find-절을 여러 개 이어 붙일 수 있다.
각 절은 sysnav.task.query_parser.extract_target()로 (target, relation, reference_objects)를
뽑아내고 -> 문장에 언급된 물체 phrase들만 모아 Gemini에 이미지 그라운딩(박스) + 관계 추론을
같이 요청한다. 텍스트에서 파싱된 관계(instruction text)와 Gemini가 이미지만 보고 스스로
추론한 관계(vision)를 색을 다르게 해서 같은 이미지에 같이 그려 debug 폴더에 저장하므로,
Gemini의 관계 추론이 문장에서 준 관계와 얼마나 일치/다른지 비교해볼 수 있다.

컨테이너 안에서 실행 (GEMINI_API_KEY는 ai_module/.env -> compose env_file로 자동 주입됨):
    docker exec -it iros2026_sysnav_module bash -lc \
        "source /home/docker/ai_module/install/setup.bash && \
         python3 /home/docker/ai_module/src/sysnav_ros2_mvp/tests/test_gemini_works.py"
"""

from __future__ import annotations

import json
import os
import re
import time

import cv2

from sysnav import config
from sysnav.task.query_parser import extract_target

IMAGE_PATH = "/home/docker/ai_module/debug/sysnav_detect_1784607496.591.jpg"
QUESTION = "find the tv on the desk, and find the red chair in front of the table"

_BOX_COLOR_BGR = (0, 255, 0)
_TEXT_EDGE_COLOR_BGR = (255, 255, 0)  # cyan: instruction text에서 파싱된 관계
_VISION_EDGE_COLOR_BGR = (0, 140, 255)  # orange: Gemini가 이미지 보고 추론한 관계

# query_parser._LEADING_COMMAND과 같은 명령 키워드 목록. "and" 뒤에 이 키워드가 오는 지점만
# 새 절의 시작으로 보고 자른다 (예: "between A and B"의 and는 안 잘림).
_CLAUSE_SPLIT = re.compile(
    r",?\s*and\s+(?=(?:please\s+)?(?:find|locate|search for|look for|go to|navigate to|"
    r"take me to|where is|where are|identify|show me)\b)",
    re.IGNORECASE,
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[ymin, xmin, ymax, xmax] normalized to 0-1000 over the full image.",
                    },
                },
                "required": ["id", "label", "attributes", "box_2d"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject_id": {"type": "integer"},
                    "predicate": {"type": "string"},
                    "object_id": {"type": "integer"},
                },
                "required": ["subject_id", "predicate", "object_id"],
            },
        },
    },
    "required": ["objects", "relations"],
}


def split_clauses(question: str) -> list[str]:
    return [part.strip() for part in _CLAUSE_SPLIT.split(question) if part.strip()]


def build_request(parsed_clauses: list[dict]) -> tuple[list[dict], list[dict]]:
    phrase_order: list[str] = []
    attributes_by_phrase: dict[str, list[str]] = {}

    def register(phrase: str, attributes: list[str] | None = None) -> None:
        if not phrase:
            return
        if phrase not in phrase_order:
            phrase_order.append(phrase)
            attributes_by_phrase[phrase] = []
        if attributes:
            attributes_by_phrase[phrase] = attributes

    relations: list[dict] = []
    for clause in parsed_clauses:
        register(clause["target"], clause["attributes"])
        for ref in clause["reference_objects"]:
            register(ref)
        if clause["relation"]:
            for ref in clause["reference_objects"]:
                relations.append({
                    "subject": clause["target"],
                    "predicate": clause["relation"].replace("_", " "),
                    "object": ref,
                })

    phrase_to_id = {phrase: idx for idx, phrase in enumerate(phrase_order)}
    objects_request = [
        {"id": phrase_to_id[phrase], "phrase": phrase, "known_attributes": attributes_by_phrase[phrase]}
        for phrase in phrase_order
    ]
    for rel in relations:
        rel["subject_id"] = phrase_to_id[rel["subject"]]
        rel["object_id"] = phrase_to_id[rel["object"]]
    return objects_request, relations


def load_image_bgr(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def encode_jpeg(image_bgr) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return encoded.tobytes()


def build_prompt(objects_request: list[dict]) -> str:
    lines = "\n".join(
        f'- id={item["id"]}: "{item["phrase"]}"'
        + (f' (mentioned attributes: {item["known_attributes"]})' if item["known_attributes"] else "")
        for item in objects_request
    )
    return f"""
You are grounding a fixed list of object phrases extracted from a robot instruction onto a
single image captured by the robot's camera.

Phrases to locate (use exactly these ids, one object per id):
{lines}

For every id above, find the single best matching object in the image and return its tight
bounding box as "box_2d": [ymin, xmin, ymax, xmax], each value an integer normalized to the
0-1000 range relative to the full image size. Also return "label" (short name of what you see)
and "attributes" (visual attributes you can confirm from the image, e.g. color, material, size).

Then, independently of anything stated in the original instruction, look at the actual image
and infer the spatial relations you observe between these objects (e.g. "on", "on top of",
"over", "under", "next to", "in front of", "behind"). Return them as "relations": a list of
{{subject_id, predicate, object_id}} using only the ids listed above. Only include relations you
can actually see evidence for in the image; it is fine to disagree with how the instruction
phrased it.

Return JSON only, matching the response schema. Return exactly one object per id listed above,
reusing the same id values.
""".strip()


def call_gemini(objects_request: list[dict], image_bytes: bytes) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (check ai_module/.env / container env).")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            build_prompt(objects_request),
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            temperature=config.GEMINI_TEMPERATURE,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )
    if not response.text:
        raise RuntimeError("Empty Gemini response")
    return json.loads(response.text)


def box_2d_to_pixels(box_2d: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = box_2d
    x1 = int(xmin / 1000 * width)
    y1 = int(ymin / 1000 * height)
    x2 = int(xmax / 1000 * width)
    y2 = int(ymax / 1000 * height)
    return x1, y1, x2, y2


def _draw_relations(overlay, centers: dict, relations: list[dict], color, label_prefix: str, offset: int) -> None:
    for rel in relations:
        src = centers.get(rel["subject_id"])
        dst = centers.get(rel["object_id"])
        if src is None or dst is None:
            continue
        cv2.line(overlay, src, dst, color, 2)
        midpoint = ((src[0] + dst[0]) // 2, (src[1] + dst[1]) // 2 + offset)
        cv2.putText(
            overlay, f"{label_prefix}:{rel['predicate']}", midpoint,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def draw_overlay(image_bgr, objects: list[dict], text_relations: list[dict], vision_relations: list[dict]):
    overlay = image_bgr.copy()
    height, width = overlay.shape[:2]
    centers = {}

    for obj in objects:
        x1, y1, x2, y2 = box_2d_to_pixels(obj["box_2d"], width, height)
        centers[obj["id"]] = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _BOX_COLOR_BGR, 2)
        label = f"[{obj['id']}] {obj['label']} {obj['attributes']}"
        cv2.putText(
            overlay, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR_BGR, 1, cv2.LINE_AA,
        )

    _draw_relations(overlay, centers, text_relations, _TEXT_EDGE_COLOR_BGR, "text", offset=-8)
    _draw_relations(overlay, centers, vision_relations, _VISION_EDGE_COLOR_BGR, "vision", offset=12)

    return overlay


def save_overlay(overlay) -> str:
    os.makedirs(config.DEBUG_DIR, exist_ok=True)
    path = os.path.join(config.DEBUG_DIR, f"gemini_graph_debug_{time.time():.3f}.jpg")
    cv2.imwrite(path, overlay)
    return path


def main() -> None:
    print("=== 0. instruction ===")
    print(QUESTION)

    clauses = split_clauses(QUESTION)
    parsed_clauses = [extract_target(clause) for clause in clauses]
    print("\n=== 1. per-clause goal parse (query_parser.extract_target) ===")
    for clause, parsed in zip(clauses, parsed_clauses):
        print(f"  clause: {clause!r}")
        print("    " + json.dumps(parsed, ensure_ascii=False))

    objects_request, text_relations = build_request(parsed_clauses)
    print("\n=== 2. requested objects (from instruction text) ===")
    for item in objects_request:
        print(f"  [{item['id']}] {item['phrase']} (known attributes: {item['known_attributes']})")

    print("\n=== 3. requested relations (from instruction text) ===")
    id_to_phrase = {item["id"]: item["phrase"] for item in objects_request}
    for rel in text_relations:
        print(f"  {id_to_phrase[rel['subject_id']]}({rel['subject_id']}) --{rel['predicate']}--> "
              f"{id_to_phrase[rel['object_id']]}({rel['object_id']})")

    image_bgr = load_image_bgr(IMAGE_PATH)
    result = call_gemini(objects_request, encode_jpeg(image_bgr))

    print("\n=== 4. grounded objects (Gemini box_2d + attributes) ===")
    for obj in result["objects"]:
        print(f"  [{obj['id']}] {obj['label']}: {obj['attributes']} box_2d={obj['box_2d']}")

    vision_relations = result["relations"]
    print("\n=== 5. relations Gemini inferred from the image itself ===")
    for rel in vision_relations:
        print(f"  {id_to_phrase.get(rel['subject_id'], '?')}({rel['subject_id']}) --{rel['predicate']}--> "
              f"{id_to_phrase.get(rel['object_id'], '?')}({rel['object_id']})")

    overlay = draw_overlay(image_bgr, result["objects"], text_relations, vision_relations)
    saved_path = save_overlay(overlay)
    print("\n=== 6. debug overlay saved (cyan=text relation, orange=vision-inferred relation) ===")
    print(f"  {saved_path}")


if __name__ == "__main__":
    main()
