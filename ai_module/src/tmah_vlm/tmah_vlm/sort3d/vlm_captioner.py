#!/usr/bin/env python3
"""Optional VLM crop captioner for SORT3D-style object attributes."""

from PIL import Image
import torch


class FlorenceCaptioner:
    """Lazy Florence-2 captioner.

    The model is intentionally optional. If loading or inference fails, callers
    should keep the existing rule-based caption.
    """

    def __init__(self, model_id="microsoft/Florence-2-base", device="cpu", max_new_tokens=64):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = int(max_new_tokens)

        patch_florence_compat(model_id)

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)
        self.model.eval()

    def caption_crop(self, image, box, margin_px=16):
        crop = crop_box(image, box, margin_px)
        return self.caption_image(crop)

    def caption_image(self, image):
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        try:
            parsed = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height),
            )
            caption = parsed.get(prompt, generated_text)
        except Exception:
            caption = generated_text

        return clean_caption(caption)


def crop_box(image, box, margin_px=16):
    img = image.convert("RGB")
    x1, y1, x2, y2 = [float(v) for v in box]
    margin = float(margin_px)
    left = max(0, int(x1 - margin))
    top = max(0, int(y1 - margin))
    right = min(img.width, int(x2 + margin))
    bottom = min(img.height, int(y2 + margin))
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


def clean_caption(text):
    text = str(text or "").replace("</s>", " ").replace("<s>", " ")
    text = text.replace("<MORE_DETAILED_CAPTION>", " ")
    text = " ".join(text.split())
    if text and text[-1] not in ".!?":
        text += "."
    return text


def patch_florence_compat(model_id):
    patch_florence_tokenizer()
    patch_florence_config(model_id)


def patch_florence_tokenizer():
    """Expose `additional_special_tokens` for Florence remote processor."""
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except Exception:
        return

    if hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
        return

    def additional_special_tokens(self):
        token_map = getattr(self, "special_tokens_map_extended", None)
        if token_map is None:
            token_map = getattr(self, "special_tokens_map", {})
        return list(token_map.get("additional_special_tokens", []))

    PreTrainedTokenizerBase.additional_special_tokens = property(additional_special_tokens)


def patch_florence_config(model_id):
    """Work around Florence-2 remote config on some transformers versions.

    The cached remote code reads `forced_bos_token_id` before the base
    PretrainedConfig has created it. Injecting the default before the original
    __init__ runs keeps AutoConfig/AutoModel loading compatible without editing
    HuggingFace cache files.
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls = get_class_from_dynamic_module(
            "configuration_florence2.Florence2LanguageConfig",
            model_id,
        )
    except Exception:
        return

    if getattr(cls, "_tmah_forced_bos_patch", False):
        return

    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        self.forced_bos_token_id = kwargs.get("forced_bos_token_id", None)
        return original_init(self, *args, **kwargs)

    cls.__init__ = patched_init
    cls._tmah_forced_bos_patch = True
