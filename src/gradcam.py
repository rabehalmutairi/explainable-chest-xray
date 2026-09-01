"""Grad-CAM explanations for the ConvNeXt-Tiny chest X-ray classifier.

Same algorithm verified in notebooks/06_convnext_gradcam.ipynb: backprop from
the raw logit of one selected pathology (not the sigmoid output, which
saturates), weight the target layer's activations by the global-average-
pooled gradient, ReLU, upsample to input resolution.

Target layer is features[7] (10x10x768 at 320x320 input) - chosen over
features[5] by mean IoU on loc_tune in that notebook (0.234 vs 0.171), not
assumed here.

This module knows nothing about HTTP; it takes a loaded classifier and
returns arrays/images. app/main.py wires it to the API.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.inference import ChestXrayClassifier, InvalidImageError, load_image

TARGET_LAYER_NAME = "features[7]"
EXPLAIN_NOTE = (
    "Grad-CAM highlights image regions that influenced this prediction. "
    "It is model explainability, not a confirmed disease location."
)


@dataclass
class Explanation:
    pathology: str
    probability: float
    threshold: float
    above_threshold: bool
    target_layer: str
    overlay_image: Image.Image  # heatmap blended onto the resized input
    cam: np.ndarray  # raw [0,1] heatmap, (H, W)


class GradCAM:
    """Grad-CAM bound to an already-loaded ChestXrayClassifier.

    Kept separate from ChestXrayClassifier (src/inference.py) deliberately:
    that module is the verified classification path and stays untouched.
    Explainability is a distinct concern composed on top of it.
    """

    def __init__(self, classifier: ChestXrayClassifier):
        self.classifier = classifier
        self.model = classifier.model
        self.target_layer = self.model.features[7]

    def _cam_for_tensor(self, input_tensor: torch.Tensor, class_idx: int):
        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def fwd_hook(module, inp, out):
            activations["v"] = out

        def bwd_hook(module, grad_in, grad_out):
            gradients["v"] = grad_out[0]

        h1 = self.target_layer.register_forward_hook(fwd_hook)
        h2 = self.target_layer.register_full_backward_hook(bwd_hook)
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(input_tensor)
            score = logits[0, class_idx]
            score.backward()
        finally:
            h1.remove()
            h2.remove()

        acts = activations["v"][0]     # [C, h, w]
        grads = gradients["v"][0]      # [C, h, w]
        weights = grads.mean(dim=(1, 2))
        cam = F.relu(torch.einsum("c,chw->hw", weights, acts))
        cam = F.interpolate(
            cam[None, None], size=input_tensor.shape[-2:],
            mode="bilinear", align_corners=False,
        ).squeeze().detach().cpu().numpy()

        lo, hi = float(cam.min()), float(cam.max())
        cam = (cam - lo) / (hi - lo) if (hi - lo) > 1e-8 else np.zeros_like(cam)

        logit = float(logits[0, class_idx].item())
        prob = float(torch.sigmoid(logits[0, class_idx]).item())
        return cam, logit, prob

    def explain(
        self,
        image: bytes | str | Image.Image,
        pathology: str,
        alpha: float = 0.45,
    ) -> Explanation:
        if pathology not in self.classifier.labels:
            raise ValueError(f"Unknown pathology: {pathology!r}")

        pil = load_image(image)
        input_tensor = self.classifier.transform(pil).unsqueeze(0).to(self.classifier.device)
        class_idx = self.classifier.labels.index(pathology)

        cam, _logit, prob = self._cam_for_tensor(input_tensor, class_idx)
        overlay = overlay_heatmap(pil, cam, alpha=alpha)

        threshold = float(self.classifier.thresholds[pathology])
        return Explanation(
            pathology=pathology,
            probability=round(prob, 4),
            threshold=threshold,
            above_threshold=prob >= threshold,
            target_layer=TARGET_LAYER_NAME,
            overlay_image=overlay,
            cam=cam,
        )


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0,1] heatmap (jet colormap) onto the image, resized to the CAM's resolution."""
    h, w = cam.shape
    base = np.asarray(pil_image.convert("RGB").resize((w, h))).astype(np.float32) / 255.0
    heat = cm.jet(cam)[:, :, :3]
    blended = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))
