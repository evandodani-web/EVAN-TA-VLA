import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TavlaInputs(transforms.DataTransformFn):
    """Inputs for the Tavla policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [14]
    - effort: [history, 14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # If true, the base (cam_high) view is replaced by a zero image and masked out of attention,
    # leaving only the two wrist views. Used for the partial-observability ablation. Applies
    # identically at train time and at serve time, since both build the model input from this
    # transform, so a policy trained with it can never be served without it.
    mask_base_image: bool = False

    def __call__(self, data: dict) -> dict:
        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]

        # Assume that base image always exists.
        base_image = _parse_image(in_images["cam_high"])

        # pi0 always expects all three image slots, so a dropped view is padded with zeros and
        # masked (the same convention aloha_policy/droid_policy use for absent cameras).
        images = {
            "base_0_rgb": np.zeros_like(base_image) if self.mask_base_image else base_image,
        }
        image_masks = {
            "base_0_rgb": np.False_ if self.mask_base_image else np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            images[dest] = _parse_image(in_images[source])
            image_masks[dest] = np.True_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "effort" in data:
            inputs["effort"] = data["effort"]

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class TavlaOutputs(transforms.DataTransformFn):
    """Outputs for the Tavla policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}
