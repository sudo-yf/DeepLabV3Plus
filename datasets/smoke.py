import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data


class SmokeSegmentation(data.Dataset):
    """Smoke segmentation dataset.

    Expected layout:
        root/
          train/image/*.jpg
          train/pt/*.png
          val/image/*.jpg
          val/pt/*.png
          test/image/*.jpg
          test/pt/*.png

    Masks are semantic class-index PNGs:
        0 = background, 1 = smoke
    """

    cmap = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root, split="train", transform=None):
        self.root = Path(os.path.expanduser(root))
        self.split = split
        self.transform = transform

        image_dir = self.root / split / "image"
        mask_dir = self.root / split / "pt"
        if not image_dir.is_dir():
            raise RuntimeError(f"Image directory not found: {image_dir}")
        if not mask_dir.is_dir():
            raise RuntimeError(f"Mask directory not found: {mask_dir}")

        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in self.image_exts)
        masks_by_stem = {p.stem: p for p in mask_dir.glob("*.png")}
        pairs = [(img, masks_by_stem[img.stem]) for img in images if img.stem in masks_by_stem]

        if not pairs:
            raise RuntimeError(f"No image/mask pairs found under split '{split}' in {self.root}")
        if len(pairs) != len(images):
            missing = [img.name for img in images if img.stem not in masks_by_stem][:10]
            raise RuntimeError(f"Missing masks for {len(images) - len(pairs)} images, examples: {missing}")

        self.images = [str(img) for img, _ in pairs]
        self.masks = [str(mask) for _, mask in pairs]

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        return cls.cmap[mask]
