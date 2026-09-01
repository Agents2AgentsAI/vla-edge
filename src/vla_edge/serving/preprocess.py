"""Threaded drop-in for MolmoAct2ImageProcessor.preprocess.

The upstream preprocess runs `image_to_patches_and_grids` serially per camera
(~40 ms CPU for the 3 YAM frames). The per-image work is independent and the
PIL/numpy heavy lifting releases the GIL, so mapping it over a small thread
pool is numerically identical: same function, same arguments, and same output
order, just concurrent.

Installed per-instance by TrtMolmoActPolicy; the original bound method is kept
for the images=None edge case and for uninstall.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor


def install_threaded_preprocess(processor, max_workers: int = 3):
    ip = processor.image_processor
    mod = sys.modules[type(ip).__module__]
    original = ip.preprocess
    pool = ThreadPoolExecutor(max_workers=max_workers)

    def preprocess(
        images,
        size=None,
        resample=None,
        image_mean=None,
        image_std=None,
        do_convert_rgb=None,
        max_crops=None,
        overlap_margins=None,
        crop_mode=None,
        patch_size=None,
        pooling_size=None,
        return_tensors=None,
        **kwargs,
    ):
        if images is None:
            return original(
                images, size=size, resample=resample, image_mean=image_mean,
                image_std=image_std, do_convert_rgb=do_convert_rgb,
                max_crops=max_crops, overlap_margins=overlap_margins,
                crop_mode=crop_mode, patch_size=patch_size,
                pooling_size=pooling_size, return_tensors=return_tensors,
                **kwargs,
            )
        # kwarg resolution mirrors upstream preprocess() exactly
        if size is not None:
            if "height" not in size or "width" not in size:
                raise ValueError("size must contain 'height' and 'width' keys.")
        else:
            size = {**ip.size}
        base_image_input_size = [size["height"], size["width"]]
        resample = resample or ip.resample
        image_mean = image_mean or ip.image_mean
        image_std = image_std or ip.image_std
        do_convert_rgb = do_convert_rgb or ip.do_convert_rgb
        max_crops = max_crops or ip.max_crops
        overlap_margins = overlap_margins or ip.overlap_margins
        crop_mode = crop_mode or ip.crop_mode
        patch_size = patch_size or ip.patch_size
        pooling_size = pooling_size or ip.pooling_size
        image_pooling_h, image_pooling_w = pooling_size

        images = ip.fetch_images(images)
        images = mod.make_flat_list_of_images(images)
        if not mod.valid_images(images):
            raise ValueError("Invalid image type.")
        if do_convert_rgb:
            images = [mod.convert_to_rgb(image) for image in images]
        images = [mod.to_numpy_array(image) for image in images]

        def work(image):
            return mod.image_to_patches_and_grids(
                image, max_crops, overlap_margins, base_image_input_size,
                resample, image_mean, image_std, patch_size,
                image_pooling_w, image_pooling_h, crop_mode,
            )

        results = list(pool.map(work, images))

        import numpy as np

        data = dict(  # noqa: C408  # keyword form is clearer for a tensor bundle
            pixel_values=np.concatenate([r[1] for r in results], 0),
            image_token_pooling=np.concatenate([r[2] for r in results], 0),
            image_grids=np.concatenate([r[0] for r in results], 0),
            image_num_crops=np.array([r[1].shape[0] for r in results]),
        )
        return mod.BatchFeature(data, tensor_type=return_tensors)

    ip.preprocess = preprocess
    return original
