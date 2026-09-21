"""ABC native token layout, used by the Thor runtime."""

CAMERAS = 3
VISION_TOKENS_PER_VIEW = 64
VISION_TOKENS = CAMERAS * VISION_TOKENS_PER_VIEW
IMAGE_BLOCK = 2 + VISION_TOKENS_PER_VIEW + 2
IMAGE_SPANS = tuple(
    (1 + i * IMAGE_BLOCK + 2, 1 + i * IMAGE_BLOCK + 2 + VISION_TOKENS_PER_VIEW)
    for i in range(CAMERAS)
)
TEXT_START = 1 + CAMERAS * IMAGE_BLOCK


def native_positions(length: int) -> tuple[list[int], list[int]]:
    if length <= TEXT_START:
        raise ValueError("prefix is shorter than the three image blocks")
    image = [p for start, stop in IMAGE_SPANS for p in range(start, stop)]
    image_set = set(image)
    return image, [p for p in range(length) if p not in image_set]
