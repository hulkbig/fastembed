import numpy as np
import pytest
from PIL import Image

from fastembed.image.transform.functional import normalize, resize, resize_longest_edge
from fastembed.image.transform.operators import Compose


def test_center_crop_odd_padding_keeps_batch_shape_and_pixels() -> None:
    pixels = np.arange(1, 28, dtype=np.uint8).reshape(3, 3, 3)
    processor = Compose.from_config(
        {
            "do_resize": False,
            "do_center_crop": True,
            "crop_size": 4,
            "do_rescale": False,
        }
    )
    images = [Image.fromarray(pixels), Image.new("RGB", (4, 4))]

    batch = np.array(processor(images))
    expected = np.zeros((3, 4, 4), dtype=np.float32)
    expected[:, 1:, 1:] = pixels.transpose(2, 0, 1)

    assert batch.shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(batch[0], expected)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((100, 200), (200, 100)),  # the bug: a non-square size came back transposed
        ((224, 224), (224, 224)),  # the square path every shipped model takes
    ],
)
def test_resize_tuple_is_height_width(size: tuple[int, int], expected: tuple[int, int]) -> None:
    """A ``(height, width)`` size must reach Pillow as ``(width, height)``."""
    resized = resize(Image.new("RGB", (300, 300)), size=size)

    assert resized.size == expected  # PIL reports (width, height)


def test_resize_int_keeps_shortest_edge_behaviour() -> None:
    """The int branch already emitted Pillow order; it must not be disturbed."""
    landscape = Image.new("RGB", (400, 200))
    portrait = Image.new("RGB", (200, 400))

    # size sets the shortest edge, and the aspect ratio is preserved.
    assert resize(landscape, size=100).size == (200, 100)
    assert resize(portrait, size=100).size == (100, 200)


@pytest.mark.parametrize("longest_edge", [16, 2048])
@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((2, 1), (1, 0.5)),
        ((1, 2), (0.5, 1)),
    ],
)
def test_longest_edge_resize_keeps_ordinary_dimensions(
    longest_edge: int, size: tuple[int, int], expected: tuple[float, float]
) -> None:
    image = Image.new("RGB", (size[0] * longest_edge, size[1] * longest_edge))
    resized = resize_longest_edge(image, longest_edge)
    assert resized.size == (int(expected[0] * longest_edge), int(expected[1] * longest_edge))


@pytest.mark.parametrize("longest_edge", [16, 2048])
@pytest.mark.parametrize("portrait", [False, True])
def test_longest_edge_resize_clamps_thin_images_after_even_rounding(
    longest_edge: int, portrait: bool
) -> None:
    for width, expected_height in [(longest_edge, 2), (2 * longest_edge, 1)]:
        size = (1, width) if portrait else (width, 1)
        expected = (expected_height, longest_edge) if portrait else (longest_edge, expected_height)
        resized = resize_longest_edge(Image.new("RGB", size, (32, 64, 128)), longest_edge)
        assert resized.size == expected
        assert resized.getpixel((0, 0)) == (32, 64, 128)


@pytest.mark.parametrize(("longest_edge", "patch_size"), [(16, 4), (2048, 512)])
@pytest.mark.parametrize("case", ["ordinary", "thin_boundary", "landscape", "portrait", "mixed"])
def test_idefics3_compose_handles_thin_images(
    longest_edge: int, patch_size: int, case: str
) -> None:
    # The 2048/512 case uses Qdrant/colmodernvbert's shipped processor configuration:
    # https://huggingface.co/Qdrant/colmodernvbert/blob/main/preprocessor_config.json
    processor = Compose.from_config(
        {
            "do_convert_rgb": True,
            "do_image_splitting": True,
            "do_normalize": True,
            "do_pad": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "Idefics3ImageProcessor",
            "image_std": [0.5, 0.5, 0.5],
            "max_image_size": {"longest_edge": patch_size},
            "processor_class": "ColModernVBertProcessor",
            "resample": 1,
            "rescale_factor": 0.00392156862745098,
            "size": {"longest_edge": longest_edge},
        }
    )
    sizes = {
        "ordinary": [(32, 16), (16, 32)],
        "thin_boundary": [(longest_edge, 1), (1, longest_edge)],
        "landscape": [(2 * longest_edge, 1)],
        "portrait": [(1, 2 * longest_edge)],
        "mixed": [(32, 16), (2 * longest_edge, 1), (1, 2 * longest_edge)],
    }[case]
    colors = [(32 + i * 16, 64 + i * 8, 128 - i * 16) for i in range(len(sizes))]
    output = processor([Image.new("RGB", size, color) for size, color in zip(sizes, colors)])

    assert len(output) == len(sizes)
    for i, (patches, color) in enumerate(zip(output, colors)):
        expected_count = 9 if case == "ordinary" or (case == "mixed" and i == 0) else 5
        assert len(patches) == expected_count
        expected_pixel = (np.array(color, dtype=np.float32) / 255 - 0.5) / 0.5
        for patch in patches:
            assert patch.shape == (3, patch_size, patch_size)
            assert patch.dtype == np.float32
            np.testing.assert_allclose(patch[:, 0, 0], expected_pixel, atol=1e-7)


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        ([0.1, 0.2, 0.3], [0.5, 0.6, 0.7]),  # per-channel, as every model config gives it
        (0.5, 0.25),  # scalar, expanded to one value per channel
    ],
)
def test_normalize_chw_is_channel_wise(
    mean: list[float] | float, std: list[float] | float
) -> None:
    """Each channel must be normalized by its own mean/std, not by any other axis."""
    rng = np.random.default_rng(0)
    image = rng.random((3, 5, 7)).astype(np.float32)
    means = mean if isinstance(mean, list) else [mean] * 3
    stds = std if isinstance(std, list) else [std] * 3

    result = normalize(image, mean=mean, std=std)

    for c in range(3):
        assert np.allclose(result[c], (image[c] - means[c]) / stds[c], atol=1e-6)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_normalize_batched_matches_per_image(batch_size: int) -> None:
    """A batch must give exactly what the (C, H, W) path gives image by image.

    batch_size 2 used to raise, since transposing reversed every axis; batch_size 3
    matched the channel count and silently normalized along the batch axis instead.
    """
    rng = np.random.default_rng(2)
    batch = rng.random((batch_size, 3, 4, 4)).astype(np.float32)
    mean, std = [0.1, 0.2, 0.3], [0.5, 0.6, 0.7]

    result = normalize(batch, mean=mean, std=std)

    per_image = np.stack([normalize(image, mean=mean, std=std) for image in batch])
    assert result.shape == batch.shape
    assert np.array_equal(result, per_image)


def test_normalize_rejects_input_without_a_channel_axis() -> None:
    """Every pipeline runs ConvertToRGB first, so normalize only ever sees (C, H, W)."""
    with pytest.raises(ValueError, match=r"must be \(C, H, W\)"):
        normalize(np.zeros((4, 6), dtype=np.float32), mean=0.5, std=0.25)


@pytest.mark.parametrize(
    ("mean", "std", "expected"),
    [
        ([0.1, 0.2], [1.0, 1.0, 1.0], "mean must"),
        ([0.1, 0.2, 0.3], [1.0, 1.0], "std must"),
    ],
)
def test_normalize_channel_count_mismatch_raises(
    mean: list[float], std: list[float], expected: str
) -> None:
    image = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match=expected):
        normalize(image, mean=mean, std=std)
