"""Shared H5 storage helpers for Piper recordings.

Raw recordings historically stored every camera frame as a dense CHW array
with gzip compression.  That format is kept readable, while new recordings
can store RGB frames as JPEG byte arrays.  JPEG is substantially smaller for
camera observations than lossless gzip and remains easy to decode from H5.
"""

from __future__ import annotations

from typing import Any, Optional

import h5py
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the recorder already depends on cv2
    cv2 = None


IMAGE_CODECS = ("jpeg", "gzip", "none")
DEFAULT_COMPRESSION = "gzip"
DEFAULT_GZIP_LEVEL = 4
DEFAULT_JPEG_QUALITY = 85


def _attr_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _attr_text(value.item())
    return str(value)


def is_rgb_path(name: str) -> bool:
    """Return whether an H5 object path belongs to obs/rgb."""

    return "rgb" in {part for part in str(name).split("/") if part}


def dataset_codec(dataset: h5py.Dataset) -> str:
    return _attr_text(dataset.attrs.get("codec", "")).lower()


def _dense_dataset_kwargs(
    data: np.ndarray,
    compression: str = DEFAULT_COMPRESSION,
) -> dict:
    if data.ndim == 0 or data.size == 0 or compression == "none":
        return {}
    if compression != "gzip":
        raise ValueError(f"Unsupported H5 compression: {compression}")
    # shuffle is lossless and improves gzip on float/bool arrays.  It is also
    # useful for depth arrays, while RGB JPEG datasets do not use this path.
    return {
        "compression": "gzip",
        "compression_opts": DEFAULT_GZIP_LEVEL,
        "shuffle": True,
    }


def _to_hwc(frame: np.ndarray) -> tuple[np.ndarray, str]:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        return np.transpose(frame, (1, 2, 0)), "CHW"
    if frame.ndim == 3 and frame.shape[-1] in (1, 3):
        return frame, "HWC"
    if frame.ndim == 2:
        return frame, "HW"
    raise ValueError(f"JPEG expects a 2D image or CHW/HWC frame, got {frame.shape}")


def _encode_jpeg(frame: np.ndarray, quality: int) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python is required for H5 JPEG image storage.")

    hwc, _ = _to_hwc(frame)
    if hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)

    encode_input = hwc
    if hwc.ndim == 3 and hwc.shape[-1] == 3:
        # H5 consumers expose RGB, while OpenCV's JPEG API expects BGR.
        encode_input = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        encode_input,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise ValueError("OpenCV failed to encode an RGB frame as JPEG.")
    return np.asarray(encoded, dtype=np.uint8)


def _decode_jpeg(payload: Any, original_shape: tuple[int, ...]) -> np.ndarray:
    if cv2 is None:
        raise ImportError("opencv-python is required to read H5 JPEG image storage.")

    encoded = np.asarray(payload, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError("OpenCV failed to decode a JPEG frame from H5.")

    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if len(original_shape) == 3 and original_shape[0] in (1, 3):
        if frame.ndim == 2:
            frame = frame[None, ...]
        else:
            frame = np.transpose(frame, (2, 0, 1))
    return np.asarray(frame, dtype=np.uint8).reshape(original_shape)


def _create_jpeg_dataset(
    group: h5py.Group,
    key: str,
    data: np.ndarray,
    jpeg_quality: int,
) -> h5py.Dataset:
    if data.ndim < 3 or data.shape[0] == 0:
        # There is no useful variable-length representation for an empty
        # episode; retain the ordinary dense representation in that case.
        return group.create_dataset(key, data=data, **_dense_dataset_kwargs(data))

    frame_shape = tuple(int(x) for x in data.shape[1:])
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    dataset = group.create_dataset(key, shape=(data.shape[0],), dtype=dtype)
    # Encode one frame at a time to avoid holding a second full episode-sized
    # list of JPEG payloads in memory.
    for idx in range(data.shape[0]):
        dataset[idx] = _encode_jpeg(data[idx], jpeg_quality)

    dataset.attrs["codec"] = "jpeg"
    dataset.attrs["jpeg_quality"] = int(jpeg_quality)
    dataset.attrs["original_shape"] = np.asarray(frame_shape, dtype=np.int64)
    dataset.attrs["original_dtype"] = str(data.dtype)
    dataset.attrs["layout"] = "CHW"
    dataset.attrs["color_space"] = "RGB"
    return dataset


def create_h5_dataset(
    group: h5py.Group,
    key: str,
    data: Any,
    *,
    rgb: bool = False,
    image_codec: str = "gzip",
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> h5py.Dataset:
    """Create a compressed dense or encoded H5 dataset."""

    codec = str(image_codec).lower()
    if codec not in IMAGE_CODECS:
        raise ValueError(f"image_codec must be one of: {', '.join(IMAGE_CODECS)}")
    array = np.asarray(data)
    if rgb and codec == "jpeg":
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        return _create_jpeg_dataset(group, key, array, int(jpeg_quality))

    # Depth and all non-image signals remain lossless.  `none` is accepted as
    # an explicit escape hatch for benchmarking or legacy interoperability.
    dense_compression = "none" if (rgb and codec == "none") else DEFAULT_COMPRESSION
    return group.create_dataset(
        key,
        data=array,
        **_dense_dataset_kwargs(array, dense_compression),
    )


def read_h5_dataset(dataset: h5py.Dataset, index: Optional[int] = None) -> np.ndarray:
    """Read both legacy dense datasets and JPEG-backed datasets as ndarrays."""

    if dataset_codec(dataset) != "jpeg":
        return dataset[()] if index is None else dataset[index]

    shape_attr = dataset.attrs.get("original_shape")
    if shape_attr is None:
        raise ValueError(f"JPEG dataset {dataset.name} has no original_shape metadata.")
    original_shape = tuple(int(x) for x in np.asarray(shape_attr).reshape(-1))

    if index is not None:
        return _decode_jpeg(dataset[index], original_shape)

    frames = [_decode_jpeg(dataset[idx], original_shape) for idx in range(dataset.shape[0])]
    return np.stack(frames, axis=0) if frames else np.empty((0,) + original_shape, dtype=np.uint8)


def copy_h5_dataset(
    src_dataset: h5py.Dataset,
    dst_group: h5py.Group,
    key: str,
    *,
    image_codec: str = "gzip",
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> h5py.Dataset:
    """Copy a dataset, optionally converting RGB arrays to JPEG storage."""

    if is_rgb_path(src_dataset.name) and str(image_codec).lower() == "jpeg":
        if dataset_codec(src_dataset) == "jpeg":
            # Preserve already encoded frames without a decode/re-encode cycle.
            dtype = h5py.vlen_dtype(np.dtype("uint8"))
            dst_dataset = dst_group.create_dataset(
                key,
                shape=(src_dataset.shape[0],),
                dtype=dtype,
            )
            for idx in range(src_dataset.shape[0]):
                dst_dataset[idx] = src_dataset[idx]
            for attr_key, attr_value in src_dataset.attrs.items():
                dst_dataset.attrs[attr_key] = attr_value
            return dst_dataset
        data = read_h5_dataset(src_dataset)
        return create_h5_dataset(
            dst_group,
            key,
            data,
            rgb=True,
            image_codec="jpeg",
            jpeg_quality=jpeg_quality,
        )

    data = read_h5_dataset(src_dataset)
    return create_h5_dataset(
        dst_group,
        key,
        data,
        rgb=is_rgb_path(src_dataset.name),
        image_codec=image_codec,
        jpeg_quality=jpeg_quality,
    )
