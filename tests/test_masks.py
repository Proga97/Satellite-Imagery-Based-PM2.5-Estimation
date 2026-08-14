"""Client-side patch parsing: valid_fraction and n_images math."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from thesis.gee.patches import parse_pixels, station_grid


def make_structured(h, w, cnt_value):
    arr = np.zeros((h, w), dtype=[("B4", "u2"), ("B3", "u2"), ("B2", "u2"), ("CNT", "u2")])
    arr["B4"] = 500
    arr["B3"] = 400
    arr["B2"] = 300
    arr["CNT"] = cnt_value
    return arr


def test_parse_all_valid():
    arr = make_structured(4, 4, cnt_value=3)
    rgb, vf, n, _ = parse_pixels(arr, ["B4", "B3", "B2"])
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint16
    assert vf == 1.0 and n == 3


def test_parse_half_masked():
    arr = make_structured(2, 2, cnt_value=1)
    arr["CNT"][0, :] = 0
    _, vf, n, _ = parse_pixels(arr, ["B4", "B3", "B2"])
    assert vf == 0.5 and n == 1


def test_parse_no_images():
    arr = make_structured(2, 2, cnt_value=0)
    _, vf, n, _ = parse_pixels(arr, ["B4", "B3", "B2"])
    assert vf == 0.0 and n == 0


def test_station_grid_is_centered_and_snapped():
    g = station_grid(34.05, -118.25, epsg=32611, size_px=224, scale_m=10)
    assert g.translate_x % 10 == 0 and g.translate_y % 10 == 0
    # center of grid should be within one pixel of the station point
    from thesis.gee.patches import to_utm
    x, y = to_utm(34.05, -118.25, 32611)
    cx = g.translate_x + 224 * 10 / 2
    cy = g.translate_y - 224 * 10 / 2
    assert abs(cx - x) <= 10 and abs(cy - y) <= 10


def test_parse_single_scene_date():
    import numpy as np
    from thesis.gee.patches import parse_pixels
    h = w = 2
    dt = np.dtype([("B4", "u2"), ("B3", "u2"), ("B2", "u2"), ("CNT", "u2"), ("DATE", "u2")])
    arr = np.zeros((h, w), dtype=dt)
    for b in ("B4", "B3", "B2"): arr[b] = 1000
    arr["CNT"] = 1
    arr["DATE"] = 19724  # 2024-01-02
    _, vf, n, scene_date = parse_pixels(arr, ["B4", "B3", "B2"])
    assert scene_date == "2024-01-02" and n == 1

    arr["DATE"] = 0
    _, _, n0, sd0 = parse_pixels(arr, ["B4", "B3", "B2"])
    assert n0 == 0 and sd0 == ""
