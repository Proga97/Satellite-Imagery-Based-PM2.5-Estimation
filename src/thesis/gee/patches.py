"""Build per-(station, week) Sentinel-2 composite requests.

Server side (one computePixels call per patch):
  filter collection to week x patch bounds -> per-image cloud mask
  (SCL for L2A, CloudScore+ for L1C) -> median composite of B4/B3/B2
  + a CNT band (per-pixel count of unmasked observations).

Client side: valid_fraction and n_images are derived from the returned
array (CNT band), so no extra server roundtrips are needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import ee
import numpy as np
from pyproj import Transformer

_transformers: dict[int, Transformer] = {}


def to_utm(lat: float, lon: float, epsg: int) -> tuple[float, float]:
    if epsg not in _transformers:
        _transformers[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    return _transformers[epsg].transform(lon, lat)


@dataclass(frozen=True)
class GridSpec:
    epsg: int
    translate_x: float  # UTM x of patch upper-left corner
    translate_y: float  # UTM y of patch upper-left corner
    size_px: int
    scale_m: float

    def to_request_grid(self) -> dict:
        return {
            "dimensions": {"width": self.size_px, "height": self.size_px},
            "affineTransform": {
                "scaleX": self.scale_m, "shearX": 0, "translateX": self.translate_x,
                "scaleY": -self.scale_m, "shearY": 0, "translateY": self.translate_y,
            },
            "crsCode": f"EPSG:{self.epsg}",
        }


def station_grid(lat: float, lon: float, epsg: int, size_px: int, scale_m: float) -> GridSpec:
    """Patch grid centered on the station, snapped to a 10 m raster grid."""
    x, y = to_utm(lat, lon, epsg)
    half = size_px * scale_m / 2
    ulx = np.floor((x - half) / scale_m) * scale_m
    uly = np.ceil((y + half) / scale_m) * scale_m
    return GridSpec(epsg=epsg, translate_x=float(ulx), translate_y=float(uly),
                    size_px=size_px, scale_m=scale_m)


def _mask_l2a(img: ee.Image, scl_drop: list[int]) -> ee.Image:
    scl = img.select("SCL")
    bad = ee.Image.constant(0)
    for cls in scl_drop:
        bad = bad.Or(scl.eq(cls))
    return img.updateMask(bad.Not())


def _mask_l1c(img: ee.Image, cs_cdf_min: float) -> ee.Image:
    # CloudScore+ image is linked per-granule; cs_cdf in [0,1], higher = clearer
    return img.updateMask(img.select("cs_cdf").gte(cs_cdf_min))


def build_composite(
    lat: float, lon: float, start: str, end: str, cfg_patches: dict, epsg: int,
) -> tuple[ee.Image, GridSpec]:
    """Returns (composite image with bands B4,B3,B2,CNT as uint16, grid spec)."""
    product = cfg_patches["product"]
    bands = list(cfg_patches["bands"])
    grid = station_grid(lat, lon, epsg, cfg_patches["size_px"], cfg_patches["scale_m"])

    # Filter by patch footprint (in WGS84; GEE handles the reprojection)
    half_deg = cfg_patches["size_px"] * cfg_patches["scale_m"] / 2 / 111_000 * 1.5
    region = ee.Geometry.Rectangle([lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg])

    col = (
        ee.ImageCollection(cfg_patches["collections"][product])
        .filterBounds(region)
        .filterDate(start, end)
    )
    if product == "l2a":
        col = col.map(lambda im: _mask_l2a(im, cfg_patches["scl_drop"]))
    else:
        cs = ee.ImageCollection(cfg_patches["cloudscore_collection"])
        col = col.linkCollection(cs, ["cs_cdf"]).map(
            lambda im: _mask_l1c(im, cfg_patches["cs_cdf_min"])
        )

    mode = cfg_patches.get("mode", "median")
    if mode in ("single", "scene"):
        # Least-cloudy single scene by granule metadata. Pixels are saved
        # UNMASKED (smoke must survive); the mask only feeds CNT so
        # valid_fraction still reports how "clear" CloudScore+/SCL thinks
        # the scene is. DATE band = days since epoch of the chosen scene.
        raw = (
            ee.ImageCollection(cfg_patches["collections"][product])
            .filterBounds(region)
            .filterDate(start, end)
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        first_raw = ee.Image(raw.first())
        first_masked = ee.Image(col.sort("CLOUDY_PIXEL_PERCENTAGE").first())
        cnt = first_masked.select(bands[0]).mask().rename("CNT").toUint16()
        days = ee.Number(first_raw.get("system:time_start")).divide(86_400_000).floor()
        date_band = ee.Image.constant(days).rename("DATE").toUint16()
        comp = first_raw.select(bands).toUint16().addBands(cnt).addBands(date_band)
        comp = comp.unmask(0)
        return comp, grid

    cnt = col.select(bands[0]).count().rename("CNT").toUint16()
    comp = col.select(bands).median().toUint16().addBands(cnt)
    comp = comp.unmask(0)
    return comp, grid


def parse_pixels(arr: np.ndarray, bands: list[str]) -> tuple[np.ndarray, float, int, str]:
    """structured array from computePixels -> (H,W,3) uint16, valid_fraction, n_images, scene_date."""
    rgb = np.stack([arr[b] for b in bands], axis=-1).astype(np.uint16)
    cnt = arr["CNT"].astype(np.int32)
    valid_fraction = float((cnt > 0).mean())
    n_images = int(cnt.max())
    scene_date = ""
    if "DATE" in (arr.dtype.names or ()):
        days = int(arr["DATE"].max())
        if days > 0:
            scene_date = (np.datetime64("1970-01-01") + np.timedelta64(days, "D")).astype(str)
        # single scene: n_images is 0/1 mask count; report 1 if a scene existed
        n_images = 1 if days > 0 else 0
    return rgb, valid_fraction, n_images, scene_date
