"""Earth Engine initialization with a clear failure message."""
from __future__ import annotations

import ee

REAUTH_HELP = """\
Earth Engine auth failed. Fix (run in your own terminal, needs a browser login):

  /opt/anaconda3/envs/thesis0/bin/earthengine authenticate

Then set the GCP project either in configs/pipeline.yaml (gee.project)
or via the environment variable EE_PROJECT. The project must be registered
for Earth Engine at https://code.earthengine.google.com/register
"""


def init_ee(project: str | None) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # ee raises generic EEException
        raise RuntimeError(f"{REAUTH_HELP}\nOriginal error: {exc}") from exc
