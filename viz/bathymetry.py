"""Re-export of the bathymetry module, which now lives in the task package.

It moved to `dclde_localization.physics.bathymetry` because the sound-speed work needs
water depth along a propagation path, and `dclde_localization.physics` must not import
the visualizer package to get it. One definition, imported both ways.
"""

from dclde_localization.physics.bathymetry import (  # noqa: F401
    GMRT_BOUNDS,
    GMRT_URL,
    NO_DATA,
    GeoGrid,
    LocalGrid,
    ValidationReport,
    _sample,
    decode_gmt_grid,
    fetch_gmrt,
    reproject_to_local,
    validate_against_cable,
)

__all__ = ["GMRT_BOUNDS", "GMRT_URL", "NO_DATA", "GeoGrid", "LocalGrid",
           "ValidationReport", "decode_gmt_grid", "fetch_gmrt",
           "reproject_to_local", "validate_against_cable"]
