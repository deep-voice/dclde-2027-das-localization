"""deep-voice localization core, imported from deep-voice/dclde_2027.

Straight-ray forward model, misfit fields, multi-basin solver, cable geometry
in a local metric frame, GMRT bathymetry, the picks snapshot loader, and the
sound-speed profile / ConstantC / PathAverage models that solve() takes.

Lives beside (not inside) Tomer's geometry.py / picks.py / localize.py so both
implementations can be compared before they are merged.
"""
