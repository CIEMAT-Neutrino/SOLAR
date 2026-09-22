"""Study-queue helper: with SOLAR_SKIP_IMAGE_EXPORT=1, plotly static image export (kaleido -> headless Chrome) is a no-op.
Kaleido hangs under heavy load and never returns; the pipelines call write_image only for PNGs, never for data."""
import os

if os.environ.get("SOLAR_SKIP_IMAGE_EXPORT") == "1":
    try:
        import plotly.io as _pio
        import plotly.basedatatypes as _bdt

        _pio.write_image = lambda *a, **k: None
        _bdt.BaseFigure.write_image = lambda self, *a, **k: None
    except Exception:
        pass
