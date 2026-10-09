"""Select the versioned, project-local LLaVA runtime explicitly."""

import sys
from pathlib import Path


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "llava"


def activate_vendor():
    """Fail instead of mixing this project's runtime with another llava install."""
    existing = sys.modules.get("llava")
    if existing is not None:
        origin = Path(existing.__file__).resolve()
        if VENDOR not in origin.parents:
            raise RuntimeError(
                f"Another llava runtime is already imported from {origin}. "
                "Run mask_fusion in a fresh Python process."
            )
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return VENDOR
