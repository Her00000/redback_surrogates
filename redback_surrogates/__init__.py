"""redback_surrogates package"""

from redback_surrogates import utils, afterglowmodels, model_library, supernovamodels, data_management

try:
    from redback_surrogates import kilonovamodels
except Exception as e:  # pragma: no cover - optional dependency
    import warnings
    warnings.warn(f"kilonovamodels unavailable: {e}")
