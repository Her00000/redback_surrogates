from redback_surrogates import afterglowmodels, supernovamodels
from inspect import getmembers, isfunction

try:
    from redback_surrogates import kilonovamodels
except Exception as e:  # pragma: no cover - optional dependency
    kilonovamodels = None
    import warnings
    warnings.warn(f"kilonovamodels unavailable: {e}")
def get_functions_dict(module):
    models_dict = {}
    _functions_list = [o for o in getmembers(module) if isfunction(o[1])]
    _functions_dict = {f[0]: f[1] for f in _functions_list}
    models_dict[module.__name__.split('.')[-1]] = _functions_dict
    return models_dict

modules = [afterglowmodels, supernovamodels]
if kilonovamodels is not None:
    modules.insert(1, kilonovamodels)

all_models_dict = dict()
modules_dict = dict()
for module in modules:
    models_dict = get_functions_dict(module)
    modules_dict.update(models_dict)
    for k, v in models_dict[module.__name__.split('.')[-1]].items():
        all_models_dict[k] = v