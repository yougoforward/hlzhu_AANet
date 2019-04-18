from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *
from .aanet import *
from .aanet_ml import *
from .dict_aanet import *
from .topk_aanet import *
from .pydict import *
from .pydict_encnet import *
from .aanet_encnet import *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,
        'aanet': get_aanet,
        'aanet_ml': get_aanet_ml,
        'dict_aanet': get_dict_aanet,
        'topk_aanet': get_topk_aanet,
        'pydict': get_pydict,
        'pydict_encnet': get_pydict_encnet,
        'aanet_encnet': get_aanet_encnet,
    }
    return models[name.lower()](**kwargs)
