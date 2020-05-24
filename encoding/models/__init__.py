from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *
from .aanet import *
from .topk_aanet import *
from .pydict import *
from .pydict_encnet import *
from .fcn_8s import *
from .fcn_du import *
from .amca import *
from .aca import *
from .aspp import *
from .cam import *
from .psaa import *
from .new_psp import *
from .aspoc import *
from .new_psp3 import *
from .new_psp3_att import *
from .new_psp3_noatt import *
from .new_psp3_base import *
from .deeplabv3_att import *
from .psp_att import *
from .gsnet import *
from .gsnet2 import *
from .gsnet3 import *
from .gsnet4 import *
from .gsnet5 import *
from .gsnet6 import *
from .gsnet7 import *
from .gsnet8 import *
from .gsnet9 import *
from .gsnet10 import *
from .gsnet11 import *
from .gsnet12 import *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'aanet': get_aanet,
        'topk_aanet': get_topk_aanet,
        'pydict': get_pydict,
        'pydict_encnet': get_pydict_encnet,
        'fcn_8s': get_fcn_8s,
        'fcn_du': get_fcn_du,
        'amca': get_amcanet,
        'aca': get_acanet,
        'aspp': get_asppnet,
        'cam': get_camnet,
        'psaa': get_psaanet,
        'aspoc': get_aspocnet,
        'new_psp': get_new_psp,
        'new_psp3': get_new_psp3net,
        'new_psp3_att': get_new_psp3_attnet,
        'new_psp3_noatt': get_new_psp3_noattnet,
        'new_psp3_base': get_new_psp3_basenet,
        'deeplabv3': get_deeplabv3,
        'deeplabv3_att': get_deeplabv3_att,
        'psp_att': get_psp_att,
        'gsnet': get_gsnet,
        'gsnet2': get_gsnet2,
        'gsnet3': get_gsnet3,
        'gsnet4': get_gsnet4,
        'gsnet5': get_gsnet5,
        'gsnet6': get_gsnet6,
        'gsnet7': get_gsnet7,
        'gsnet8': get_gsnet8,
        'gsnet9': get_gsnet9,
        'gsnet10': get_gsnet10,
        'gsnet11': get_gsnet11,
        'gsnet12': get_gsnet12,
    }
    return models[name.lower()](**kwargs)
