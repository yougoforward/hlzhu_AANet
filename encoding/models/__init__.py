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
from .aanet_nopam import *
from .aanet_pam_metric import *
from .amca_asppaca import *
from .amca_gap_aca import *
from .aspoc_secam import *
from .asp_oc_gap_secam import *
from .pgfnet import *
from .asp_pgfnet import *
from .asppacaca import *
from .aspoc_gsecam_net import *
from .aspoc_gsecam_du_net import *
from .fcn_8s import *
from .fcn_du import *
from .aanet_simple import *
from .amca import *
from .amca2 import *
from .aca import *
from .aca2 import *
from .aspp import *
from .aspp2 import *
from .cam import *
from .cam2 import *
from .asppcam import *
from .asppaca import *
from .amcacam import *
from .amcaaca import *
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
        'aanet_nopam': get_aanet_nopam,
        'aanet_metric': get_aanet_metric,
        'amca_asppaca': get_amca_aspp_acanet,
        'aspoc_secam': get_aspoc_secamnet,
        'asp_oc_gap_secam': get_asp_oc_gap_secamnet,
        'pgfnet': get_pgfnet,
        'asp_pgfnet': get_asp_pgfnet,
        'asppacaca': get_asppacaca,
        'aspoc_gsecam': get_aspoc_secamnet,
        'aspoc_gsecam_du': get_aspoc_gsecam_dunet,
        'fcn_8s': get_fcn_8s,
        'fcn_du': get_fcn_du,

        'aanet_simple': get_aanet_fast,
        'amca': get_amcanet,
        'amca2': get_amca2net,
        'aca': get_acanet,
        'aca2': get_aca2net,
        'aspp': get_asppnet,
        'aspp2': get_aspp2net,
        'cam': get_camnet,
        'cam2': get_cam2net,
        'asppcam': get_asppcamnet,
        'asppaca': get_asppacanet,
        'amcaaca': get_amcacamnet,
        'amcacam': get_amcacamnet,

    }
    return models[name.lower()](**kwargs)
