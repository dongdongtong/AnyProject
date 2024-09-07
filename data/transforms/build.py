from utils.register import Registry
from utils.tools import check_availability

TRANSFORM_REGISTRY = Registry("TRANSFORM")


def build_transform(cfg, is_train=False, only_other_transforms=False):
    avai_transforms = TRANSFORM_REGISTRY.registered_names()
    check_availability(cfg.INPUT.TRANSFORM, avai_transforms)
    if cfg.VERBOSE:
        print("Loading transforms: {}".format(cfg.INPUT.TRANSFORM))
    return TRANSFORM_REGISTRY.get(cfg.INPUT.TRANSFORM)(cfg, is_train, only_other_transforms)