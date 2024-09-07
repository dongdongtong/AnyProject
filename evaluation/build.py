from utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry('EVALUATOR')


def build_evaluator(cfg, lab2cname):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.EVALUATION.NAME, avai_evaluators)
    if cfg.VERBOSE:
        print("Loading evaluator: {}".format(cfg.EVALUATION.NAME))
    return EVALUATOR_REGISTRY.get(cfg.EVALUATION.NAME)(cfg, lab2cname)


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError