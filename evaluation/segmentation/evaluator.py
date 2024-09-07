from ..build import EVALUATOR_REGISTRY, EvaluatorBase

from collections import defaultdict, OrderedDict

import numpy as np
from medpy.metric import dc, asd, assd, hd95
from sklearn.metrics import f1_score, confusion_matrix, precision_score


@EVALUATOR_REGISTRY.register()
class Segmentation(EvaluatorBase):
    """Evaluator for Segmentation."""

    def __init__(self, cfg, lab2cname, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._per_class_res = None
        self._samples = 0
        
        self._dices = []
        self._asds = []
        self._assds = []
        self._hd95s = []
        self._precisions = []
        
        self._per_class_res = defaultdict(defaultdict(list))
        
        self.need_other_metrics = cfg.TEST.OTHER_METRICS

    def reset(self):
        self._samples = 0
        
        self._dices = []
        self._asds = []
        self._assds = []
        self._hd95s = []
        self._precisions = []

        self._per_class_res = defaultdict(defaultdict(list))

    def process(self, mo, gt, spacing=None):
        # mo (torch.Tensor): model output [1, num_classes, height, width, ...]
        # gt (torch.LongTensor): ground truth [1, height, width, ...]
        
        assert mo.shape[0] == 1, "Batch size must be 1 for model output in segmentation evaluation"
        assert gt.shape[0] == 1, "Batch size must be 1 for ground truth in segmentation evaluation"
        
        pred = mo.max(1)[1].cpu().numpy()
        gt = gt.cpu().numpy()
        
        for label, cname in self._lab2cname.items():
            gt_label = (gt == label).astype(np.float32)
            pred_label = (pred == label).astype(np.float32)
            
            if np.sum(gt_label) > 0:
                dice_val = dc(pred_label, gt_label)
                precision_val = precision_score(gt_label, pred_label)
                
                self._per_class_res[label]["dice"].append(dice_val)
                self._per_class_res[label]["precision"].append(precision_val)
                
                if self.need_other_metrics:
                    asd_val = asd(pred_label, gt_label, spacing)
                    assd_val = assd(pred_label, gt_label, spacing)
                    hd95_val = hd95(pred_label, gt_label, spacing)
                    
                    self._per_class_res[label]["asd"].append(asd_val)
                    self._per_class_res[label]["assd"].append(assd_val)
                    self._per_class_res[label]["hd95"].append(hd95_val)
                
            elif np.sum(gt_label) == 0 and np.sum(pred_label) > 0:
                precision_val = precision_score(gt_label, pred_label)
                
                self._per_class_res[label]["precision"].append(precision_val)
        
        self._samples += 1

    def evaluate(self):
        results = OrderedDict()
        
        _dices = []
        _asds = []
        _assds = []
        _hd95s = []
        _precisions = []
        
        for label, cname in self._lab2cname.items():
            dice_val = np.mean(self._per_class_res[label]["dice"])
            precision_val = np.mean(self._per_class_res[label]["precision"])
            
            _dices.append(dice_val)
            _precisions.append(precision_val)
            
            if self.need_other_metrics:
                asd_val = np.mean(self._per_class_res[label]["asd"])
                assd_val = np.mean(self._per_class_res[label]["assd"])
                hd95_val = np.mean(self._per_class_res[label]["hd95"])
                
                _asds.append(asd_val)
                _assds.append(assd_val)
                _hd95s.append(hd95_val)

        if self.need_other_metrics:
            print(
                "=> result\n"
                f"* total: {self._samples:,}\n"
                f"* dice: {np.mean(_dices):.1f}%\n"
                f"* asd: {np.mean(_asds):.1f}%\n"
                f"* assd: {np.mean(_assds):.1f}%\n"
                f"* hd95: {np.mean(_hd95s):.1f}%\n"
                f"* precision: {np.mean(_precisions):.1f}%\n"
            )
        else:
            print(
                "=> result\n"
                f"* total: {self._samples:,}\n"
                f"* dice: {np.mean(_dices):.1f}%\n"
                f"* precision: {np.mean(_precisions):.1f}%\n"
            )
            
        results["dice"] = np.mean(_dices)
        results["precision"] = np.mean(_precisions)

        # if self._per_class_res is not None:
        #     labels = list(self._per_class_res.keys())
        #     labels.sort()

        #     print("=> per-class result")
        #     accs = []

        #     for label in labels:
        #         classname = self._lab2cname[label]
        #         res = self._per_class_res[label]
        #         correct = sum(res)
        #         total = len(res)
        #         acc = 100.0 * correct / total
        #         accs.append(acc)
        #         print(
        #             f"* class: {label} ({classname})\t"
        #             f"total: {total:,}\t"
        #             f"correct: {correct:,}\t"
        #             f"acc: {acc:.1f}%"
        #         )
        #     mean_acc = np.mean(accs)
        #     print(f"* average: {mean_acc:.1f}%")

        #     results["perclass_accuracy"] = mean_acc

        return results