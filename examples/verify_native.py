from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-root', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.master_root.resolve()
    cfg_file = root / 'ultralytics/cfg/models/master/v0_10/yolo-master.yaml'
    if not cfg_file.is_file():
        parser.error('audited Master v0.10 YAML not found')
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT, YAML
    from pear_innovations.master_adapter import PearMasterAdapter, PearConfig
    from pear_innovations.training import master_training_loss
    import ultralytics
    if root not in Path(ultralytics.__file__).resolve().parents:
        raise RuntimeError('imported a different Ultralytics installation')
    torch.manual_seed(29)
    torch.set_num_threads(2)
    cfg = YAML.load(str(cfg_file))
    cfg['scale'] = 'n'
    native = DetectionModel(cfg, ch=3, nc=1, verbose=False)
    native.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model = PearMasterAdapter(native, PearConfig(score_threshold=0.0, max_detections=5, pre_nms_topk=64)).to(args.device).train()
    criterion = model.master.init_criterion()
    batch = {'img': torch.randn(2, 3, 128, 128, device=args.device), 'batch_idx': torch.tensor([0, 1], device=args.device), 'cls': torch.tensor([[1.0], [2.0]], device=args.device), 'bboxes': torch.tensor([[0.5, 0.5, 0.6, 0.7], [0.55, 0.55, 0.5, 0.6]], device=args.device), 'evidence': torch.zeros(2, 3, 16, 16, device=args.device), 'evidence_valid': torch.ones(2, 3, dtype=torch.bool, device=args.device)}
    batch['evidence'][:, 0] = 1
    batch['evidence'][:, 1, 3:8, 2:5] = 1
    batch['evidence'][:, 2, 4:6, 2:4] = 1
    batch['shallow_targets'] = torch.zeros(2, 2, 128, 128, device=args.device)
    batch['shallow_targets'][:, 0] = 1
    batch['shallow_targets'][:, 1, 32:64, 16:40] = 1
    batch['shallow_valid'] = torch.ones_like(batch['shallow_targets'])
    result = master_training_loss(model, batch, criterion)
    result['loss'].backward()
    if not torch.isfinite(result['loss']):
        raise RuntimeError('nonfinite native synthetic loss')
    gradient_norms = {}
    for name, module in [('LECF', model.fusion), ('CSEG', model.grade_head), ('RCPM', model.downsample), ('native_Detect', model.detection_head)]:
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or not all((torch.isfinite(g).all() for g in gradients)):
            raise RuntimeError(f'{name} missing/nonfinite gradient')
        gradient_norms[name] = float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())
        if gradient_norms[name] <= 0:
            raise RuntimeError(f'{name} has zero total gradient')
    predictions = model.eval().predict(batch['img'])
    if not all((len(p['boxes']) > 0 for p in predictions)):
        raise RuntimeError('nonempty inference branch was not exercised')
    report = {'scope': 'synthetic native-interface forward/backward and inference correctness only; no optimizer update or dataset experiment', 'native_source': root.name, 'device': args.device, 'torch': torch.__version__, 'loss': float(result['loss'].detach()), 'graded_instances': result['graded_instances'], 'fallback_instances': int(result['selection']['fallback'].sum()), 'response_pairs': result['response_details']['pairs'], 'response_loss': float(result['response_loss'].detach()), 'detections_per_image': [len(p['boxes']) for p in predictions], 'gradient_norms': gradient_norms, 'inference_test_only': {'score_threshold': 0.0, 'max_detections': 5, 'not_accuracy_evaluation': True}, 'grade_probability_shapes': [list(p['grade_probabilities'].shape) for p in predictions], 'loss_components': {k: float(result[k].detach()) for k in ('grade_loss', 'retention_loss', 'evidence_loss', 'shallow_loss', 'fusion_support_loss')}}
    body = json.dumps(report, ensure_ascii=False, indent=2)
    print(body)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding='utf-8')
if __name__ == '__main__':
    main()
