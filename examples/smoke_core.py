import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pear_innovations import RCPM, LECF, CSEGGradeHead, GSFIoULoss


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    phase = RCPM(8, 16)
    fine = phase(torch.randn(2, 8, 16, 16))
    fusion = LECF(16, 8, 24)
    features = fusion(fine, torch.randn(2, 8, 4, 4))
    head = CSEGGradeHead(24, feature_dim=16)
    result = head(features)
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]], requires_grad=True)
    evidence = torch.zeros(2, 3, 8, 8)
    evidence[:, 0] = 1.0
    evidence[:, 1, 2:6, 2:6] = 1.0
    evidence[:, 2, 3:5, 3:5] = 1.0
    localization = GSFIoULoss()(boxes, boxes.detach(), evidence)
    loss = -result['grade_logits'][:, 1].mean() + localization
    loss.backward()
    torch.testing.assert_close(result['grade_probs'].sum(-1), torch.ones(2))
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite core smoke-test loss')
    print(json.dumps({'scope': 'synthetic core-module forward and backward', 'feature_shape': list(features.shape), 'grade_probability_shape': list(result['grade_probs'].shape), 'loss': float(loss.detach()), 'torch': torch.__version__}, indent=2))


if __name__ == '__main__':
    main()
