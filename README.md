# Pear_OTDE

PyTorch modules for evidence-preserving on-tree detection and appearance grading of Korla fragrant pears.

| Component | Public API | Role |
| --- | --- | --- |
| Reference-Conditioned Phase Moments | `RCPM` | Local reference-based downsampling |
| Local Energy-Constrained Fusion | `LECF` | Constrained cross-scale feature fusion |
| Critical Support and Extent Grading | `CSEGGradeHead` | Query-dependent appearance-grade probabilities |
| Grading-Statistic Fidelity IoU Loss | `GSFIoULoss` | Evidence-preserving box regression |
| Detector integration | `PearMasterAdapter`, `PearConfig`, `master_training_loss` | Explicit YOLO-Master training and inference interfaces |

## Installation

Python 3.10 or later and PyTorch 2.2 or later are required. The release was checked with Python 3.13.9 and PyTorch 2.10.0 on CPU. Install a PyTorch build appropriate for your device before installing the package.

```shell
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then run:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python examples/smoke_core.py
python run_validation.py --output docs/validation_results.json
```

The import namespace remains `pear_innovations`:

```python
from pear_innovations import RCPM, LECF, CSEGGradeHead, GSFIoULoss
```

## Training and inference

The detector adapter expects an externally supplied, single-fruit-class YOLO-Master v0.10 model with the inspected 25-node graph. Install that project's dependencies in the same environment. The adapter checks the graph and native criterion interfaces before use.

```python
import torch
from pear_innovations import PearConfig, PearMasterAdapter, master_training_loss

model = PearMasterAdapter(native_model, PearConfig()).to(device).train()
criterion = model.master.init_criterion()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)

for batch in training_loader:
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    output = master_training_loss(model, batch, criterion)
    output['loss'].backward()
    optimizer.step()

model.eval()
predictions = model.predict(images.to(device))
```

`native_model`, `training_loader`, `images`, and `device` are supplied by the application. This minimal API example performs one optimizer update per batch; the manuscript's accumulation, augmentation, scheduling, and model-selection protocol is specified separately in [docs/INTEGRATION.md](docs/INTEGRATION.md).

Outputs contain `boxes`, `fruit_scores`, `grade_probabilities`, `grade_scores`, and `grades`, where A/B/C are encoded as 0/1/2. Boxes are in the preprocessed input frame. Use the stored preprocessing transform to restore original image coordinates before dataset evaluation.

## Included materials

This release contains the four research components, detector adapters, control implementations, unit tests, and synthetic validation examples. The complete YOLO-Master source, KFP-OTDG images/annotations, trained checkpoints, and manuscript evaluation pipelines are distributed separately and are not included in this archive.

- [Data format and supervision](docs/DATA_FORMAT.md)
- [Native integration, training, and evaluation](docs/INTEGRATION.md)
- [Manuscript correspondence and implementation scope](docs/MANUSCRIPT_CORRESPONDENCE.md)
- [Release validation and changes](docs/VALIDATION.md)
- [Machine-readable validation results](docs/validation_results.json)

## Citation and license

The accompanying manuscript, *Preserving grading evidence for on-tree detection and appearance grading of Korla fragrant pears*, is unpublished. Citation metadata is provided in [CITATION.cff](CITATION.cff).

The self-developed modules are licensed under [MIT](LICENSE). External detector code and dependencies retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
