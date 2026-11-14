# Data format

`master_training_loss(model, batch, criterion)` accepts a dictionary with the following tensor fields. Images, boxes, and annotations must undergo the same geometric transformation.

| Field | Shape and type | Meaning |
| --- | --- | --- |
| `img` | `[B,3,H,W]`, floating point | Preprocessed RGB input, normally in [0,1]; H and W are multiples of 32 |
| `batch_idx` | `[M]` or `[M,1]`, integer-valued | Image index for each original fruit instance |
| `cls` | `[M,1]`, integer-valued 0/1/2 | Original A/B/C appearance grade |
| `grades` | `[M]`, optional | Explicit grading labels, overriding `cls` for grading |
| `bboxes` | `[M,4]`, floating point | Normalized center-x, center-y, width, height in the preprocessed image frame |
| `evidence` | `[M,3,He,We]`, optional | Joint V/D/S support maps in each GT box's local frame |
| `evidence_valid` | `[M,3]`, Boolean | Complete-channel annotation availability; required when `evidence` is supplied |
| `shallow_targets` | `[B,2,Hs,Ws]`, optional | Full-image visible support and conditional damage targets |
| `shallow_valid` | Same shape as `shallow_targets` | Valid annotation domains for the shallow targets |

`cls` remains required when `grades` is provided. All M-instance fields follow the same original order. The bridge stably groups detection targets by image and maps native TAL assignments back to that order. It uses a single fruit detection class internally while retaining the A/B/C labels for grading.

V denotes the target fruit's entire visible surface, D its joint damage support, and S its joint critical support. Known maps lie in [0,1] and satisfy S <= D <= V wherever the corresponding channels are available. If D is missing but V and S are known, S <= V is still required. A completely annotated empty region is an all-zero map with availability true. An unavailable channel has availability false; its placeholder values are excluded from the relevant supervision. Confirmed-empty and unknown regions are different annotation states.

The context target builder reprojects joint maps before forming the conditional damage and critical targets. Box/class labels alone do not supply V/D/S. With no regional annotations, grade supervision falls back to GT ROIs, while the unavailable regional, statistical, and response terms are zero.

Shallow scene targets must be supplied by the data pipeline: project local annotations into the image, retain uniquely owned GT regions and explicitly annotated background, exclude overlapping/unknown domains, and form conditional damage D/V. The loss resizes these targets and domains by nearest-neighbor interpolation and weights damage supervision by GT visible support. No scene-target builder or orchard augmentation loader is included in this release.

For file storage, choose a format that preserves these arrays and instance identifiers, such as tensor dictionaries or NPZ plus metadata. The package consumes tensors and does not prescribe a dataset directory layout. Use group-level train/validation/test splits established before augmentation; keep images of the same fruit or associated sequence together.
