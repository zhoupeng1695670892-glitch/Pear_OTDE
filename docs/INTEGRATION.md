# Native detector integration

## Required detector interface

The adapter targets the inspected YOLO-Master v0.10 configuration at `ultralytics/cfg/models/master/v0_10/yolo-master.yaml`. It expects:

- A 25-node model with detection inputs at nodes 17, 20, and 23.
- A single fruit class, `nc=1`, with `end2end=False`.
- Native `forward_head`, `_get_decode_boxes`, `bbox_decode`, and `get_assigned_targets_and_loss` interfaces.
- Native CIoU and DFL, with no additional IoU auxiliary loss or WiseIoU replacement.
- The native mixture-loss composition interface when a wrapped criterion provides routing losses.

Supply a constructed native model, wrap it with `PearMasterAdapter`, then create the optimizer so the added modules are included. The package does not register an Ultralytics Trainer or Validator by import.

## Verify the native environment

```shell
python examples/verify_native.py --master-root /path/to/YOLO-Master-Standard --device cpu --output docs/native_interface_check.json
```

This command constructs the native model and checks a synthetic loss, backward gradients, and nonempty predictions. Its zero score threshold is only for exercising untrained inference. It performs no optimizer update and does not evaluate a dataset.

## Training

The README provides a minimal explicit training loop. The full manuscript schedule uses 200 epochs, microbatch size 8, accumulation over 2 steps, AdamW with learning rate 0.001 and weight decay 0.0005, five warmup epochs, and cosine decay to 0.01 of the initial learning rate. Bias and normalization parameters are excluded from decay; the routing learning-rate multiplier is 0.5, gradient clipping is 10, expert warmup lasts 3 epochs, and EMA is disabled. The image pipeline uses letterbox preprocessing and horizontal flipping with probability 0.5. The source study selects checkpoints by validation ABC AP50:95 and uses seeds 17, 42, and 2026.

Configure these operations in the native training application. For accumulation, divide each microbatch loss by the actual number of microbatches in its accumulation window, including a shorter final window. Save the adapter's `state_dict`, its `PearConfig` values, native model configuration, preprocessing settings, and optimizer/scheduler state. Reload them into the same architecture before evaluation.

The main loss retains the native detection criterion and appends grade, GSF statistical, context, shallow, fusion-support, and response losses. `GSFIoULoss.forward` includes geometry for standalone use; the native bridge calls `retention` so CIoU is not counted twice. Setting `response_weight=0` retains both context-BCE terms.

## Inference and evaluation outputs

Call `model.eval()` before `model.predict(images)`. Default inference uses fruit score 0.05, pre-NMS top-k 1,000, class-agnostic NMS IoU 0.6, and at most 100 boxes per image. Returned boxes are clipped to the preprocessed image frame. Restore the original image coordinates in the application, then remove any boxes made degenerate by the inverse transform.

For the manuscript's ABC AP export, construct `PearConfig(score_threshold=0.001)` and rank each candidate by the joint score of its selected grade. The retained `grade_scores[:, 2]` provide the grade-C review score. The package returns all three probabilities and joint scores, including learned prior outputs for low-support queries.

Dataset metrics, association-group bootstrap intervals, fixed-capacity review, common-map fitting, same-fruit pairing, and cross-crop evaluation are manuscript protocols rather than bundled evaluators. Their denominators and filtering rules are specified in Sections 4.1 and A3–A5. Synthetic validation results in this repository are implementation checks and are not the manuscript's AP or grading results.

## Controls

`rcpm`, `lecf`, and `cseg` in `PearConfig` switch components. `cseg=False` selects a coarse-only basic ROI head and disables the response branch, so it changes more than aggregation alone. `use_fine_roi=False` retains the structured readout while changing the feature scale. The lower-level `DetectorBridge`, `FeatureAdapter`, `OrdinalEvidenceReadout`, and point/coverage loss classes are retained as control implementations; `PearMasterAdapter` and the public `master_training_loss` define the main method.
