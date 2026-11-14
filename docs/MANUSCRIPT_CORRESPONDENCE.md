# Manuscript correspondence

The mapping below uses the current manuscript's main equations (1)–(28) and Appendix A equations, replacing the older equation-number list shipped in the supplied archive.

| Manuscript location | Implementation |
| --- | --- |
| Eq. (1), Appendix A2.2: joint annotations and availability | `training_helpers._sanitize_joint_maps`; `GSFIoULoss.statistics` |
| Eq. (2), Appendix A2.1: expanded context and grid sampling | `master_adapter.expanded_context_rois`; `roi.sample_rois` |
| Eqs. (3)–(6): damage extent and component retention | `CSEGGradeHead.readout_from_probabilities`; `critical_support.py` |
| Eqs. (7)–(9), (A.7)–(A.8): ordinal probabilities and stable calculation | `ordinal_readout.extent_log_odds`; `structured_grade_probabilities`; `calibration` |
| Eqs. (10)–(11): reference-calibrated soft capture | `grading_iou.normalized_soft_capture` |
| Eqs. (12)–(17), (A.9): GSF statistics and instance reduction | `GSFIoULoss`; `box_geometry.instance_mean`; `training.master_training_loss` |
| Eq. (18): grade-supervision ROI selection | `grade_selection.select_sufficient_grade_rois` |
| Eqs. (19)–(20): fixed-context query-response loss | `training.response_query_boxes`; `fixed_context_response_loss` |
| Eqs. (21)–(23), (A.13): reference phase moments | `phase_moments.RCPM`; `reference_moments`; `phase_base.aggregate_evidence_reference` |
| Eqs. (24)–(26), (A.3)–(A.4): constrained fusion | `energy_fusion.LECF`; `constrain_reference_energy` |
| Eq. (27), (A.10)–(A.12): supervision and total loss | `training.master_training_loss`; `grade_support.evidence_supervision_loss`; `training_helpers.shallow_evidence_loss` |
| Eq. (28): detection and grading outputs | `DetectorBridge.predict`; `PearMasterAdapter.grade` |
| Eq. (A.14): native detection objective | Externally supplied native detector criterion |

## Verified implementation scope

The main APIs use a 1.25 context expansion, 18/35 sampling grids, four-connected threshold-0.5 critical components, a retention power of 2, a capture temperature of 0.04, and the documented ROI tolerances. Predictions and annotation fields retain separate component identities across fixed-context response queries. Visible support is detached in grade aggregation, while regional supervision trains it. The fusion projection retains gradients through the input-defined support weights and reference scale.

Eq. (A.3) was checked against its native equation object, including the square root in `diag(sqrt(w_E))`; flattened text extraction can omit that radical. The implementation's single weighting by `w_E` is consistent with the quadratic energy.

## Release differences and boundaries

- The release adds the missing validation of S <= V when V and S are known but D is unavailable. The main helper and standalone GSF interface previously accepted such inconsistent annotations. Equivalent validation was added to the retained control interfaces. Valid-input calculations are unchanged.
- Four-connected components are labeled by a CPU traversal, which synchronizes accelerator inputs. This release has not been benchmarked against the manuscript's RTX 4090 latency measurements.
- The supplied package declared PyTorch >=2.2; the current release was tested on PyTorch 2.10.0 CPU. The manuscript reports a separate PyTorch 2.1.0/CUDA 12.1 experiment environment. Compatibility with that older environment is not established by these release tests.
- Native detector sources, trained checkpoints, dataset splits and annotations, common-map fitting, evaluation scripts, and full training orchestration are not present in the supplied archive. The adapter and data contract support integration with an external application.
- Existing lower-level control implementations remain available. Their configurations and statistics are not interchangeable with the main `PearMasterAdapter` path; the README and integration guide identify the public method entry points.

No new grading algorithm, trained weights, evaluation results, or retrospective experimental claims were introduced during release preparation.
