# Release validation

Version 0.3.1 was checked on Python 3.13.9 and PyTorch 2.10.0+cpu with two CPU threads.

| Check | Result |
| --- | --- |
| Supplied archive's original unit suite | 178 passed, 0 failed, 0 errors, 0 skipped |
| Release unit suite | 183 passed, 0 failed, 0 errors, 0 skipped |
| Core module forward/backward example | Passed |
| Native YOLO-Master synthetic forward/backward and nonempty inference | Passed |
| Native numerical comparison against the original archive | Identical reported loss components, response count, gradient norms, and predictions' shapes/counts |
| Python syntax compilation | Passed |
| Wheel build and isolated installation/import | Passed |
| CITATION.cff | Passed the official CFF 1.2.0 schema |
| Python explanatory comments and docstrings | Removed; no remaining comments or docstrings |
| English-only release text and filenames | Checked |

The unit suite covers phase sampling, reference moments, projection energy, gradients, finite probabilities, connected support, soft box capture, annotation availability, ROI selection, query-response learning, and adapter contracts. Five additional tests cover nonadjacent V/S containment, valid partial annotations, critical-only annotations, and empty prediction outputs.

Detailed records are in [validation_results.json](validation_results.json) and [native_interface_check.json](native_interface_check.json). Native verification uses an externally installed YOLO-Master source tree and synthetic inputs. No dataset training, optimizer update, GPU timing, AP computation, or replication of the manuscript's reported results was performed for these release checks.

## Changes from the supplied archive

1. Added the missing known-S-versus-known-V containment validation when D is unavailable, including the standalone GSF and retained control entry points. Valid-input behavior is unchanged.
2. Removed Python explanatory comments and docstrings through an AST-checked transformation; retained all original research modules and tests.
3. Replaced workspace-specific and non-English documentation with installation, tensor-format, integration, evaluation-output, and manuscript-correspondence guides.
4. Replaced historical validation records with fresh results and removed personal absolute paths from the native-check report.
5. Added the contributor-authorized MIT license, third-party notices, unpublished-manuscript citation metadata, dependency files, packaging metadata, and a core smoke example.

The original archive remains unchanged. No upstream detector sources, weights, data, credentials, caches, build products, or unrelated manuscript assets are bundled.

The citation schema used for validation is maintained by the [Citation File Format project](https://github.com/citation-file-format/citation-file-format/blob/main/schema.json).
