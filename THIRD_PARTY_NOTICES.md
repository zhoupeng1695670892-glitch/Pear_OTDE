# Third-party components

The self-developed modules in this repository are distributed under the MIT license in [LICENSE](LICENSE), as authorized by the contributor providing the source archive. The source archive contained no pre-existing copyright or license notices in these modules.

PyTorch is an external dependency and is distributed under its own license. YOLO-Master and its Ultralytics-derived detector code are external integration dependencies and are not redistributed here. Obtain those sources through their authorized distribution channel and retain their applicable licenses, copyright notices, and distribution conditions. This repository's MIT license does not relicense those projects or their dependencies.

The adapter requires the inspected YOLO-Master v0.10 graph and native criterion interfaces described in [docs/INTEGRATION.md](docs/INTEGRATION.md). Installing an arbitrary package named `ultralytics` does not establish compatibility with that interface.
