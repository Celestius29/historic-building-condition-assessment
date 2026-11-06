# historic-building-condition-assessment

Crack segmentation, morphology indices and damage quantification for historic masonry and timber buildings, with point-cloud and geospatial modules.

This repository is a work in progress. Nothing here has been run or validated yet.

## 1. Motivation and problem definition

TODO: Describe why condition assessment of historic masonry and timber buildings matters, and define the problem.

## 2. Pipeline overview

TODO: Add a diagram.

- 2D branch: crack segmentation > morphology indices > damage function
- 3D branch (separate): point-cloud module
- Geospatial branch (separate): geospatial module

## 3. Datasets and licences

See [datasets/README.md](datasets/README.md). Datasets are not stored in this repository. The code is under the MIT licence; datasets keep their own licences.

TODO: Summarise licence terms per dataset.

## 4. Methods

### Segmentation
TODO

### Morphology indices
TODO

### Damage function
TODO

### Point-cloud processing
TODO

### Geospatial
TODO

## 5. Experiments and results

TODO: No results yet.

| Model | Dataset | Split | IoU |
|-------|---------|-------|-----|
| TODO  | TODO    | TODO  | TODO |

| Model | Dataset | Split | Dice |
|-------|---------|-------|------|
| TODO  | TODO    | TODO  | TODO |

## 6. Reproducibility

### Colab
TODO

### Local
TODO

## 7. Limitations

- Image data and 3D data come from different sources and are not fused.
- The damage indicator is an experimental, unvalidated composite.
- A single scan gives no condition monitoring.

## 8. Status

| Task | Status |
|------|--------|
| Crack segmentation | Planned |
| Morphology indices | Planned |
| Matlab check | Planned |
| Damage function | Planned |
| Point-cloud module | Planned |
| Geospatial module | Planned |
| Report | Planned |
