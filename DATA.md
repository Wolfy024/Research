# Data protocol

Datasets are downloaded separately and are never committed to this repository.

## Showcase training cohort

- Source: COCO val2017.
- Input directory: data/coco/val2017.
- Selection: discover supported images in sorted path order, shuffle indices
  with Python seed 24, and take the first 2,000 indices.
- Processing: convert to RGB and center-fit to 256×256 with Lanczos resampling.

The training cohort digest in results/showcase_manifest.json hashes the ordered
list of selected filenames.

## Showcase evaluation cohort

- Source: DIV2K validation.
- Input directory: data/div2k/DIV2K_valid_HR.
- Selection: every discovered image in sorted path order (100 images).
- Processing: convert to RGB and center-fit to 256×256 with Lanczos resampling.
- Random payload seed: 2407.

The manifest stores every evaluation filename and source-file SHA-256. The
evaluation is content-disjoint from COCO training by construction.

## Licensing

COCO and DIV2K retain their original dataset and image licenses. Review those
terms before downloading or using the data. This repository commits only
derived charts, a qualitative research panel, hashes, and metrics; it does not
redistribute either raw dataset.
