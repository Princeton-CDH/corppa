# CHANGELOG

## 0.4.0
- Now supports and tested against both Python 3.11 and 3.12
- Now licensed under Apache 2
### Documentation
- Set up Sphinx documentation
- New documentation page specifically for Ends of Prosody participants
- Improved code documentation throughout the code
- Tutorial notebooks by @WHaverals to provide orientation to PPA data and related corppa functionality
- Split out developer documentation from main README
-
### Annotation
- Prodigy custom command recipe for reporting on annotation progress
- `process_adjudication_data` script to process reviewed annotation data
### Poetry Detection
- Code for parsing and reporting on tags/attributes in Chadwyck-Healey poetry corpus
- New `dataclasses` for `Span`, `Excerpt`, and `LabeledExcerpt`
- Evaluation code and documentation for comparing spans
- `merge_excerpts` script for combining labeled and unlabeled poem excerpts
- Code for working with passim (preparing corpus input files, running passim, working with the results)
- Polars utility methods for working with excerpt data
- `refmatcha` script for identifying excerpts based on matches in local reference corpora (preliminary)
- Preliminary Jupyter notebooks for reviewing found poetry excerpt data
- Preliminary configuration handling for found poetry and reference corpora
### Utilities
- `collate_txt` script to create work-level text corpora files after running OCR
- `build_text_corpus` script to convert a directory of text files into a JSONL corpus
- `get_ppa_source` method now supports all PPA sources (added support for EEBO-TCP)
- New utility function for extracting the page number from the filename of page-level content (e.g., text or image) (currently Gale/ECCO only)
- New utility function that returns a relative path generator of files with one or more extensions under a specified base directory
### Misc
- Added GitHub Actions workflow to check Jupyter notebooks
- Renamed the GitHub repository from `ppa-nlp` to `corppa`; early experimental work not included
  in this package preserved in https://github.com/Princeton-CDH/ppa-nlp-archive
- Increased use of Python type hinting
- Configured codecov with separate reporting for tests and whole project, with different targets for coverage

## 0.3.0
- New dependency: intspan
### Poetry Detection
- New Prodigy recipe for adjudicating text annotations
- Refactored recipes to use Prodigy API
- Extended recipes to optionally fetch media (i.e., images)
- Added unit testing
### Misc
- Fixed Codecov integration

## 0.2.0
- Now requires Python 3.12
### Corppa Utilities
- Basic readme documentation for filter script
- New script for OCR with google vision
- Updated filter script:
  - Uses PPA work ids instead of source ids
  - Additional filtering by volume and page
  - Additional filtering by include or exclude key-pair values
- New utilities function for working with PPA corpus file paths
- New script for generating PPA page subset to be used in conjunction with the filter script
- New script for adding image relative paths to a PPA text corpus
### Poetry Detection
- New Prodigy recipes and custom CSS for image and text annotation
- Script to add PPA work-level metadata for display in Prodigy
### Misc
- Ruff precommit hook now configured to autofix import order


## 0.1.0
- Utility to filter the full text corpus by source ID
- Experimental Scripts
  - OCR evaluation
  - Character-level statistics
