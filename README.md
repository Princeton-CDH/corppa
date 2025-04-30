# corppa

This repository is research software developed as part of the [Ends of Prosody](https://cdh.princeton.edu/projects/the-ends-of-prosody/), which is associated with the [Princeton Prosody Archive](https://prosody.princeton.edu/) (PPA). This software is particularly focused on research and work related to PPA full-text and page image corpora.

Documentation for this package is available at <https://princeton-cdh.github.io/corppa/>.

**WARNING:** This code is primarily for internal team use.
Specific portions that may be useful are included in the
[Ends of Prosody utilities](https://princeton-cdh.github.io/corppa/eop-docs.html) documentation
which was created for participates of the [Ends of Prosody](https://cdh.princeton.edu/events/the-ends-of-prosody/)
conference.

For early experimental work on this project that is not included in the corppa python package, see https://github.com/Princeton-CDH/ppa-nlp-archive

## Basic Usage

### Installation

Use pip to install as a python package directly from GitHub.  Use a branch or tag name, e.g. `@develop` or `@0.1` if you need to install a specific version.

```sh
pip install git+https://github.com/Princeton-CDH/corppa.git#egg=corppa
```
or
```sh
pip install git+https://github.com/Princeton-CDH/corppa.git@v0.1#egg=corppa
```

### Scripts

Installing `corppa` currently provides access to two command line scripts:
* `corppa-filter`: For filtering a PPA page-level corpus. Corresponds to `corppa.ocr.gvision_ocr.py`.
* `corppa-ocr`: For generating OCR text for images using Google Vision API. Corresponds to `corppa.utils.filter.py`.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

(c)2025 Trustees of Princeton University. Permission granted for non-commercial
distribution online under a standard Open Source license.

## Experimental Scripts

Experimental scripts associated with `corppa` are located within the `scripts` directory.
See this directory's README for more detail.
