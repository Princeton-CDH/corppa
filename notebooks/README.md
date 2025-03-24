# corppa demo notebooks

These Jupyter notebooks are provided with `corppa` to provide preliminary
access and structural analysis of the poetry excerpt dataset generated 
in part by this codebase.

## Setup

To run locally, you need a python environment with corppa and related
dependencies installed. See the [main readme](.../README.md) setup instructions.

The notebooks require a local copy of compiled excerpt data along with associated
metadata for poems and PPA works. Make sure to create `corppa.cfg` as
described in the main readme, and set the path to your downloaded copy of 
the found poems dataset.

Run `jupyter-lab`, which should open in your web browser and allow
you to browse and run the notebooks in this folder.

## Automated checks

The notebooks in this folder are checked using a GitHub Actions workflow
that runs [treon](https://github.com/ReviewNB/treon) to confirm that
notebooks still run after any changes to the corppa code used in the notebook.

To allow these checks to work on GitHub, a small set of sample data is
included in the `/notebooks/sample_data` folder, along with a config file.

If you want to run treon locally, you can specify a directory or a specific path:

```console
treon notebooks/
treon notebooks/poetry_excerpt_review.ipynb
``

Be aware that running treon locally will use whatever poem dataset path
is configured in your local `corppa_config.yml` config file.


 