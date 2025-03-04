# Passim Pipeline
This directory contains the various scripts needed for running Passim
to detect the reuse of poetry on pages from works in PPA.

> [!WARNING]
> This workflow is quite preliminary and may undergo extensive change.

## Preliminary Passim Workflow

### 1. Create input corpora for `passim`
The pipeline requires one or more corpus JSONL files formatted for Passim.
The script `create_passim_corpus.py` will transform a given text corpus
JSONL file into a new format suitable for passing to Passim.

Minimally, for each record (i.e. line) the follow three fields are created:
1. `id`: The working id for the text record (e.g. ppa page or reference poem id).
         This is derived from the input record's `id` field. Optionally, this can
         be set to an alternative record field.
2. `corpus`: The corpus name the text record belongs to. This is provided as
             input to `create_passim_corpus.py`.
3. `text`: The record's text derived from the input record's `text` field.

Note that the `text` field is modified to increase passim's chance's of identifying
poetry reuse. This involves some character-level normalization as well as collapsing
converting all sequences of whitespace into a singular space character.

Optionally, all other fields of the input records may be preserved by including the
`--preserve-fields` flag.

Example usage:
```
python create_passim_corpus.py ppa_pages.jsonl ppa-pages-passim.jsonl ppa
python create_passim_corpus.py poems.jsonl ref-poems.jsonl ref-poems
```

### 2. Run `passim`
Then, we can run `passim` using the `run_passim.py` script. Note that passim requires
Java 8\*/11/17. The script has the following required arguments:
- `--ppa-corpus`: A PPA corpus file (JSONL) produced in step 1
- `--ref-corpus`: A reference corpus (JSONL) produced in step 1; can be repeated to
                  specify multiple
- `--output-dir`: Pathname to the top-level output directory for passim results

Note that this will compare PPA texts (`corpus = "ppa"`) with texts from other sources
(`corpus != "ppa"`)

Example usage:
```
python run_passim.py --ppa-corpus passim-ppa.jsonl --ref-corpus passim-ref.jsonl --output-dir passim-output
env PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH" python run_passim.py --ppa-corpus ppa.jsonl --ref-corpus refa.jsonl --ref-corpus refb.jsonl --output-dir passim-output
```

#### Optional `passim` parameters
This script can also take several optional arguments that can be passed along to passim.
They are:
- `--max-df`: The upper limit on document frequency for ngrams
- `--min-match`: The minimum number of n-gram matches between documents
- `--ngram-size`: The size of n-gram to use
- `--floating-ngrams`: Allow n-grams to float from word boundaries
- `--gap`: The minimum gap size that separates passages
- `--min-align`: The minimum length of alignment

### 3. Build page-level results from `passim` output
After running `passim`, we can use the `get_passim_page_results.py` script to build
the page-level passages identified by `passim` in a JSONL file. This will include records
for pages where `passim` identifies *no* reuse.

The intended use cases of this output are two-fold:
1. Analysing and evaluting the performance of passim
2. Extracting the excerpts identified by passim

Currently, this script has two effective modes. One which includes the underlying text
excerpts and one that does not. This is due to the fact that `passim` output files do not
contain the full excerpts themselves, only an aligned version (one that includes "-" symbols
to indicate places where a character was "inserted" for alignment).

### 3. Build `passim` page-level and excerpt-level results
After running `passim`, we can use `get_passim_results.py` to build the page-level and excerpt-level
passages identified by `passim`.

Page-level results are saved as a JSONL file, and include records for pages where `passim` identifies
for pages where `passim` identifies *no* reuse. This file is primarily meant for evaluating (and
optimizing) the performance of `passim`.

Excerpt-level results are saved as a CSV file. With each row corresponding to a single poetry excerpt
identified by `passim` in the form of `LabeledExcerpt` objects.

This script requires quite a few files beyond passim's output directory. This script also requires all
of the input corpora for the `passim` run. Additionally, since the PPA text corpus underwent text
transformations, the original PPA text corpus must be provided if accurate PPA excerpts are desired.
Currently this (`--ppa-text-corpus`) is an optional parameter.

Example usage:
```
    get_passim_results.py --ppa-passim-corpus ppa_passim.jsonl --ref-corpus ref.jsonl \
        --passim-dir passim_output --page-results passim_page_results.jsonl \
        --span-results passim_spans.csv
    get_passim_results.py --ppa-passim-corpus ppa_passim.jsonl --ref-corpus ref_a.jsonl \
        --ref-corpus ref_b.jsonl --passim-dir passim_output \
        --page-results passim_page_results.jsonl --span-results passim_spans.csv \
        --ppa-text-corpus ppa.jsonl.gz
```
