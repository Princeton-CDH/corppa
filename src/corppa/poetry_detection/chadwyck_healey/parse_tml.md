# Parse the weird, dirty *.TML Files

This script parses a the poetry Ccorpus preserved as a collection of .TML files.
Files use a very inconsistent markup language (and long-forgotten? TML?).
Because of the "dirty" nature of these files, the parser must handle several edge cases and formats. 
The scripts has "evolved" to accommodate the numerous irregularities that were found... The main ones, I've listed below.

cmd to run the script: `python tml_parser.py --input_dir "tml" --output_dir "tml_parsed"`

optional args: 
`--num_files 1000`  (leave out to process all files)
`--verbose` (for verbose output)

## Irregularities

1. **`<div type="line">` tags (most common)**
   - the bulk of the poetry is usually enclosed in `<div type="line">` tags;
   - occasionally, these line elements are nested within a `<div type="firstline">` element, which can lead to duplicate outputs if not handled correctly
   - *Example*:
     ```html
     <div type="firstline">
       <div type="line" n="1">The only</div>
     </div>
     <div type="line" n="2">aliens</div>
     <div type="line" n="3">we like</div>
     ...
     ```

2. **`<p>` tags with attributes**
   - some files use `<p>` tags that may include line attributes (like `n="[number]"`) to denote individual lines or stanzas;
   - there is also a variant where `<p>` tags contain nested `<div type="line">` or `<div type="firstline">` elements, which must be de-duplicated.

3. **list-based `<ul><li>` elements**
   - in rarer cases, the poetry text is embedded in unordered lists. So, if neither `<div>` nor `<p>` elements are found, the parser will look for `<ul><li>` structures

4. **some files contain only mentions to (non-existant/non-linked) figures**
   - some TML files include only a `<figure>` element (often an image) without any poetry text. These files are flagged and excluded from the corpus.

5. **other irregularities**
   - Many files use SGML/XML entities (e.g., `&amp;`, `&indent`) that need replacing with their proper characters;
   - The text is "lightly" normalized (e.g. trimming extra spaces, fixing broken encodings, preserving small caps, but getting rid of italics).

## In short, our workflow looks like this:

The parser processes each file by:
- Attempting to read with multiple encodings.
- Extracting metadata from the `<head>` section (authors, translator, title, edition, period, genre, etc.).
- Extracting poetry text from the `<body>`, following a cascading set of rules:
  - First, look for `<div type="line">` (or nested within `<div type="firstline">`);
  - If absent, try extracting text from `<p>` tags or `<ul><li>` lists;
  - Then look at `<pre><sl>` blocks for concrete poems (preserving spacing);
  - Get rid of all the other poems -- these are either empty, or just contain a `<figure>` element.