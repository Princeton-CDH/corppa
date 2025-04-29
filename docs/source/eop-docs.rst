Ends of Prosody: corppa utilities
#################################

Filter Utility
--------------
The ``corppa-filter`` command is a utility for filtering a PPA page-level text corpus.
The PPA page-level text corpus is shared as a json lines (`.jsonl`) file, which may or may not be compressed (e.g., `.jsonl.gz`).
It's often useful to filter the full corpus to a subset of pages for a specific task.

Currently, ``corppa-filter`` supports the following types of filtering:

* A list of PPA work ids (as a text file, id-per-line)
* A CSV file specifying work pages (by digital page number)(csv, page-per-line)
* A key-value pair for either inclusion or exclusion

These filtering options can be combined, generally as a logical AND.
Pages filtered by work ids or page numbers will be further filtered by the key-value logic.
In cases where both work- and page-level filtering occurs, works not specified in teh page filtered are included in full.
Works that are specified in both will be limited to the pages specified in page-level filtering.

**NOTE:** *PPA work identifiers* are based on source identifiers, i.e., the identifier from the original source (HathiTrust, Gale/ECCO, EEBO-TCP).
In most cases the work identifier and the source identifier are the same, but *if you are working with any excerpted content the work id is NOT the same as the source identifier*.
Excerpt ids are based on the combination of source identifier and the first original page included in the excerpt.
In some cases PPA contains multiple excerpts from the same source, so this provides guaranteed unique work ids.

Examples
""""""""
To create a subset corpus with *all pages* for a set of specific volumes, create a text file with a list of **PPA work identifiers**, one id per line,
and then run ``corppa-filter`` with the input file, desired output file, and path to id file. ::

  corppa-filter ppa_pages.jsonl my_subset.jsonl --idfile my_ids.txt

To create a subset of *specific pages* from specific volumes, create a CSV file that includes fields `work_id` and `page_num`,
and pass that to the filter script with the `--pg-file` option: ::

  corppa-filter ppa_pages.jsonl my_subset.jsonl --pg_file my_work_pages.csv

You can filter a page corpus to exclude or include pages based on exact-matches for attributes included in the jsonl data.
For example, to get all pages with the original page number roman numeral 'i': ::

  corppa-filter ppa_pages.jsonl i_pages.jsonl --include label=i

Filters can also be combined; for example, to get the original page 10 for every volume from a list, you could specify a list of ids and the `--include` filter: ::

  corppa-filter ppa_pages.jsonl my_subset_page10.jsonl --idfile my_ids.txt --include label=10


PPA ID Utilities
--------------------

Here are some utility functions that may be useful when working with PPA work and volume identifiers.

.. autofunction:: corppa.utils.path_utils.get_volume_id
   :noindex:

.. autofunction:: corppa.utils.path_utils.get_ppa_source
   :noindex:

.. autofunction:: corppa.utils.path_utils.encode_htid
   :noindex:

.. autofunction:: corppa.utils.path_utils.decode_htid
   :noindex:


Image Path Utilities
--------------------

Here are some utility functions that may be useful when working with the Gale/ECCO page images.

.. autofunction:: corppa.utils.path_utils.get_vol_dir
   :noindex:

.. autofunction:: corppa.utils.path_utils.get_stub_dir
   :noindex:

.. autofunction:: corppa.utils.path_utils.get_page_number
   :noindex:
