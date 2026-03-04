#!/usr/bin/env python
"""
This is an experimental script to test loading found poem
excerpt data into Grist database / spreadsheet tool (https://www.getgrist.com/).
It's preserved here for documentation purposes in case scripted
Grist import is useful for other projects.

Requires installing Grist python API client: `pip install pygrister`

Configure environment variables for Grist access:
- GRIST_API_KEY
- GRIST_DOC_ID
- GRIST_API_URL

Uses data paths in corppa config and subset_excerpts.csv created by subset script.

When experimenting, PPA metadata was imported via web upload, and then
references in excerpt data to PPA work and Poem id were converted to reference
fields manually.  This only works with a smaller amount of data (limits unclear;
5000 is ok but the larger amounts tried were not.

"""

import os
import pathlib

import polars as pl
import tqdm
from pygrister.api import GristApi

from corppa.config import get_config
from corppa.poetry_detection.polars_utils import load_excerpts_df
from corppa.poetry_detection.ppa_works import extract_page_meta, load_ppa_works_df

grist_api_key = os.environ.get("GRIST_API_KEY")
grist_doc_id = os.environ.get("GRIST_DOC_ID")
grist_api_url = os.environ.get("GRIST_API_URL")
if not grist_api_key or not grist_doc_id or not grist_api_url:
    raise SystemExit(
        "Must configure GRIST_API_KEY, GRIST_DOC_ID, and GRIST_API_URL as environment variables"
    )

corppa_cfg = get_config()
data_dir = pathlib.Path(corppa_cfg["compiled_dataset"]["data_dir"])


# load excerpt data
excerpts_df = load_excerpts_df("subset_excerpts.csv")
# extract ppa work id and page number from page id
excerpts_df = extract_page_meta(excerpts_df).with_columns(
    # convert list fields back to delimited string
    detection_methods=pl.col("detection_methods").list.join(";"),
    identification_methods=pl.col("identification_methods").list.join(";"),
)


def pl_type_to_grist(pl_type) -> str:
    # do a quick mapping from polars type to grist/python
    match pl_type:
        case int() | pl.Int64():
            return "Int"
        case str() | pl.String():
            return "Text"
    raise Exception(f"unsupported type: {pl_type}")


grist_columns = []

# subset fields to essentials to see if that will get us under grist limits
excerpts_df = excerpts_df.select(
    "page_id", "ppa_span_text", "poem_id", "ref_span_text", "ppa_work_id", "page_num"
)

ppa_meta_df = load_ppa_works_df(data_dir / "ppa_work_metadata.csv")

# convert schema to python dict for simpler iteration
for field_name, field_type in excerpts_df.schema.to_python().items():
    grist_columns.append(
        {
            "id": field_name,
            "fields": {
                "label": field_name,
                # use field name as key to get polars type
                "type": pl_type_to_grist(excerpts_df.schema[field_name]),
            },
        }
    )


grist = GristApi(
    config={
        "GRIST_SELF_MANAGED": "Y",
        "GRIST_SELF_MANAGED_HOME": grist_api_url,
        "GRIST_API_KEY": grist_api_key,
        "GRIST_WORKSPACE_ID": "7",  # found via apiconsole, not sure how else to know
        "GRIST_API_SERVER": grist_api_url,
        "GRIST_SELF_MANAGED_SINGLE_ORG": "Y",
        "GRIST_TEAM_SITE": "docs",
        "GRIST_DOC_ID": grist_doc_id,
    }
)


status_code, response = grist.list_tables()
table_ids = [record["id"] for record in response]
# create excerpt table if not present
if "Excerpts" not in table_ids:
    print("Creating Excerpts table")
    status_code, response = grist.add_tables(
        [{"id": "Excerpts", "columns": grist_columns}]
    )

# now add rows in batches
# default of 10k is too large; 1k also too large

# get a subset for testing purposes. use existing order so it's replicable
# loading too much data crashes grist
excerpts_subset_df = excerpts_df.limit(5000)

for chunk in tqdm.tqdm(excerpts_subset_df.iter_slices(n_rows=500)):
    status_code, response = grist.add_records("Excerpts", records=chunk.to_dicts())

# only include poems referenced by excerpts
poems_df = pl.read_csv(
    pathlib.Path(corppa_cfg["compiled_dataset"]["data_dir"]) / "poem_meta.csv"  # .gz"
).filter(pl.col("poem_id").is_in(excerpts_df["poem_id"]))

# create poem table if not present
if "Poems" not in table_ids:
    poem_grist_columns = []
    # convert schema to python dict for simpler iteration
    for field_name in poems_df.schema:
        poem_grist_columns.append(
            {
                "id": field_name,
                "fields": {
                    "label": field_name,
                    # use field name as key to get polars type
                    "type": pl_type_to_grist(poems_df.schema[field_name]),
                },
            }
        )

    print("Creating Poems table")
    status_code, response = grist.add_tables(
        [{"id": "Poems", "columns": poem_grist_columns}]
    )

for chunk in tqdm.tqdm(poems_df.iter_slices(n_rows=500)):
    status_code, response = grist.add_records("Poems", records=chunk.to_dicts())
