#!/usr/bin/env python

import pathlib

import polars as pl

from corppa.config import get_config
from corppa.poetry_detection.polars_utils import add_ref_poems_meta, load_excerpts_df
from corppa.poetry_detection.ppa_works import (
    add_ppa_works_meta,
    extract_page_meta,
    load_ppa_works_df,
)


def main():
    cfg = get_config()
    data_dir = pathlib.Path(cfg["compiled_dataset"]["data_dir"])

    ppa_meta_df = load_ppa_works_df(data_dir / "ppa_work_metadata.csv")
    # filter to clustered works, then group by clusters
    # count the works and sum the pages, then order by number of works
    ppa_clusters_df = (
        ppa_meta_df.filter(pl.col("ppa_cluster_id").is_not_null())
        .group_by("ppa_cluster_id")
        .agg(
            [
                pl.col("ppa_work_id").count().alias("work_count"),
                pl.col("ppa_num_excerpts").sum().alias("total_excerpts"),
                pl.col("ppa_num_excerpts").min().alias("min_excerpts"),
                pl.col("ppa_num_excerpts").max().alias("max_excerpts"),
                pl.col("ppa_num_poems").sum().alias("total_poems"),
                pl.col("ppa_num_poems").min().alias("min_poems"),
                pl.col("ppa_num_poems").max().alias("max_poems"),
            ]
        )
        .with_columns(
            excerpt_variance=pl.col("max_excerpts").sub(pl.col("min_excerpts")),
            poem_variance=pl.col("max_poems").sub(pl.col("min_poems")),
        )
        .sort("work_count", descending=True)
    )

    # identify clusters with low variance in excerpts found in clustered works
    lowvariance_clusters_df = ppa_clusters_df.filter(
        pl.col("excerpt_variance").le(50) | pl.col("poem_variance").le(30)
    )

    lowvariance_clusters = lowvariance_clusters_df["ppa_cluster_id"].to_list()
    print(
        f"{len(lowvariance_clusters)} of {ppa_clusters_df.height} clusters have low variance in excerpts"
    )

    # for works in low-variance clusters, choose first work in each cluster
    ppa_lvcluster_works = ppa_meta_df.filter(
        pl.col("ppa_cluster_id").is_in(lowvariance_clusters)
    )
    ppa_lvcluster_exemplars = (
        ppa_lvcluster_works.sort("ppa_pub_year")
        .group_by("ppa_cluster_id", maintain_order=True)
        .agg(pl.first("ppa_work_id"), pl.first("ppa_pub_year"))
    )
    # how many works are we omitting ?
    num_omitted_works = ppa_lvcluster_works.height - len(lowvariance_clusters)

    # and collect all other works
    other_works = ppa_meta_df.filter(
        pl.col("ppa_cluster_id").is_null()
        | ~pl.col("ppa_cluster_id").is_in(lowvariance_clusters)
    )
    selected_works = other_works.select("ppa_work_id").extend(
        ppa_lvcluster_exemplars.select("ppa_work_id")
    )
    print(
        f"Selecting {ppa_lvcluster_exemplars.height} exemplars from low-variance "
        + f"clusters and {other_works.height:,} other works (omitted {num_omitted_works:,})"
    )
    print(
        f"PPA work subset is {selected_works.height:,} of {ppa_meta_df.height:,} total works."
    )

    # load excerpts and then filter based on identified works
    excerpts_df = extract_page_meta(
        load_excerpts_df(
            data_dir / "excerpts.csv.gz",
            # ppa_works_meta=data_dir / "ppa_work_metadata.csv",
            # ref_poems_meta=data_dir / "poem_meta.csv",
        )
    )
    # extract page meta for joining but otherwise don't modify
    print(f"\nLoaded {excerpts_df.height:,} excerpts. ")

    subset_excerpts_df = excerpts_df.filter(
        pl.col("ppa_work_id").is_in(selected_works["ppa_work_id"])
    )
    print(f"\tFilter by low-variance cluster: {subset_excerpts_df.height:,} excerpts.")

    # filtering by cluster still too large, so also subset by PPA collection
    lit_ob_works = ppa_meta_df.filter(
        pl.col("ppa_collections").list.contains("Literary")
        | pl.col("ppa_collections").list.contains("Original Bibliography")
    )
    subset_excerpts_df = subset_excerpts_df.filter(
        pl.col("ppa_work_id").is_in(lit_ob_works["ppa_work_id"])
    )
    print(
        f"\tLimit to Literary and Original Bibliography collections: {subset_excerpts_df.height:,} excerpts."
    )

    # internet poems ref corpus is probably largely duplicative; omit for now
    subset_excerpts_df = subset_excerpts_df.filter(
        ~pl.col("ref_corpus").eq("internet_poems")
    )
    print(
        f"Exclude internet poems reference corpus: {subset_excerpts_df.height:,} excerpts."
    )

    # save the subset
    print("Saving subset to subset_excerpts.csv")
    subset_excerpts_df.with_columns(
        # convert list fields back to delimited string
        detection_methods=pl.col("detection_methods").list.join(";"),
        identification_methods=pl.col("identification_methods").list.join(";"),
    ).write_csv("subset_excerpts.csv")

    # combine with poem/ppa metadata
    # TODO: need to subset ppa metadata
    subset_excerpts_df = add_ppa_works_meta(
        subset_excerpts_df, data_dir / "ppa_work_metadata.csv"
    )
    subset_excerpts_df = add_ref_poems_meta(
        subset_excerpts_df, data_dir / "poem_meta.csv"
    )

    # omit ppa metadata fields that we don't need
    subset_excerpts_df = subset_excerpts_df.drop(
        ["ppa_added", "ppa_updated", "ppa_num_excerpts", "ppa_num_poems"]
    )
    subset_excerpts_df.with_columns(
        # convert list fields back to delimited string
        detection_methods=pl.col("detection_methods").list.join(";"),
        identification_methods=pl.col("identification_methods").list.join(";"),
        ppa_collections=pl.col("ppa_collections").list.join(";"),
    ).write_csv("subset_excerpts_with_poem_ppa_metadata.csv")


if __name__ == "__main__":
    main()
