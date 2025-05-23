# Copyright (c) 2024-2025, Center for Digital Humanities, Princeton University
# SPDX-License-Identifier: Apache-2.0

"""
Custom data type for poetry excerpts identified with the text of PPA pages.
"""

import types
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from typing import Any, Optional, Self, get_args, get_origin

from Bio.Align import PairwiseAligner

# Table of supported detection methods and their corresponding prefixes
DETECTION_METHODS = {
    "adjudication": "a",
    "manual": "m",
    "passim": "p",
    "xml": "x",
}


@dataclass
class Span:
    """
    Span object representing a Pythonic "closed open" interval
    """

    start: int
    end: int
    label: str

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("Start index must be less than end index")

    def __len__(self) -> int:
        return self.end - self.start

    def has_overlap(self, other: "Span", ignore_label: bool = False) -> bool:
        """
        Returns whether this span overlaps with the other span.
        Optionally, span labels can be ignored.
        """
        if ignore_label or self.label == other.label:
            return self.start < other.end and other.start < self.end
        return False

    def is_exact_match(self, other: "Span", ignore_label: bool = False) -> bool:
        """
        Checks if the other span is an exact match. Optionally, ignores
        span labels.
        """
        if self.start == other.start and self.end == other.end:
            return ignore_label or self.label == other.label
        else:
            return False

    def overlap_length(self, other: "Span", ignore_label: bool = False) -> int:
        """
        Returns the length of overlap between this span and the other span.
        Optionally, span labels can be ignored for this calculation.
        """
        if not self.has_overlap(other, ignore_label=ignore_label):
            return 0
        else:
            return min(self.end, other.end) - max(self.start, other.start)

    def overlap_factor(self, other: "Span", ignore_label: bool = False) -> float:
        """
        Returns the overlap factor with the other span. Optionally, span
        labels can be ignored for this calculation.

        The overlap factor is defined as follows:

            * If no overlap (overlap = 0), then overlap_factor = 0.

            * Otherwise, overlap_factor = overlap_length / longer_span_length

        So, the overlap factor has a range between 0 and 1 with higher values
        corresponding to a higher degree of overlap.
        """
        overlap = self.overlap_length(other, ignore_label=ignore_label)
        return overlap / max(len(self), len(other))


def field_real_type(field_type) -> type:
    """Return the real type for a dataclass field type annotation.
    For unions or optional values (e.g. `Optional[int]`), returns the first
    non-None type; for type aliases (e.g. `set[str]`, returns the original type
    that was used to create the alias. For example:
        - int -> int
        - Optional[int] -> int
        - set[str] -> set
    """
    # if it's a regular type, return unchanged
    if isinstance(field_type, type):
        return field_type
    # for a type alias, return the original type
    # e.g. for annotation of set[str] return the set type
    origin = get_origin(field_type)
    if isinstance(origin, type):
        return origin
    # if Optional or Union, return the first non-none type
    ftypes = get_args(field_type)
    if ftypes:
        return [arg for arg in ftypes if arg != types.NoneType][0]

    # if we get here, this is an input we can't handle
    raise TypeError(f"Cannot determine real type for '{field_type}'")


#: character to use when converting sets to and from delimited string
MULTIVAL_DELIMITER = "; "


def input_to_set(input_val: list | str | set) -> set:
    """Convert supported inputs to set; intended for convenience
    when initializing :attr:`Excerpt.detection_methods` and
    :attr:`LabeledExcerpt.identification_methods`.
    """
    # match case syntax equivalent to isinstance(input_val, list)
    match input_val:
        case list():  # format used by to_json
            return set(input_val)
        case str():  # format used by to_csv
            return set(input_val.split(MULTIVAL_DELIMITER))
        case set():
            return input_val
        case _:
            raise ValueError(f"Unexpected value type '{type(input_val).__name__}'")


@dataclass(kw_only=True, frozen=True)
class Excerpt:
    """
    A detected excerpt of poetry within a PPA page text. Excerpt objects are immutable.
    """

    # PPA page related
    page_id: str
    ppa_span_start: int
    ppa_span_end: int
    ppa_span_text: str
    # Detection methods
    detection_methods: set[str]
    # Optional notes field
    notes: Optional[str] = None
    # Excerpt id, set in post initialization
    # Note: Cannot be passed in at initialization
    excerpt_id: str = field(init=False)

    def __post_init__(self):
        # Check PPA span indices
        if self.ppa_span_end <= self.ppa_span_start:
            raise ValueError(
                f"PPA span's start index {self.ppa_span_start} must be less than its end index {self.ppa_span_end}"
            )

        # Check that dectection method set is not empty
        if not self.detection_methods:
            raise ValueError("Must specify at least one detection method")

        # Validate detection methods
        unsupported_methods = self.detection_methods - DETECTION_METHODS.keys()
        if unsupported_methods:
            error_message = "Unsupported detection method"
            if len(unsupported_methods) == 1:
                error_message += f": {next(iter(unsupported_methods))}"
            else:
                error_message += f"s: {', '.join(unsupported_methods)}"
            raise ValueError(error_message)

        # Set excerpt id
        ## Get ID prefix
        if len(self.detection_methods) == 1:
            [detect_name] = self.detection_methods
            detect_pfx = DETECTION_METHODS[detect_name]
        else:
            # c for combination
            detect_pfx = "c"
        excerpt_id = f"{detect_pfx}@{self.ppa_span_start}:{self.ppa_span_end}"
        object.__setattr__(self, "excerpt_id", excerpt_id)

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a JSON-friendly dict of the poem excerpt. Note that unset optional fields
        are not included.
        """
        json_dict = {}
        for key, value in asdict(self).items():
            # Skip unset / null fields
            if value is not None:
                # Convert sets to lists
                if type(value) is set:
                    json_dict[key] = list(value)
                else:
                    json_dict[key] = value
        return json_dict

    def to_csv(self) -> dict[str, int | str]:
        """
        Returns a CSV-friendly dict of the poem excerpt. Note that like `to_dict` unset
        fields are not included.
        """
        csv_dict: dict[str, int | str] = {}
        for key, value in asdict(self).items():
            if value is not None:
                # Convert sets to delimited string
                if type(value) is set:
                    # to guarantee deterministic order, sort before joining
                    csv_dict[key] = MULTIVAL_DELIMITER.join(sorted(value))
                else:
                    csv_dict[key] = value
        return csv_dict

    @classmethod
    def fieldnames(cls, required_only=False) -> list[str]:
        """Return a list of names for the fields in this class,
        in order. Takes an optional parameter `required_only` to
        return the list of fields that are required for initialization."""
        cls_fields = fields(cls)
        # when requested, filter required fields based on default
        # value and fields where init=False
        if required_only:
            cls_fields = [
                f
                for f in cls_fields
                if (f.default == MISSING and f.default_factory == MISSING and f.init)
            ]

        return [f.name for f in cls_fields]

    @classmethod
    def field_types(cls) -> dict[str, Any]:
        """Return a dictionary of field names and corresponding types
        for this class."""
        return {f.name: field_real_type(f.type) for f in fields(cls)}

    @classmethod
    def from_dict(cls, d: dict) -> "Excerpt":
        """
        Constructs a new Excerpt from a dictionary in the format generated by
        meth:`Excerpt.to_dict` or meth:`Excerpt.to_csv`.  Input values
        for set fields are converted with :meth:`input_to_set`; input
        values for integer fields support conversion from string.
        """
        input_args = deepcopy(d)
        # Remove excerpt_id if present
        input_args.pop("excerpt_id", None)
        cls_field_types = cls.field_types()
        # Convert any set-type fields (i.e., detection methods)
        set_fields = [k for k, v in cls_field_types.items() if v is set]
        for field_name in set_fields:
            try:
                input_args[field_name] = input_to_set(input_args[field_name])
            except ValueError as err:
                raise ValueError(f"{err} for {field_name}")

        # support conversion from string to integer when loading from csv
        int_fields = [k for k, v in cls_field_types.items() if v is int]
        for field_name in int_fields:
            input_val = input_args.get(field_name)
            if isinstance(input_val, str):
                if input_val == "":
                    del input_args[field_name]
                else:
                    input_args[field_name] = int(input_val)

        return cls(**input_args)

    def strip_whitespace(self) -> Self:
        """
        Return a copy of this excerpt with any leading and trailing whitespace
        removed from the text and start and end indices updated to match any changes.
        """
        ldiff = len(self.ppa_span_text) - len(self.ppa_span_text.lstrip())
        rdiff = len(self.ppa_span_text) - len(self.ppa_span_text.rstrip())
        return replace(
            self,
            ppa_span_start=self.ppa_span_start + ldiff,
            ppa_span_end=self.ppa_span_end - rdiff,
            ppa_span_text=self.ppa_span_text.strip(),
        )

    def correct_page_excerpt(self, page_text: str) -> Self:
        """
        For an excerpt that may have undergone textual transformations during
        the detection process, this method attemps to correct the excerpt such
        that the returned excerpt has the indicies and text correpsonding to
        the original page text (rather than the transformed version). The
        correspondence is determined using the Needleman-Wunsch algorithm.

        Warning: If the wrong page text is passed in, this method will still
        find an alignment, but "correct" the excerpt... but not to something
        meaningful. In the future, we may add support to guard against this
        issue.
        """
        # TODO: Make the alignment settings more visible. However, they are
        #       meant to be fixed values.
        # See BioPython docs for more detail on the PairwiseAligner
        # https://biopython.org/docs/dev/Tutorial/chapter_pairwise.html#sec-pairwise-aligner
        aligner = PairwiseAligner(
            mismatch_score=-0.5,
            gap_score=-0.5,
            query_left_gap_score=0,  # no penalty for gaps to the left of the excerpt
            query_right_gap_score=0,  # no penlty for gaps to the right of the excerpt
        )
        # List of best alignments, there can be more than one
        results = aligner.align(page_text, self.ppa_span_text)
        # Use first resulting alignment even if there are more than one
        alignment = results[0]
        # Get the PPA pages aligned sequences.
        ppa_aligned_seqs = alignment.aligned[0]
        ## Starting index of the first aligned sequence
        new_start = ppa_aligned_seqs[0][0]
        ## Ending index of the final aligned sequence
        new_end = ppa_aligned_seqs[-1][1]
        return replace(
            self,
            ppa_span_start=new_start,
            ppa_span_end=new_end,
            ppa_span_text=page_text[new_start:new_end],
        )


@dataclass(kw_only=True, frozen=True)
class LabeledExcerpt(Excerpt):
    """
    An identified excerpt of poetry within a PPA page text.
    """

    # Reference poem related
    poem_id: str
    ref_corpus: str
    ref_span_start: Optional[int] = None
    ref_span_end: Optional[int] = None
    ref_span_text: Optional[str] = None

    # Identification methods
    identification_methods: set[str]

    def __post_init__(self):
        # Run Excerpt's post initialization
        super().__post_init__()
        # Check that identification method set is not empty
        if not self.identification_methods:
            raise ValueError("Must specify at least one identification method")
        # Check that both reference span indicies are set or unset
        if (self.ref_span_start is None) ^ (self.ref_span_end is None):
            raise ValueError("Reference span's start and end index must both be set")
        # Check reference span indices if set
        if self.ref_span_end is not None and self.ref_span_end <= self.ref_span_start:
            raise ValueError(
                f"Reference span's start index {self.ref_span_start} must be less than its end index {self.ref_span_end}"
            )

    @classmethod
    def from_excerpt(cls, ex: Excerpt, **kwargs: dict) -> "LabeledExcerpt":
        """Create a :class:`LabeledExcerpt` using an :class:`Excerpt` as a
        starting point and supplying data for additional fields."""
        excerpt_info = asdict(ex)
        excerpt_info.update(kwargs)
        excerpt_info.pop("excerpt_id")
        return cls(**excerpt_info)
