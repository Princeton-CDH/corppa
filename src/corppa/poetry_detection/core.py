"""
Custom data type for poetry excerpts identified with the text of PPA pages.
"""

import functools
import types
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Optional, TypeAliasType, Union, get_args, get_origin

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
    """Return the base type for a dataclass field type annotation.
    For unions or optional values, returns the first non-None type; for type
    aliases, returns the original type.
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
    raise ValueError(f"Cannot determine real type for '{field_type}'")


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
                # Convert sets to comma-separated lists
                if type(value) is set:
                    csv_dict[key] = ", ".join(value)
                else:
                    csv_dict[key] = value
        return csv_dict

    @classmethod
    @functools.cache
    def fieldnames(cls) -> list[str]:
        """Return a list of names for the fields in this class,
        in order."""
        return [f.name for f in fields(cls)]

    @classmethod
    @functools.cache
    def field_types(cls) -> dict[str, Any]:
        """Return a dictionary of field names and corresponding types
        for this class."""
        return {f.name: field_real_type(f.type) for f in fields(cls)}

    @staticmethod
    def from_dict(d: dict) -> "Excerpt":
        """
        Constructs a new Excerpt from a dictionary of the form produced by `to_dict`
        or `to_json`.
        """
        input_args = deepcopy(d)
        # Remove excerpt_id if present
        input_args.pop("excerpt_id", None)
        # Convert detection methods to set
        detection_methods = input_args["detection_methods"]
        if type(detection_methods) is list:
            ## `to_json` format
            input_args["detection_methods"] = set(detection_methods)
        elif type(detection_methods) is str:
            ## `to_csv` format
            if ", " in detection_methods:
                input_args["detection_methods"] = set(detection_methods.split(", "))
            else:
                input_args["detection_methods"] = {detection_methods}
        else:
            raise ValueError(
                f"Unexpected value type '{type(detection_methods).__name__}' for detection_methods"
            )
        return Excerpt(**input_args)

    def strip_whitespace(self) -> "Excerpt":
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

    @staticmethod
    def from_dict(d: dict) -> "LabeledExcerpt":
        """
        Constructs a new LabeledExcerpt from a dictionary of the form produced by `to_dict`
        or `to_json`.
        """
        input_args = deepcopy(d)
        # Remove excerpt_id if present
        input_args.pop("excerpt_id", None)
        # Convert detection & identification methods to set
        for fieldname in ["detection_methods", "identification_methods"]:
            value = input_args[fieldname]
            if type(value) is list:
                ## `to_json` format
                input_args[fieldname] = set(value)
            elif type(value) is str:
                ## `to_csv` format
                input_args[fieldname] = set(value.split(", "))
            else:
                raise ValueError(
                    f"Unexpected value type '{type(value).__name__}' for {fieldname}"
                )

        return LabeledExcerpt(**input_args)
