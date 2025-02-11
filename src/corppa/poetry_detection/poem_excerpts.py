"""
Custom data type for poetry excerpts identified with the text of PPA pages.
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(kw_only=True)
class PoemExcerpt:
    """
    A poem excerpt data type representing an identified poem excerpt within a
    PPA page
    """

    # PPA page related
    page_id: str
    ppa_span_start: int
    ppa_span_end: int
    ppa_span_text: str
    # Reference poem related
    poem_id: str
    ref_corpus: str
    ref_span_start: Optional[int] = None
    ref_span_end: Optional[int] = None
    ref_span_text: Optional[str] = None
    # Methods & additional notes
    detection_methods: set[str]
    identification_methods: set[str]
    notes: Optional[str] = None

    def __post_init__(self):
        # Check PPA span indices
        if self.ppa_span_end <= self.ppa_span_start:
            raise ValueError(
                f"PPA span's start index {self.ppa_span_start} must be less than its end index {self.ppa_span_end}"
            )
        # Check that both reference span indicies are set or unset
        if (self.ref_span_start is None) ^ (self.ref_span_end is None):
            raise ValueError("Reference span's start and end index must both be set")
        # Check reference span indices if set
        if self.ref_span_end is not None and self.ref_span_end <= self.ref_span_start:
            raise ValueError(
                f"Reference span's start index {self.ref_span_start} must be less than its end index {self.ref_span_end}"
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a JSON-friendly dict of the poem excerpt. Note that unset optional fields
        are not included.
        """
        json_dict = asdict(self)
        json_dict["detection_methods"] = list(self.detection_methods)
        json_dict["identification_methods"] = list(self.identification_methods)

        # Remove optional fields if set to None
        for field in ["ref_span_start", "ref_span_end", "ref_span_text", "notes"]:
            if json_dict[field] is None:
                del json_dict[field]
        return json_dict
