"""
Script for parsing Chadwyck-Healey custom poem (.TML) files. For an input directory
of poem (.TML) files, this script does two things:
    1. Extracts and saves the plain text of the poem into a .TXT file
    2. Extracts and compiles the metadata of all poems into a single metadata .CSV file

These outputs are saved within a specified top-level directory. Optionally, a cut-off
for the number of files to be parsed may be provided.

Examples:
`python tml_parser.py "tml" "plaintext" "ch_metadata.csv"`
`python tml_parser.py "tml" "plaintext" "ch_metadata.csv" --num-files 1000`
"""

import argparse
import csv
import re
import sys
import traceback  # for error reporting
from pathlib import Path
from subprocess import run
from typing import Optional

import ftfy
from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm

# Based on most common entities found within the Chadwyck-Healey corpus
CUSTOM_ENTITY_MAP = {
    "&indent;": "\t",  # treat indents as tabs
    "&yogh;": "ȝ",  # U+021D
    "&YOGH;": "Ȝ",  # U+021C
    "&point;": ".",
    "&pbar;": "ꝑ",  # “p–bar” is a medieval abbreviation (a p with a stroke, here U+A751)
    "&ibreve;": "ĭ",  # U+012D
    "&eshort;": "ĕ",  # e with breve (U+0115)
    "&ishort;": "ĭ",  # U+012D
    ## letters with bar / macron ##
    "&abar;": "ā",  # U+0101
    "&ebar;": "ē",  # U+0113
    "&ibar;": "ī",  # i with macron (U+012B)
    "&obar;": "ō",  # U+014D
    "&mbar;": "m\u0304",  # m + combining macron
    "&nbar;": "n\u0304",  # n + combining macron
    "&ubar;": "ū",  # U+016B
    ## dashes ##
    "&lblank;": "\u2014",  # perhaps longer em dash?
    "&sblankl;": "-",  # acts like short dash
    "&wblank;": "\u2014",  # acts like em dash
    ## superscript letters ##
    "&supera;": "ᵃ",  # U+1D43
    "&superb;": "ᵇ",  # U+1D47
    "&superB;": "ᴮ",  # U+1D2D
    "&superc;": "ᶜ",  # U+1D9C
    "&superd;": "ᵈ",  # U+1D48
    "&supere;": "ᵉ",  # U+1D49
    "&superh;": "ʰ",  # U+02B0
    "&superi;": "ⁱ",  # U+2071
    "&superl;": "ˡ",  # U+02E1
    "&superm;": "ᵐ",  # U+1D50
    "&supern;": "ⁿ",  # U+207F
    "&supero;": "ᵒ",  # U+1D52
    "&superr;": "ʳ",  # U+02B3
    "&superR;": "ᴿ",  # U+1D3F
    "&supers;": "ˢ",  # U+02E2
    "&supert;": "ᵗ",  # U+1D57
    "&superu;": "ᵘ",  # U+1D58
    ## Greek characters ##
    "&gra;": "α",  # U+03B1
    "&graa;": "ά",  # U+1F71
    "&grac;": "ᾶ",  # U+1FB6
    "&grag;": "ὰ",  # U+1F70
    "&grai;": "ᾳ",  # U+1FB3
    "&graia;": "ᾴ",  # U+1FB4
    "&graic;": "ᾷ",  # U+1FB7
    "&grar;": "ἁ",  # U+1F01
    "&grara;": "ἅ",  # U+1F05
    "&gras;": "ἀ",  # U+1F00
    "&grasa;": "ἄ",  # U+1F04
    "&grasg;": "ἂ",  # U+1F02
    "&grA;": "Α",  # U+0391
    "&GRAa;": "Ά",  # U+1FBB
    "&GRAc;": "ᾶ",  # U+1FB6
    "&GRAg;": "Ὰ",  # U+1FBA
    "&GRAr;": "Ἁ",  # U+1F09
    "&GRAra;": "Ἅ",  # U+1F0D
    "&GRAs;": "Ἀ",  # U+1F08
    "&GRAsa;": "Ἄ",  # U+1F0C
    "&grb;": "β",  # U+03B2
    "&grB;": "Β",  # U+0392
    "&grc;": "ξ",  # U+03BE
    "&grC;": "Ξ",  # U+039E
    "&grd;": "δ",  # U+03B4
    "&grD;": "Δ",  # U+0394
    "&gre;": "ε",  # U+03B5
    "&grea;": "έ",  # U+1F73
    "&greg;": "ὲ",  # U+1F72
    "&grer;": "ἑ",  # U+1F11
    "&grera;": "ἕ",  # U+1F15
    "&gres;": "ἐ",  # U+1F10
    "&gresa;": "ἔ",  # U+1F14
    "&grE;": "Ε",  # U+0395
    "&GREa;": "Έ",  # U+1FC9
    "&GREr;": "Ἑ",  # U+1F19
    "&GREra;": "Ἕ",  # U+1F1D
    "&GREs;": "Ἐ",  # U+1F18
    "&GREsa;": "Ἔ",  # U+1F1C
    "&grf;": "φ",  # U+03C6
    "&grF;": "Φ",  # U+03A6
    "&grg;": "γ",  # U+03B3
    "&grG;": "Γ",  # U+0393
    "&grh;": "η",  # U+03B7
    "&grha;": "ή",  # U+1F75
    "&grhc;": "ῆ",  # U+1FC6
    "&grhg;": "ή",  # U+1F74
    "&grhi;": "ῃ",  # U+1FC3
    "&grhia;": "ῄ",  # U+1FC4
    "&grhic;": "ῇ",  # U+1FC7
    "&grhr;": "ἡ",  # U+1F21
    "&grhra;": "ἥ",  # U+1F25
    "&grhrc;": "ἧ",  # U+1F27
    "&grhs;": "ἠ",  # U+1F20
    "&grhsa;": "ἤ",  # U+1F24
    "&grhsc;": "ἦ",  # U+1F26
    "&grhsg;": "ἢ",  # U+1F22
    "&grH;": "Η",  # U+0397
    "&GRHg": "Ὴ",  # U+1FCA
    "&GRHr;": "Ἡ",  # U+1F29
    "&GRHra;": "Ἥ",  # U+1F2D
    "&GRHrc;": "Ἧ",  # U+1F2F
    "&GRHs;": "Ἠ",  # U+1F28
    "&GRHsa;": "Ἤ",  # U+1F2C
    "&GRHsc;": "Ἦ",  # U+1F2E
    "&GRHsg;": "Ἢ",  # U+1F2A
    "&gri;": "ι",  # U+03B9
    "&gria;": "ί",  # U+1F77
    "&gric;": "ῖ",  # U+1FD6
    "&grid;": "ϊ",  # U+03CA
    "&grida;": "ΐ",  # U+1FD3
    "&gridg;": "ῒ",  # U+1FD2
    "&grig;": "ὶ",  # U+1F76
    "&grir;": "ἱ",  # U+1F31
    "&grira;": "ἵ",  # U+1F35
    "&grirc;": "ἷ",  # U+1F37
    "&grirg;": "ἳ",  # U+1F33
    "&gris;": "ἰ",  # U+1F30
    "&grisa;": "ἴ",  # U+1F34
    "&grisc;": "ἶ",  # U+1F36
    "&grI;": "Ι",  # U+0399
    "&GRIa;": "Ί",  # U+1FDB
    "&GRIg;": "Ὶ",  # U+1FDA
    "&GRIr;": "Ἱ",  # U+1F39
    "&GRIra;": "Ἵ",  # U+1F3D
    "&GRIrg;": "Ἳ",  # U+1F3B
    "&GRIs;": "Ἰ",  # U+1F38
    "&GRIsa;": "Ἴ",  # U+1F3C
    "&grk;": "κ",  # U+03BA
    "&grK;": "Κ",  # U+039A
    "&grl;": "λ",  # U+03BB
    "&grL;": "Λ",  # U+039B
    "&grm;": "μ",  # U+03BC
    "&grM;": "Μ",  # U+039C
    "&grn;": "ν",  # U+03BD
    "&grN;": "Ν",  # U+039D
    "&gro;": "ο",  # U+03BF
    "&groa;": "ό",  # U+1F79
    "&grog;": "ὸ",  # U+1F78
    "&gror;": "ὁ",  # U+1F41
    "&grora;": "ὅ",  # U+1F45
    "&grorg;": "ὃ",  # +1F43
    "&gros;": "ὀ",  # U+1F40
    "&grosa;": "ὄ",  # U+1F44
    "&grO;": "Ο",  # U+039F
    "&GROa;": "Ό",  # U+1FF9
    "&GROg;": "Ὸ",  # U+1FF8
    "&GROr;": "Ὁ",  # U+1F49
    "&GROra;": "Ὅ",  # U+1F4D
    "&GROrg;": "Ὃ",  # U+1F4B
    "&GROs;": "Ὀ",  # U+1F48
    "&GROsa;": "Ὄ",  # U+1F4C
    "&grp;": "π",  # U+03C0
    "&grP;": "Π",  # U+03A0
    "&grq;": "θ",  # U+03B8
    "&grQ;": "Θ",  # U+0398
    "&grr;": "ρ",  # U+03C1
    "&grrr;": "ῥ",  # U+1FE5
    "&grrs;": "ῤ",  # U+1FE4
    "&grR;": "Ρ",  # U+03A1
    "&GRRr;": "Ῥ",  # U+1FEC
    "&grs;": "σ",  # U+03C3
    "&grS;": "Σ",  # U+03A3
    "&grst;": "ς",  # U+03C2
    "&GRST;": "ς",  # U+03C2
    "&grt;": "τ",  # U+03C4
    "&grT;": "Τ",  # U+03A4
    "&gru;": "υ",  # U+03C5
    "&grua;": "ύ",  # U+1F7B
    "&gruc;": "ῦ",  # U+1FE6
    "&grud;": "ϋ",  # U+03CB
    "&gruda;": "ΰ",  # U+1FE3
    "&grug;": "ὺ",  # U+1F7A
    "&grur;": "ὑ",  # U+1F51
    "&grura;": "ὕ",  # U+1F55
    "&grurc;": "ὗ",  # U+1F57
    "&grurg;": "ὓ",  # U+1F53
    "&grus;": "ὐ",  # U+1F50
    "&grusa;": "ὔ",  # U+1F54
    "&grusc;": "ὖ",  # U+1F56
    "&grU;": "Υ",  # U+03A5
    "&GRUr;": "Ὑ",  # U+1F59
    "&GRUra;": "Ὕ",  # U+1F5D
    "&grw;": "ω",  # U+03C9
    "&grwa;": "ώ",  # U+1F7D
    "&grwc;": "ῶ",  # U+1FF6
    "&grwg;": "ὼ",  # U+1F7C
    "&grwi;": "ῳ",  # U+1FF3
    "&grwia;": "ῴ",  # U+1FF4
    "&grwic;": "ῷ",  # U+1FF7
    "&grwr;": "ὡ",  # U+1F61
    "&grwra;": "ὥ",  # U+1F65
    "&grwrc;": "ὧ",  # U+1F67
    "&grwrg;": "ὣ",  # U+1F63
    "&grws;": "ὠ",  # U+1F60
    "&grwsa;": "ὤ",  # U+1F64
    "&grwsc;": "ὦ",  # U+1F66
    "&grwsg;": "ὢ",  # U+1F62
    "&grW;": "Ω",  # U+03A9
    "&GRWr;": "Ὡ",  # U+1F69
    "&GRWra;": "Ὥ",  # U+1F6D
    "&GRWrc;": "Ὧ",  # U+1F6F
    "&GRWrg;": "Ὣ",  # U+1F6B
    "&GRWs;": "Ὠ",  # U+1F68
    "&GRWsa;": "Ὤ",  # U+1F6C
    "&GRWsc;": "Ὦ",  # U+1F6E
    "&grx;": "χ",  # U+03C7
    "&grX;": "Χ",  # U+03A7
    "&gry;": "ψ",  # U+03C8
    "&grY;": "Ψ",  # U+03A8
    "&grz;": "ζ",  # U+03B6
    "&grZ;": "Ζ",  # U+0396
    "&grap;": "'",  # "greek" apostrophe
    "&grcolon;": "·",  # U+0387 middot
    ## alt Greek characters ##
    "&agr;": "α",  # U+03B1
    "&egr;": "ε",  # U+03B5
    "&igr;": "ι",  # U+03B9
    "&ogr;": "ο",  # U+03BF
    "&ngr;": "ν",  # U+03BD
    "&rgr;": "ρ",  # U+03C1
    "&sgr;": "σ",  # U+03C3
    "&Sgr;": "Σ",  # U+03A3
    "&tgr;": "τ",  # U+03C4
}


def determine_encoding(text_file: Path):
    """
    Determine encoding for a given text file
    """
    # First use `file` bash commmand to differentiate UTF-8 vs single byte encodings
    p = run(["file", text_file], capture_output=True, text=True)
    if " ISO-8859 text" in p.stdout:
        # Assume ISO-8859-like texts are Latin-1 (hopefully it's not macroman)
        return "latin1"
    elif " Non-ISO extended-ASCII text" in p.stdout:
        # Assume this is Windows-1252
        return "cp1252"
    elif " UTF-8 text" in p.stdout:
        return "utf-8"
    elif " ASCII text" in p.stdout:
        # Treat ASCII as UTF-8
        return "utf-8"
    else:
        raise ValueError(f"Unknown encoding: {p.stdout}")


def replace_entities(text: str, entity_map: dict[str, str] = CUSTOM_ENTITY_MAP) -> str:
    """
    Replace SGML/XML entities with their corresponding characters.
    Parameters:
        text (str): The text in which to replace entities.
    Returns:
        str: Text with entities replaced.
    """
    if not text or "&" not in text:
        return text

    # replace all known entities
    for entity, replacement in entity_map.items():
        text = text.replace(entity, replacement)
    return text


class TMLPoetryParser:
    """
    Parser object for parsing Chadwyck-Healey poem .TML files
    """

    # Field names for output metadata
    metadata_fields = [
        "filename",
        "author_lastname",
        "author_firstname",
        "author_birth",
        "author_death",
        "author_period",
        "transl_lastname",
        "transl_firstname",
        "transl_birth",
        "transl_death",
        "title_id",
        "title_main",
        "title_sub",
        "edition_id",
        "edition_text",
        "period",
        "genre",
        "rhymes",
    ]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        output_csv: Path,
        show_progress: bool = True,
        verbose: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.metadata_file = output_csv
        self.show_progress = show_progress
        self.verbose = verbose

        # Create output directory if it doesn't already exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # keep track of files that contain only a figure
        self.figure_only = []

        # For messaging
        self.current_file: Optional[Path] = None

    def try_read_file(self, file_path: Path) -> tuple[str, str]:
        """
        Attempt to determine a file's text encoding and then read it
        Parameters:
            file_path (Path): The path to the file.
        Returns:
            tuple: (content of file as string, encoding used)
        """
        encoding = determine_encoding(file_path)
        text = file_path.read_text(encoding=encoding)
        return text, encoding

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        Steps:
            - Collapse multiple whitespace characters into a single space.
            - Fix common text encoding issues using ftfy.
            - Replace known entities.
            - Strip leading/trailing whitespace.
        Parameters:
            text (str): The raw text to be cleaned.
        Returns:
            str: Cleaned and normalized text.
        """
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = ftfy.fix_text(text, unescape_html=True, normalization="NFKC")
        text = replace_entities(text)
        text = text.strip()
        return text

    def get_text_from_element(self, element: Tag) -> str:
        """
        Extract text from an element while preserving inline formatting.
        This includes:
         - Converting italic spans to plain text.
         - Converting small caps (and similar) to uppercase.
         - Cleaning the final output text.
        Parameters:
            element (Tag): A BeautifulSoup element.
        Returns:
            str: The cleaned text content.
        """
        if not element:
            return ""

        # convert italic spans (e.g., <span class="italic">word</span> => word)
        for span in element.find_all("span", class_="italic"):
            span.replace_with(span.get_text())

        # convert small caps or other fancy spans if needed, e.g. class="smcap",
        # also handle handle 'smcapit' in the same way
        for span in element.find_all("span", {"class": ["smcap", "smcapit"]}):
            uppercase_text = span.get_text().upper()
            span.replace_with(uppercase_text)

        # now get the text!
        return self.clean_text(element.get_text())

    def print_warning(self, message: str) -> None:
        warning = f"Warning: {message}"
        if self.current_file:
            warning += f" for {self.current_file}"
        print(warning)

    def extract_poetry_text(self, soup: Tag, warning_ref: Optional[str] = None) -> str:
        """
        This is the big cheese.
        Extract the poetry text from the BeautifulSoup-parsed TML file.
        The extraction follows a cascading logic:
          1. Remove unwanted notes and copyright (we can still add to this, if needed)
          2. Collect concrete poem lines from <pre><sl> blocks
            (preserving formatting, these are concrete poems that use the arrangement
            of words to create a visual image).
          3. Process inline <div> elements inside <p> tags.
          4. Process standard line-based elements (<div type="line">).
          5. Fallback to parsing <p> or <ul><li> if no recognized line elements exist.
        Parameters:
            soup (BeautifulSoup): Parsed TML file.
        Returns:
            str: The extracted and cleaned poetry text.
        """
        try:
            body = soup.find("body")
            if not body:
                self.print_warning("No body found in document")
                return ""

            # helper function: whenever we want to return "text",
            # merge with concrete_lines (if any) and then return
            def final_return(text_str: str) -> str:
                """
                Helper to merge concrete lines (from <pre><sl>) with the rest of the text.
                Parameters:
                    text_str (str): The main extracted text.
                Returns:
                    str: Combined text if concrete lines exist, otherwise the original text.
                """
                if concrete_lines:
                    if text_str.strip():
                        return "\n".join(concrete_lines) + "\n" + text_str
                    else:
                        return "\n".join(concrete_lines)
                else:
                    return text_str

            #####################################################################
            # STEP 1: Remove unwanted notes and copyright
            # Remove any <div> or <p> tags that are identified as notes or copyright information.
            #####################################################################
            for div in body.find_all(["div", "p"], class_="note"):
                div.decompose()
            for div in body.find_all("div", type="note"):
                div.decompose()
            for div in body.find_all("div", type="copyright"):
                div.decompose()

            #####################################################################
            # STEP 2: Collect concrete poem lines from <pre><sl> blocks
            #
            # Some files (especially concrete poems) use a <pre> block containing <sl> tags.
            # This block preserves the spatial formatting.
            # We extract each <sl> block, remove any nested tags while preserving spaces,
            # replace entities, and collect these lines.
            #####################################################################
            concrete_lines = []
            for pre_block in body.find_all("pre"):
                sl_tags = pre_block.find_all("sl")
                if sl_tags:
                    for sl in sl_tags:
                        # remove any embedded tags but keep spacing
                        line_text = sl.get_text()
                        # convert entities (&amp; -> & etc.)
                        line_text = replace_entities(line_text)
                        # strip trailing newlines
                        line_text = line_text.rstrip("\r\n")
                        if line_text.strip():
                            concrete_lines.append(line_text)
                else:
                    print(f"{self.current_file}: Has <pre> block without <sl> tags!")

                # remove <pre> so it won't confuse further logic
                pre_block.decompose()

            #####################################################################
            # STEP 3: Process inline <div> elements within <p> tags
            #
            # Some files embed <div type="line"> or <div type="firstline"> inside <p> tags.
            # These paragraphs are processed separately to ensure no poetry text is missed.
            #####################################################################
            inline_paragraph_lines = []
            paragraphs_with_line_div = []
            for p in body.find_all("p"):
                direct_div_children = [
                    c
                    for c in p.children
                    if c.name == "div" and c.get("type") in ["line", "firstline"]
                ]
                if direct_div_children:
                    paragraphs_with_line_div.append(p)

            for p in paragraphs_with_line_div:
                full_para_text = self.get_text_from_element(p)
                if full_para_text.strip():
                    inline_paragraph_lines.append(full_para_text.strip())
                p.decompose()

            #####################################################################
            # STEP 4: Normal line-based logic
            #
            # Extract poetry by locating <div> tags with type "line" or "firstline".
            # Note:
            #  - Avoid double output if a <div type="firstline"> contains nested <div type="line">!!
            #####################################################################
            line_divs = body.find_all(
                lambda tag: (
                    tag.name == "div" and tag.get("type") in ["line", "firstline"]
                )
            )

            filtered_line_divs = []
            for d in line_divs:
                if d.get("type") == "firstline" and d.find("div", {"type": "line"}):
                    continue
                filtered_line_divs.append(d)
            line_divs = filtered_line_divs

            #####################################################################
            # STEP 5: Fallback
            #
            # If no <div type="line"> elements are found, try other structures:
            #   - <p> tags (if they haven't been processed already)
            #   - <ul><li> list items (rare, but they're there...)
            #   - Check if the document only has a <figure> element, in which case there is no poetry text (?)
            #####################################################################
            if not line_divs:
                paragraphs = body.find_all("p")
                if not paragraphs:
                    lists = body.find_all("ul")
                    if lists:
                        list_lines = []
                        for ul in lists:
                            for li in ul.find_all("li"):
                                li_text = self.get_text_from_element(li)
                                if li_text:
                                    list_lines.append(li_text.strip())
                        if list_lines:
                            all_text_parts = inline_paragraph_lines + list_lines
                            return final_return("\n\n".join(all_text_parts).strip())
                        else:
                            figure = body.find("figure")
                            body_text_stripped = body.get_text(strip=True)
                            if figure and not body_text_stripped:
                                return final_return("")
                            # TODO: Determine if this should be removed
                            self.print_warning(
                                "No poetry content found in any recognized format"
                            )

                            return final_return(
                                "\n\n".join(inline_paragraph_lines).strip()
                            )
                    else:
                        figure = body.find("figure")
                        body_text_stripped = body.get_text(strip=True)
                        if figure and not body_text_stripped:
                            return final_return("")
                        if inline_paragraph_lines:
                            return final_return(
                                "\n\n".join(inline_paragraph_lines).strip()
                            )
                        # TODO: Determine if this should be removed
                        self.print_warning(
                            "No poetry content found in any recognized format"
                        )
                        return final_return("")
                else:
                    paragraph_texts = []
                    for p in paragraphs:
                        raw = p.get_text(strip=True)
                        if raw == "[Copyright permission not received at this time.]":
                            return final_return("")
                        if not p.get("class") == ["note"] and p.get("type") != "note":
                            cleaned = self.get_text_from_element(p)
                            if cleaned:
                                paragraph_texts.append(cleaned)

                    if paragraph_texts:
                        all_text_parts = inline_paragraph_lines + paragraph_texts
                        text = "\n\n".join(all_text_parts).strip()
                        return final_return(text)
                    else:
                        if inline_paragraph_lines:
                            return final_return(
                                "\n\n".join(inline_paragraph_lines).strip()
                            )
                        # TODO: Determine if this should be removed
                        self.print_warning(
                            "No poetry content found in any recognized format"
                        )
                        return final_return("")

            #####################################################################
            # STEP 6: Parse collected <div type="line"> elements
            #
            # Process each line element:
            #  - Insert stanza breaks when a new stanza-like parent is encountered.
            #  - Handle indentation if the raw HTML contains the &indent entity.
            #####################################################################
            lines = []
            found_content = True  # confirm we do have lines

            last_stanza_parent = None
            stanza_like_types = {"stanza", "versepara", "strophe", "epigraph"}

            for line_div in line_divs:
                stanza_parent = line_div.find_parent("div", {"type": stanza_like_types})
                if (
                    stanza_parent != last_stanza_parent
                    and last_stanza_parent is not None
                ):
                    lines.append("")
                last_stanza_parent = stanza_parent

                line_text = self.get_text_from_element(line_div)
                if "&indent" in line_div.decode():
                    indent_count = line_div.decode().count("&indent")
                    line_text = "\t" * indent_count + line_text.lstrip()
                lines.append(line_text)

            lines_text = "\n".join(lines)
            lines_text = re.sub(r"\n\s*\n", "\n\n", lines_text).strip()

            # combine inline paragraph lines (if any) with the standard line output
            if inline_paragraph_lines:
                combined_parts = inline_paragraph_lines + [lines_text]
                final_text = "\n\n".join(
                    part for part in combined_parts if part
                ).strip()
                return final_return(final_text)

            return final_return(lines_text if found_content else "")
        except Exception as e:
            print(f"Error extracting poetry text: {str(e)}", file=sys.stderr)
            print(traceback.format_exc())
            return ""

    def process_file(self, file_path: Path):
        """
        Process a single TML file: read it, parse metadata and poetry text, and return both.
        Parameters:
            file_path (Path): The path to the TML file.
        Returns:
            tuple: (metadata dictionary, poetry text string), or (None, None) on error.
        """
        metadata = None
        poetry_text = None
        self.current_file = file_path
        try:
            if self.verbose:
                print(f"\nProcessing: {file_path}")

            # reading with different encodings
            try:
                content, used_encoding = self.try_read_file(file_path)
                if used_encoding != "utf-8" and self.verbose:
                    print(
                        f"Note: File {file_path.name} was read using {used_encoding} encoding"
                    )
            except ValueError as e:
                print(
                    f"Error: Could not read {file_path.name} with any supported encoding",
                    file=sys.stderr,
                )
                print(f"Error details: {str(e)}", file=sys.stderr)
                return None, None

            # parse with BeautifulSoup
            # TODO: This is definitely not the right parser to use!
            soup = BeautifulSoup(content, "html.parser")
            # extract metadata from the header section
            metadata = self.extract_metadata(soup)
            metadata["filename"] = file_path.name
            # extract the poetry text from the body
            soup = BeautifulSoup(content, "lxml")
            poetry_text = self.extract_poetry_text(soup)

            # debug info for files with missing content
            if self.verbose:
                if not poetry_text or not any(
                    v for k, v in metadata.items() if k != "filename"
                ):
                    print(f"Debug info for {file_path.name}:")
                    print(f"Found poetry text: {bool(poetry_text)}")
                    print(
                        f"Found metadata: {bool(any(v for k, v in metadata.items() if k != 'filename'))}"
                    )
                    print("Document structure:")
                    print(f"- Has body tag: {bool(soup.find('body'))}")
                    print(
                        f"- Number of <div type='line'>: {len(soup.find_all('div', type='line'))}"
                    )
                    print(f"- Number of <p> tags: {len(soup.find_all('p'))}")
                    print(
                        f"- Has firstline divs: {bool(soup.find('div', type='firstline'))}"
                    )
        except Exception as e:
            print(f"Error processing {file_path}:", file=sys.stderr)
            print(f"Error type: {type(e).__name__}", file=sys.stderr)
            print(f"Error message: {str(e)}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            # Ensure these are set to None
            metadata, poetry_text = None, None
        # Unset current file
        self.current_file = None
        return metadata, poetry_text

    def process_directory(self, num_files: Optional[int] = None):
        """
        Processes the TML files within this parser's input directory of TML files.
        For each file, its metadata and poetry text are extracted with its metadata
        being written to the metadata CSV file and its poetry text being saved as a
        plaintext (.TXT) fiel toe the output directory. Optionally, a cut-off for
        the number of files to be processed can be provided.

        Parameters:
            num_files (int or None): The number of files to process. If None, process all files.
        """
        # Setup progress bar
        desc = "Processing TML files"
        tml_gen = self.input_dir.glob("*.tml")
        if num_files:
            file_progress = tqdm(
                tml_gen, desc=desc, total=num_files, disable=not self.show_progress
            )
        else:
            bar_format = (
                "{desc}: {n:,} files processed | elapsed: {elapsed}, {rate_fmt}"
            )
            file_progress = tqdm(tml_gen, desc=desc, disable=not self.show_progress)

        n_attempted = 0
        n_processed = 0
        failed_files = []
        # open CSV file for writing metadata
        with open(self.metadata_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metadata_fields)
            writer.writeheader()

            for i, file_path in enumerate(file_progress):
                if num_files and i == num_files:
                    # Exit early if limit reached
                    break

                metadata, poetry_text = self.process_file(file_path)
                if metadata and poetry_text:
                    # write metadata to CSV
                    writer.writerow(metadata)

                    # write poetry to text file
                    output_file = self.output_dir / f"{file_path.stem}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(poetry_text)
                    n_processed += 1
                else:
                    failed_files.append(file_path.name)
                n_attempted += 1
        print(
            f"\nProcessing complete!\nSuccessfully processed {n_processed} "
            + f"out of {n_attempted} files"
        )
        if failed_files:
            print(f"Failed to process {len(failed_files)} files:")
            for fname in failed_files:
                print(f"  - {fname}")

        # print a list of TML files that contained only a figure
        if self.figure_only:
            print("\nThe following files contained only a figure (no text):")
            for f in self.figure_only:
                print("  -", f)

    def extract_metadata(self, soup: Tag):
        """
        Extract metadata from the TML file's head section.
        Handles multiple authors (original and translator) as well as special cases
        such as anonymous authors with embedded time periods.
        Parameters:
            soup (BeautifulSoup): Parsed TML file.
        Returns:
            dict: A dictionary containing metadata fields.
        """
        metadata = {field: "" for field in self.metadata_fields}

        try:
            # author information
            authors = soup.find_all("author")
            orig_author = {}
            translator = {}
            anon_author = {}

            for author in authors:
                role = author.find("role")
                role_text = self.clean_text(role.string) if role else ""

                lname = (
                    self.clean_text(author.find("lname").string)
                    if author.find("lname")
                    else ""
                )
                fname = (
                    self.clean_text(author.find("fname").string)
                    if author.find("fname")
                    else ""
                )
                birth = (
                    self.clean_text(author.find("dob").string)
                    if author.find("dob")
                    else ""
                )
                death = (
                    self.clean_text(author.find("dod").string)
                    if author.find("dod")
                    else ""
                )

                author_info = {
                    "lastname": lname,
                    "firstname": fname,
                    "birth": birth,
                    "death": death,
                    "period": "",  # special handling for "Anon." cases with a period in fname...
                }

                # special handling: if the last name is "Anon." and the first name is actually a time range.
                if lname == "Anon." and re.match(r"^\d{3,4}(-\d{3,4})?$", fname):
                    author_info["firstname"] = ""
                    author_info["period"] = fname  # Move time range to new field

                if role_text == "orig.":
                    orig_author = author_info
                elif role_text == "trans.":
                    translator = author_info
                elif lname == "Anon.":
                    anon_author = author_info

            # assign extracted metadata to final dictionary
            if anon_author:
                metadata["author_lastname"] = "Anon."
                metadata["author_firstname"] = anon_author.get("firstname", "")
                metadata["author_birth"] = anon_author.get("birth", "")
                metadata["author_death"] = anon_author.get("death", "")
                metadata["author_period"] = anon_author.get("period", "")

            else:  # otherwise, use normal author handling
                metadata["author_lastname"] = orig_author.get("lastname", "")
                metadata["author_firstname"] = orig_author.get("firstname", "")
                metadata["author_birth"] = orig_author.get("birth", "")
                metadata["author_death"] = orig_author.get("death", "")

            metadata["transl_lastname"] = translator.get("lastname", "")
            metadata["transl_firstname"] = translator.get("firstname", "")
            metadata["transl_birth"] = translator.get("birth", "")
            metadata["transl_death"] = translator.get("death", "")

            # extract title and edition information
            title = soup.find("title")
            if title:
                metadata["title_id"] = title.get("id", "")
                metadata["title_main"] = (
                    self.clean_text(title.find("main").string)
                    if title.find("main")
                    else ""
                )
                metadata["title_sub"] = (
                    self.clean_text(title.find("sub").string)
                    if title.find("sub")
                    else ""
                )

                edition = title.find("edition")
                if edition:
                    metadata["edition_id"] = edition.get("id", "")
                    metadata["edition_text"] = self.clean_text(edition.string)

            # extract <period>, <genre>, and <rhymes> directly (parsing the meta-tag first doesn't work...)
            period_tag = soup.find("period")
            genre_tag = soup.find("genre")
            rhymes_tag = soup.find("rhymes")

            metadata["period"] = (
                self.clean_text(period_tag.get_text(strip=True)) if period_tag else ""
            )
            metadata["genre"] = (
                self.clean_text(genre_tag.get_text(strip=True)) if genre_tag else ""
            )
            metadata["rhymes"] = (
                self.clean_text(rhymes_tag.get_text(strip=True)) if rhymes_tag else ""
            )

        except Exception as e:
            print(f"Error extracting metadata: {str(e)}", file=sys.stderr)
            print(traceback.format_exc())

        return metadata


def main():
    parser_arg = argparse.ArgumentParser(
        description="Process ChadwychTML Poetry Files and extract metadata and poetry text."
    )
    parser_arg.add_argument(
        "input_dir",
        type=Path,
        help="Path to the input directory containing .TML files",
    )
    parser_arg.add_argument(
        "output_dir",
        type=Path,
        help="Path of output directory for extracted plaintext files (.TXT)",
    )
    parser_arg.add_argument(
        "output_csv",
        type=Path,
        help="Filename of output metadata file (.CSV)",
    )
    parser_arg.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Number of files to process. If not provided, process all files.",
    )
    parser_arg.add_argument(
        "--progress",
        help="Show progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    # TODO: Determine if this should be removed (or renamed to debug)
    parser_arg.add_argument("-v", "--verbose", action="store_true")

    args = parser_arg.parse_args()

    # Validate input directory
    if not args.input_dir.is_dir():
        print(
            f"ERROR: input directory {args.input_dir} does not exist", file=sys.stderr
        )
        sys.exit(1)

    # Create an instance of TMLPoetryParser using the command-line parameters
    parser_instance = TMLPoetryParser(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        show_progress=args.progress,
        verbose=args.verbose,
    )
    # Process files (if args.num_files is None, all files are processed)
    parser_instance.process_directory(num_files=args.num_files)


if __name__ == "__main__":
    main()
