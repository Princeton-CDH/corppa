import argparse
import csv
import os
import random
import re
import traceback  # for error reporting
from pathlib import Path

import ftfy
from bs4 import BeautifulSoup
from entity_map import entity_map as custom_entity_map
from tqdm import tqdm

# cmd to run the script: python tml_parser.py --input_dir "tml" --output_dir "tml_parsed"
#
# optional args: --num_files 1000  (leave out to process all files)
#                --verbose (for verbose output)

###############################################
# TML Parser for Chadwyck-Healey Poetry Files #
###############################################


class TMLPoetryParser:
    def __init__(self, input_dir, output_dir, verbose=True):
        """
        Initialize the TMLPoetryParser with input and output directories.

        Parameters:
            input_dir (str): Path to the directory containing TML files.
            output_dir (str): Path to the directory where parsed files and metadata will be saved.
            verbose (bool): If True, print progress and debug messages.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_file = self.output_dir / "metadata.csv"
        self.metadata_fields = [
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

        # use imported custom entity map (see: entity_map.py)
        self.entity_map = custom_entity_map
        self.verbose = verbose

        if self.verbose:
            print(f"Creating output directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # keep track of files that contain only a figure
        self.figure_only = []

    def try_read_file(self, file_path):
        """
        Attempt to read a file using multiple encodings.
        Parameters:
            file_path (Path): The path to the file.
        Returns:
            tuple: (content of file as string, encoding used)
        Raises:
            UnicodeDecodeError: If the file cannot be read with any of the attempted encodings.
        """
        encodings = ["utf-8", "latin-1", "windows-1252", "cp1252", "ascii"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read(), encoding
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError(
            f"Failed to read {file_path} with any of the attempted encodings: {encodings}"
        )

    def replace_entities(self, text):
        """
        Replace SGML/XML entities with their corresponding characters.
        Parameters:
            text (str): The text in which to replace entities.
        Returns:
            str: Text with entities replaced.
        """
        if not text:
            return ""

        # replace all known entities
        for entity, replacement in self.entity_map.items():
            # handle both with and without semicolon
            entity_with_semicolon = entity if entity.endswith(";") else f"{entity};"
            text = text.replace(entity_with_semicolon, replacement)
            text = text.replace(entity.rstrip(";"), replacement)

        return text

    def clean_text(self, text):
        """
        Clean and normalize text content.
        Steps:
            - Collapse multiple whitespace characters into a single space.
            - Fix common text encoding issues.
            - Strip leading/trailing whitespace.
            - Replace known entities.
        Parameters:
            text (str): The raw text to be cleaned.
        Returns:
            str: Cleaned and normalized text.
        """
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = ftfy.fix_text(text)
        text = text.strip()
        text = self.replace_entities(text)
        return text

    def get_text_from_element(self, element):
        """
        Extract text from an element while preserving inline formatting.
        This includes:
         - Converting italic spans to plain text.
         - Converting small caps (and similar) to uppercase.
         - Cleaning the final output text.
        Parameters:
            element (bs4.element.Tag): A BeautifulSoup element.
        Returns:
            str: The cleaned text content.
        """
        if not element:
            return ""

        # convert italic spans (e.g., <span class="italic">word</span> => word)
        for span in element.find_all("span", class_="italic"):
            span.replace_with(f"{span.get_text()}")

        # convert small caps or other fancy spans if needed, e.g. class="smcap", also handle 'smcapit' in the same way
        for span in element.find_all("span", {"class": ["smcap", "smcapit"]}):
            uppercase_text = span.get_text().upper()
            span.replace_with(uppercase_text)

        # now get the text!
        return self.clean_text(element.get_text())

    def extract_poetry_text(self, soup):
        """
        This is the big cheese.
        Extract the poetry text from the BeautifulSoup-parsed TML file.
        The extraction follows a cascading logic:
          1. Remove unwanted notes and copyright (we can still add to this, if needed)
          2. Collect concrete poem lines from <pre><sl> blocks
            (preserving formatting, these are concrete poems that use the arrangement of words to create a visual image).
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
                print("Warning: No body found in document")
                return ""

            # helper function: whenever we want to return "text",
            # merge with concrete_lines (if any) and then return
            def final_return(text_str):
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
                        raw_html = sl.decode_contents()
                        # remove any embedded tags but keep spacing
                        line_text = re.sub(r"<[^>]+>", "", raw_html)
                        # convert entities (&amp; -> & etc.)
                        line_text = self.replace_entities(line_text)
                        # strip trailing newlines
                        line_text = line_text.rstrip("\r\n")
                        if line_text.strip():
                            concrete_lines.append(line_text)
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
                            print(
                                "Warning: No poetry content found in any recognized format"
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
                        print(
                            "Warning: No poetry content found in any recognized format"
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
                        print(
                            "Warning: No poetry content found in any recognized format"
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
            print(f"Error extracting poetry text: {str(e)}")
            print(traceback.format_exc())
            return ""

    def process_file(self, file_path):
        """
        Process a single TML file: read it, parse metadata and poetry text, and return both.
        Parameters:
            file_path (Path): The path to the TML file.
        Returns:
            tuple: (metadata dictionary, poetry text string), or (None, None) on error.
        """
        try:
            if self.verbose:
                print(f"\nProcessing: {file_path}")

            # reading with different encodings
            try:
                content, used_encoding = self.try_read_file(file_path)
                if used_encoding != "utf-8":
                    print(
                        f"Note: File {file_path.name} was read using {used_encoding} encoding"
                    )
            except UnicodeDecodeError as e:
                print(
                    f"Error: Could not read {file_path.name} with any supported encoding"
                )
                print(f"Error details: {str(e)}")
                return None, None

            # parse with BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")

            # extract metadata from the header section
            metadata = self.extract_metadata(soup)
            metadata["filename"] = file_path.name
            # extract the poetry text from the body
            poetry_text = self.extract_poetry_text(soup)

            # debug info for files with missing content
            if not poetry_text or not any(
                v for k, v in metadata.items() if k != "filename"
            ):
                print(f"Debug info for {file_path.name}:")
                print(f"Found poetry text: {bool(poetry_text)}")
                print(
                    f"Found metadata: {bool(any(v for k, v in metadata.items() if k != 'filename'))}"
                )
                print(f"Document structure:")
                print(f"- Has body tag: {bool(soup.find('body'))}")
                print(
                    f"- Number of <div type='line'>: {len(soup.find_all('div', type='line'))}"
                )
                print(f"- Number of <p> tags: {len(soup.find_all('p'))}")
                print(
                    f"- Has firstline divs: {bool(soup.find('div', type='firstline'))}"
                )

            return metadata, poetry_text

        except Exception as e:
            print(f"Error processing {file_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(traceback.format_exc())
            return None, None

    def process_directory(self, num_files=1000):
        """
        Process a directory of TML files.
        This function selects up to num_files from the input directory, processes each file
        (extracting metadata and poetry text), writes the metadata to a CSV file, and saves the
        poetry text to individual text files in the output directory.
        Parameters:
            num_files (int or None): The number of files to process. If None, process all files.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        # get all TML files
        all_tml_files = list(self.input_dir.glob("*.tml"))
        total_files = len(all_tml_files)
        print(f"Found {total_files} total .tml files in {self.input_dir}")

        if num_files is None:
            selected_files = all_tml_files
            num_to_process = total_files
            print("Processing all files")
        else:
            num_to_process = min(num_files, total_files)
            selected_files = all_tml_files[:num_to_process]
            print(f"Selected {num_to_process} files for processing")

        processed_count = 0
        failed_files = []

        # open CSV file for writing metadata
        with open(self.metadata_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metadata_fields)
            writer.writeheader()

            for file_path in tqdm(
                selected_files, desc="Processing TML files", unit="file"
            ):
                metadata, poetry_text = self.process_file(file_path)

                if metadata and poetry_text:
                    # write metadata to CSV
                    writer.writerow(metadata)

                    # write poetry to text file
                    output_file = self.output_dir / f"{file_path.stem}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(poetry_text)
                    processed_count += 1
                else:
                    failed_files.append(file_path.name)

        print(f"\nProcessing complete!")
        print(
            f"Successfully processed {processed_count} out of {num_to_process} selected files"
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

        print(f"Metadata file created at: {self.metadata_file}")
        print(f"Processed poems can be found in: {self.output_dir}")

    def extract_metadata(self, soup):
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
                metadata["author_period"] = anon_author.get(
                    "period", ""
                )  # Store the time range

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
            print(f"Error extracting metadata: {str(e)}")
            print(traceback.format_exc())

        return metadata


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser(
        description="Process TML Poetry Files and extract metadata and poetry text."
    )
    parser_arg.add_argument(
        "--input_dir",
        type=str,
        default="poems",
        help="Path to the input directory containing TML files (default: 'poems')",
    )
    parser_arg.add_argument(
        "--output_dir",
        type=str,
        default="tml_parsed",
        help="Path to the output directory (default: 'tml_parsed')",
    )
    # Use a flag for verbose: if provided, verbose will be True; otherwise False.
    parser_arg.add_argument(
        "--verbose", action="store_true", help="Turn on verbose output (default: False)"
    )
    parser_arg.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Number of files to process. If not provided, process all files.",
    )

    args = parser_arg.parse_args()

    # Create an instance of TMLPoetryParser using the command-line parameters.
    parser_instance = TMLPoetryParser(
        input_dir=args.input_dir, output_dir=args.output_dir, verbose=args.verbose
    )
    # Process files (if args.num_files is None, all files are processed)
    parser_instance.process_directory(num_files=args.num_files)
