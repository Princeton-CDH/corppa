# compile the found poem dataset

from corppa.config import get_config
from corppa.poetry_detection.ref_corpora import save_poem_metadata


def main():
    config_opts = get_config()
    required_sections = ["poem_dataset", "reference_corpora"]
    for section in required_sections:
        if section not in config_opts:
            print(f"Required configuration for '{section}' not found in config file")

    # TODO: print("#### Merging excerpts")

    print("\n## Compiling reference corpora metadata")
    save_poem_metadata()

    # TODO: print("#### PPA work-level metadata")


if __name__ == "__main__":
    main()
