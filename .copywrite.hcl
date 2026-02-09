schema_version = 1

project {
  license        = "Apache-2.0"
 # copyright_year = 2025  # NOTE: this doesn't seem to be used anywhere
  copyright_holder = "2024,2025 Center for Digital Humanities, Princeton University"

  # (OPTIONAL) A list of globs that should not have copyright/license headers.
  # Supports doublestar glob patterns for more flexibility in defining which
  # files or folders should be ignored
  header_ignore = [
    "docs/build/**"
    # "vendor/**",
    # "**autogen**",
  ]
}
