#!/usr/bin/env python3
"""
remove_string.py

Recursively rename files in a directory by removing all occurrences of a given substring
from each file’s name.

Usage:
    python3 remove_string.py /path/to/target_dir "substring_to_remove"

This script performs the following steps:
    1. Walks the directory tree rooted at the specified path.
    2. For each file or directory whose name contains the target substring:
        a. Compute the new name by removing all occurrences of that substring.
        b. Rename the file or directory in place.
    3. Continues until all nested levels have been processed.

Example:
    python3 remove_string.py ~/Downloads "_copy"
    → Renames "photo_copy.jpg" to "photo.jpg" and so on for all matching names.
"""

import os
import sys
import argparse


def parse_arguments():
    """
    Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: An object with attributes:
            - root_dir (str): Path to the directory to process.
            - target (str): Substring to remove from file and directory names.
    """
    parser = argparse.ArgumentParser(
        description="Recursively remove a specified substring from all filenames."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to scan recursively."
    )
    parser.add_argument(
        "target",
        type=str,
        help="Substring to remove from each filename."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        parser.error(f"Specified path is not a directory: {args.root_dir}")

    return args


def remove_substring_in_name(name: str, target: str) -> str:
    """
    Remove all occurrences of `target` from `name`.

    Args:
        name (str): Original filename or directory name.
        target (str): Substring to remove.

    Returns:
        str: New name with all occurrences of `target` removed.
    """
    return name.replace(target, "")


def rename_entries(root_dir: str, target: str):
    """
    Walk through `root_dir` recursively, renaming files and directories by removing `target`.

    Args:
        root_dir (str): The base directory to process.
        target (str): Substring to remove from each entry’s name.

    Behavior:
        - Uses os.walk with `topdown=False` to ensure child entries are renamed before their parents.
        - Skips any entry whose new name would be identical to the old name.
        - Prints each rename operation to stdout.
    """
    # Walk from the bottom of the tree upward to rename nested items first
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        # Process files
        for filename in filenames:
            if target in filename:
                old_path = os.path.join(dirpath, filename)
                new_name = remove_substring_in_name(filename, target)
                new_path = os.path.join(dirpath, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} → {new_path}")

        # Process directories
        for dirname in dirnames:
            if target in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_name = remove_substring_in_name(dirname, target)
                new_path = os.path.join(dirpath, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed dir:  {old_path} → {new_path}")


def main():
    """
    Entry point: parse arguments and perform rename operations.
    """
    args = parse_arguments()
    rename_entries(args.root_dir, args.target)


if __name__ == "__main__":
    main()
