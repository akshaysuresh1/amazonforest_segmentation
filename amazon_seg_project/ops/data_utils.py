"""
Functions to read data files and perform sanity checks.
"""

import logging
import os
from typing import List, Set, Tuple
from dagster import op


@op
def find_corresponding_files(
    data_files: Set[str],
    ref_files: Set[str],
) -> Tuple[List[str], List[str]]:
    """
    Find corresponding reference files for each data file based on common basenames.

    Args:
        data_files: Set of data files (e.g., .tif images stored in a directory)
        ref_files: Set of reference files (e.g., binary segmentation masks corresponding)

    Returns:
        matched_data_files: List of data files with matched reference labels
        matched_reference_files: List of reference labels for matched data files
    """
    # Map basenames to their respective datafiles.
    data_map = {
        os.path.splitext(os.path.basename(datafile))[0]: datafile
        for datafile in data_files
    }
    # Map basenames to their respective reference files.
    ref_map = {
        os.path.splitext(os.path.basename(reffile))[0]: reffile for reffile in ref_files
    }

    # Lists to hold matched files
    matched_data_files = []
    matched_reference_files = []

    # Check for one-to-one correspondence
    for basename in sorted(data_map.keys()):
        if basename in ref_map:
            matched_data_files.append(data_map[basename])
            matched_reference_files.append(ref_map[basename])

    if len(matched_data_files) != len(data_files):
        logging.warning("One or more data files have no reference labels.")

    return matched_data_files, matched_reference_files
