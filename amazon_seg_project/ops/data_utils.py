"""
Functions to read data files and perform sanity checks.
"""

import logging
import os
from typing import List, Set, Tuple
from dagster import op
from dagster_aws.s3 import S3Resource
from .s3_utils import list_objects


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


@op
def load_labeled_data(
    s3_resource: S3Resource,
    s3_bucket: str,
    prefix_data: str = "",
    prefix_labels: str = "",
    datafile_ext: str = "",
    labelfile_ext: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Read in lists of data files and their labels from an AWS S3 bucket

    Args:
        s3_resource: Dagster-AWS S3 resource
        s3_bucket: Name of S3 bucket
        prefix_data: Prefix path of data files in S3 bucket
        prefix_labels: Prefix path of data labels in S3 bucket
        datafile_ext: Data file extension (e.g., ".tif", ".txt", ".img", etc.)
        labelfile_ext: Label mask extension (e.g., ".tif", ".txt", ".img", etc.)
    
    Returns:
        data_files: List of data files
        label_files: List of data labels (e.g., segmentation masks, bounding boxes, etc.)
    """
    data_files_set = set(
        list_objects(s3_resource, s3_bucket, prefix_data, datafile_ext)
    )
    label_files_set = set(
        list_objects(s3_resource, s3_bucket, prefix_labels, labelfile_ext)
    )

    data_files, label_files = find_corresponding_files(data_files_set, label_files_set)

    return data_files, label_files
