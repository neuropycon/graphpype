"""
Support function for testing datasets
"""
import os

from graphpype.utils_tests import load_test_data


def test_load_test_data():
    """Test load_test_data"""
    data_path = load_test_data("data_nii_min")
    assert os.path.exists(data_path)
