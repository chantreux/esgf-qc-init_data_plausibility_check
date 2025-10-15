#!/usr/bin/env python
"""
Test for check_duplicate_timesteps.py 
"""

from compliance_checker.base import BaseCheck
from compliance_checker.checks.data_plausibility_checks import check_duplicate_timesteps as checker
from compliance_checker.tests import BaseTestCase
from compliance_checker.tests.resources import STATIC_FILES


class TestDuplicateTimesteps(BaseTestCase):

    # PASS CASE: no duplicate consecutive timesteps expected
    def test_no_duplicate_timesteps(self):
        dataset = self.load_dataset(STATIC_FILES["data_check_reference"]) 
        variable = "tas"
        output = checker.check_duplicate_timesteps(dataset, variable, severity=BaseCheck.MEDIUM)
        results = output.to_result()
        assert results is not None
        self.assert_result_is_good(results)

    # FAIL CASE: constant timeseries should have duplicates at every consecutive pair
    def test_duplicate_timesteps_detected(self):
        dataset = self.load_dataset(STATIC_FILES["data_check_reference_duplicate_timesteps"]) 
        variable = "tas"
        output = checker.check_duplicate_timesteps(dataset, variable, severity=BaseCheck.MEDIUM)
        results = output.to_result()
        assert results is not None
        self.assert_result_is_bad(results)

    # PASS CASE (parallel): no duplicate consecutive timesteps expected
    def test_no_duplicate_timesteps_parallel(self):
        dataset = self.load_dataset(STATIC_FILES["data_check_reference"]) 
        variable = "tas"
        output = checker.check_duplicate_timesteps_parallel(dataset, variable, severity=BaseCheck.MEDIUM, workers=2)
        results = output.to_result()
        assert results is not None
        self.assert_result_is_good(results)

    # FAIL CASE (parallel): constant timeseries should have duplicates
    def test_duplicate_timesteps_detected_parallel(self):
        dataset = self.load_dataset(STATIC_FILES["data_check_reference_duplicate_timesteps"]) 
        variable = "tas"
        output = checker.check_duplicate_timesteps_parallel(dataset, variable, severity=BaseCheck.MEDIUM, workers=2)
        results = output.to_result()
        assert results is not None
        self.assert_result_is_bad(results)
