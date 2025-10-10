#!/usr/bin/env python
"""
check_constants.py

Check if the specified netCDF dataset has constant values in the data along specific dimensions.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx
import numpy as np

from compliance_checker.checks.data_plausibility_checks.utils.dimensions import get_filtered_dimensions
from compliance_checker.checks.data_plausibility_checks.utils.data import check_variable_conditions
from compliance_checker.checks.data_plausibility_checks.utils.auxiliar import (
                        ExtendedTestCtx,
                        dump_data_file_extended,
                        Coordinate)

def check_all_constant(data_slice):
    if np.isscalar(data_slice):
        return True
    elif data_slice.size == 0:
        return False
    else:
        # If it's an array, check if all values are equal to the first one
        return np.all(data_slice == data_slice.flat[0])




def check_constants(dataset, variable, severity=BaseCheck.MEDIUM):
    """
    Check for constant values in a dataset.
    Uses ExtendedTestCtx to store detailed results.
    """
    ctx = ExtendedTestCtx(
        category=severity,
        description="Check for constant values in the dataset.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_constants",
        parameters={},
        variable=variable,
    )

    check_dims = get_filtered_dimensions(dataset, variable)
    values = check_variable_conditions(dataset, variable, check_dims, check_all_constant)
    detected = [(coord, val) for coord, val in values if bool]
    if len(detected) > 0:
        for coord, value in detected:
            coord_obj = Coordinate(
                name="constant_values",
                indices=[coord],
                values=[value],
                result=True
            )
            ctx.coordinates.append(coord_obj)

        num_constants = len(detected)
        ctx.add_failure(f"Constant values detected: {num_constants}")
        dump_data_file_extended(dataset, variable, 'check_constant', ctx)
    else:
        ctx.add_pass()
        ctx.messages.append("No constant values detected in the dataset.")

    return ctx