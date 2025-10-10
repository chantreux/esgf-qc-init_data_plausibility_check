#!/usr/bin/env python
"""
check_duplicate_timesteps.py

Check if two consecutive timesteps in a variable have identical values.

Intended to be included in the WCRP plugins.
"""

from compliance_checker.base import BaseCheck, TestCtx
import numpy as np
import xarray as xr
import multiprocessing
import multiprocessing.shared_memory as shm
from concurrent.futures import ProcessPoolExecutor

from compliance_checker.checks.data_plausibility_checks.utils.dimensions import get_filtered_dimensions, get_dimension_info
from compliance_checker.checks.data_plausibility_checks.utils.auxiliar import (
    ExtendedTestCtx,
    dump_data_file_extended,
    Coordinate
    )
def _compare_timesteps(var, time_dim_index, t):
    """
    Función auxiliar para comparar timestep t y t+1.
    Devuelve (t, t+1) si son idénticos, None en caso contrario.
    """
    slicer_t = [slice(None)] * var.ndim
    slicer_tplus1 = [slice(None)] * var.ndim
    slicer_t[time_dim_index] = t
    slicer_tplus1[time_dim_index] = t + 1

    data_t = np.asanyarray(var[tuple(slicer_t)])
    data_tplus1 = np.asanyarray(var[tuple(slicer_tplus1)])

    if np.array_equal(data_t, data_tplus1):
        return (t, t + 1)
    return None
    
def _worker(shared_name, shape, dtype, t):
    existing_shm = shm.SharedMemory(name=shared_name)
    data = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    result = None
    if np.array_equal(data[t], data[t+1]):
        result = (t, t+1)
    existing_shm.close()
    return result



def check_duplicate_timesteps(dataset, variable, severity=BaseCheck.MEDIUM):
    """
    Check if two consecutive time steps in a variable have identical values,
    ignoring latitude and longitude dimensions. Uses ExtendedTestCtx to store
    detailed coordinate-level results.
    """
    ctx = ExtendedTestCtx(
        category=severity,
        description="Check for identical consecutive timesteps.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_duplicate_timesteps",
        parameters={},
        variable=variable,
    )

    var = dataset.variables[variable]

    try:
        dim_info = get_dimension_info(dataset, variable)
    except ValueError as e:
        ctx.add_failure(str(e))
        return ctx

    if 'time_dim' not in dim_info:
        ctx.add_failure(f"No time dimension found in variable '{variable}'.")
        return ctx

    time_dim_index = dim_info['time_dim']['i']
    num_timesteps = var.shape[time_dim_index]

    if num_timesteps < 2:
        ctx.add_pass()
        ctx.messages.append("Only one timestep found; nothing to compare.")
        return ctx

    filtered_dims = get_filtered_dimensions(dataset, variable)

    duplicates = []
    for t in range(num_timesteps - 1):
        slicer_t = [slice(None)] * var.ndim
        slicer_tplus1 = [slice(None)] * var.ndim
        slicer_t[time_dim_index] = t
        slicer_tplus1[time_dim_index] = t + 1

        data_t = np.asanyarray(var[tuple(slicer_t)])
        data_tplus1 = np.asanyarray(var[tuple(slicer_tplus1)])

        if np.array_equal(data_t, data_tplus1):
            duplicates.append((t, t + 1))
            
            coord_obj = Coordinate(
                name="duplicate_timestep",
                indices=[(t, t+1)],
                values=[True],
                result=True
            )
            ctx.coordinates.append(coord_obj)

    if ctx.coordinates:
        ctx.add_failure(f"Identical values found in {len(duplicates)} consecutive timestep pairs.")
        dump_data_file_extended(dataset, variable, 'check_duplicate_timesteps', ctx)
    else:
        ctx.add_pass()
        ctx.messages.append("No identical consecutive timesteps found.")

    return ctx


def check_duplicate_timesteps_extended_paralel(dataset, variable, severity=BaseCheck.MEDIUM):
    """
        Check if two consecutive time steps in a variable have identical values,
        ignoring latitude and longitude dimensions. Uses ExtendedTestCtx to store
        detailed coordinate-level results.
    """
    ctx = ExtendedTestCtx(
        category=severity,
        description="Check for identical consecutive timesteps.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_duplicate_timesteps",
        parameters={},
        variable=variable,
    )

    var = dataset.variables[variable]

    try:
        dim_info = get_dimension_info(dataset, variable)
    except ValueError as e:
        ctx.add_failure(str(e))
        return ctx

    if "time_dim" not in dim_info:
        ctx.add_failure(f"No time dimension found in variable '{variable}'.")
        return ctx

    time_dim = dim_info["time_dim"]["name"]

    da = xr.DataArray(var[:], dims=var.dimensions)

    if da.sizes[time_dim] < 2:
        ctx.add_pass()
        ctx.messages.append("Only one timestep found; nothing to compare.")
        return ctx

    diffs = da.diff(dim=time_dim)
    equal_mask = (diffs == 0).all(dim=[d for d in da.dims if d != time_dim])

    duplicates = list(np.where(equal_mask.values)[0])

    for i in duplicates:
        coord_obj = Coordinate(
            name="duplicate_timestep",
            indices=[(i, i+1)],
            values=[True],
            result=True,
        )
        ctx.coordinates.append(coord_obj)

    if ctx.coordinates:
        ctx.add_failure(
            f"Identical values found in {len(duplicates)} consecutive timestep pairs."
        )
        dump_data_file_extended(dataset, variable, "check_duplicate_timesteps", ctx)
    else:
        ctx.add_pass()
        ctx.messages.append("No identical consecutive timesteps found.")

    return ctx


def check_duplicate_timesteps_extended_classic(dataset, variable, severity=BaseCheck.MEDIUM, workers=4):
    ctx = ExtendedTestCtx(
        category=severity,
        description="Check for identical consecutive timesteps.",
        dataset_name=getattr(dataset, "filepath", lambda: "unknown")(),
        test_function="check_duplicate_timesteps",
        parameters={},
        variable=variable,
    )

    var = dataset.variables[variable]
    try:
        dim_info = get_dimension_info(dataset, variable)
    except ValueError as e:
        ctx.add_failure(str(e))
        return ctx
    if "time_dim" not in dim_info:
        ctx.add_failure(f"No time dimension found in variable '{variable}'.")
        return ctx

    time_dim_index = dim_info["time_dim"]["i"]
    num_timesteps = var.shape[time_dim_index]
    if num_timesteps < 2:
        ctx.add_pass()
        ctx.messages.append("Only one timestep found; nothing to compare.")
        return ctx

    arr = np.asanyarray(var[:])
    arr = np.moveaxis(arr, time_dim_index, 0)

    shared = shm.SharedMemory(create=True, size=arr.nbytes)
    shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shared.buf)
    shared_arr[:] = arr[:]

    duplicates = []
    try:
        # paralellization
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(_worker, shared.name, arr.shape, arr.dtype, t)
                       for t in range(num_timesteps - 1)]
            for f in futures:
                res = f.result()
                if res is not None:
                    duplicates.append(res)
                    coord_obj = Coordinate(
                        name="duplicate_timestep",
                        indices=[res],
                        values=[True],
                        result=True,
                    )
                    ctx.coordinates.append(coord_obj)
    finally:
        shared.close()
        shared.unlink()

    if ctx.coordinates:
        ctx.add_failure(
            f"Identical values found in {len(duplicates)} consecutive timestep pairs."
        )
        dump_data_file_extended(dataset, variable, "check_duplicate_timesteps", ctx)
    else:
        ctx.add_pass()
        ctx.messages.append("No identical consecutive timesteps found.")

    return ctx