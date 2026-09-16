# ardea TODO

## CONCEPTUAL / STRUCTURAL

- convert `opencl_is_available` to work on a `configure` sentinel like the `Metal` and `CUDA` equivalents
- build a `metal_exposed_device_count` to sit alongside `opencl_exposed_device_count` and `cuda_exposed_device_count`
- build companion `<framework>_devices_present` functions to work alongside `<framework>_is_available` functions that ask that at least one device is present
- users should not be surprised that they can compile with a framework present, but need a present device to successfully dispatch

## DOCUMENTATION

- `cuda_make_program` nvcc_flags documentation needs considerable expansion, especially for local device capability matching

## USAGE

- supported type checking needs to be brought up to speed
- need R level inspection of context objects - users need to be able to verify what device they've bound into a context

## HOUSEKEEPING

- documentation, vignettes, and testing
