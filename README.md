ardea 0.0.1
================
Nicholas Cooley
2026-09-15

- [Introduction](#introduction)
- [Installation](#installation)
  - [OpenCL](#opencl)
  - [Metal](#metal)
  - [CUDA](#cuda)
  - [Other](#other)

# Introduction

*Currently under construction! CUDA capabilities are being actively
folded in and CRAN submission will happen once those capabilities are
working.*

This package is an attempt at building out infrastructure for users to
dispatch code directly to GPU devices in R. The package uses OpenCL as
an ‘parent’ capability requirement, allowing it to build on machines
regardless of device vendor, or on machines that do not have a device
present.

This package is currently designed to work wherever R works. During
installation, it probes the system for the ability to successfully
compile and run simple `Hello World` style device probes for target
frameworks and their devices. Capabilities for specific frameworks are
only compiled if the tooling for those frameworks is successfully
detected.

Because I do not have a Windows development environment, Windows is not
currently supported.

# Installation

`ardea` goes through a fair amount of gymnastics to detect capabilities
and manage installation. The `ARDEA_<FRAMEWORK>_<KEYWORD>` environment
variables that the configure script ingests if present are used *during
build*, so users can hypothetically build out containers that may land
on diverse resources where these can be re-set for the new environment.

## OpenCL

OpenCL is an open source standard for interacting with GPUs and other
alternative compute devices, regardless of vendor. On Macs, this
framework is not current, and does not appear to be updateable *but does
work still*. On Linux systems, default OpenCL installations should be
detected automatically over the course of package installation. If they
are not, or users have a non-standard installation, `ardea` checks for
user environment variables:

- **ARDEA_OPENCL_LIBS**
- **ARDEA_OPENCL_INCLUDE**

to specify library and linker locations.

## Metal

Metal is Apple’s framework for interacting with GPUs and builtin devices
within their ecosystem. If the package detects Darwin as the OS, Metal
detection is triggered. Metal has gone through a few changes as Apple
has used a few different device vendors over the years and though its
most recent iteration appears focused on Silicon series chips.

Metal capabilities in this package were originally built out in the
[ACFmetal](https://github.com/npcooley/ACFmetal) github only R package.

## CUDA

CUDA is NVIDIA’s framework for interacting with NVIDIA devices. Because
CUDA installations have so many different flavors and many of their own
peculiarities, CUDA support *requires* user environment variables:

- **ARDEA_CUDA_LIBS**
- **ARDEA_CUDA_STUBS**
- **ARDEA_CUDA_INCLUDE**

These most point to the library, linker, and stubs directories of a
functioning CUDA installation, and CUDA’s `NVCC` compiler must be
present in R’s `PATH`. Package installation will check for successful
compilation, and CUDA capabilities will be enabled if *compilation* for
the probe programs is successful, an existing device is not required.

CUDA capabilities in this package were originally built out in the
[ACFcuda](https://github.com/npcooley/ACFcuda) github only R package.

## Other

There are at least two other major frameworks (for AMD and Intel
devices) that may be added in the future, should folks find this package
to be actually useful. Right now the focus for this package is on
hardening the codebase to avoid build errors and bugs, and ensuring that
functionalities are clearly described and accessible to typical R users.
