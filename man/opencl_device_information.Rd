\name{opencl_device_information}
\alias{opencl_device_information}
\title{List Available OpenCL Devices and Their Properties}
\description{
  Enumerate all OpenCL devices found across all platforms on the current system.
  Returns a list with one entry per device describing the device's properties.
}
\usage{
  opencl_device_information()
}
\details{
  All devices on the current system (see \code{clGetPlatformIDs} and \code{clGetDeviceIDs} in the OpenCL specification) will be enumerated, including POCL enabled CPUs.
  A platform that fails to enumerate its devices is skipped rather than treated as a fatal error.

  \code{\link{opencl_is_available}} is available as a lightweight check of whether any device is present at all, before attempting retrieval of full details with this function.
}
\value{
  When available devices are present, an object of class \code{c("device_list", "list")}, where each element is itself is of class \code{c("alternative_device", "list")} and describes a single OpenCL compliant device with the following components:
  \describe{
    \item{\code{index}}{
      Integer: The flat device index, starting at 0, assigned in platform-major order.
      This index is used to select a device elsewhere in the package.
    }
    \item{\code{name}}{
      Character: The device name, see \code{CL_DEVICE_NAME}.
    }
    \item{\code{vendor}}{
      Character: The device vendor, see \code{CL_DEVICE_VENDOR}.
    }
    \item{\code{version}}{
      Character: The OpenCL version supported by the device, see \code{CL_DEVICE_VERSION}.
    }
    \item{\code{driver_version}}{
      Character: The vendor driver version, see \code{CL_DRIVER_VERSION}.
    }
    \item{\code{type}}{
      Character: an interpretation of \code{CL_DEVICE_TYPE}, currently one of the following.
      \itemize{
        \item{\code{CPU}}
        \item{\code{GPU}}
        \item{\code{ACCELERATOR}}
        \item{\code{CUSTOM}}
        \item{\code{UNKNOWN}} -- returned when the queried bitfield doesn't match a recognized device type constant.
      }
    }
    \item{\code{max_compute_units}}{
      Integer: The number of parallel compute units on the device, see \code{CL_DEVICE_MAX_COMPUTE_UNITS}.
    }
    \item{\code{global_mem_size}}{
      Numeric: Global memory size in bytes, see \code{CL_DEVICE_GLOBAL_MEM_SIZE}.
    }
    \item{\code{max_work_group_size}}{
      Numeric: Maximum work group size, see \code{CL_DEVICE_MAX_WORK_GROUP_SIZE}.
    }
    \item{\code{max_work_group_dimensions}}{
      Integer: The number of work dimensions the device is capable of, see \code{CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS}.
    }
    \item{\code{max_work_item_sizes}}{
      Numeric: A vector the length of the \code{max_work_group_dimensions}, see \code{CL_DEVICE_MAX_WORK_ITEM_SIZES}.
    }
    \item{\code{max_mem_alloc_size}}{
      Numeric: Indicates the largest single allocation the device permits, see \code{CL_DEVICE_MAX_MEM_ALLOC_SIZE}.
    }
    \item{\code{local_mem_size}}{
      Numeric: See \code{CL_DEVICE_LOCAL_MEM_SIZE}.
    }
    \item{\code{local_mem_type}}{
      Character: Describes the memory type of the device, see \code{CL_DEVICE_LOCAL_MEM_TYPE}.
    }
    \item{\code{available}}{
      Logical: A signal whether the device is available for function dispatch, i.e. not powered down, see \code{CL_DEVICE_AVAILABLE}.
    }
    \item{\code{extensions}}{
      Character: A space separated vector enumerating device extensions, see \code{CL_DEVICE_EXTENSIONS}.
    }
    \item{\code{device_id}}{
      externalptr: pointer classed as \code{opencl_device_id} wrapping the underlying OpenCL device handle.
    }
    \item{\code{platform_id}}{
      externalptr: pointer classed as \code{opencl_platform_id} wrapping the underlying OpenCL platform handle.
    }
    \item{\code{framework}}{
      Character: In the case of this function, just populated with \code{opencl}.
    }
  }
  If no OpenCL compliant devices are found, this function should return \code{NULL}.
}
\note{
  This R function is a relatively thin wrapper for the C-level \code{opencl_available_devices}, which will return an empty list if no OpenCL compliant devices are present. That empty list is signaled to \code{NULL} in R.
}
\examples{
  if (opencl_is_available()) {
    devices <- opencl_device_information()
    str(devices)
  }
}
\seealso{
  \code{\link{opencl_is_available}}
}
