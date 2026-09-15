\name{metal_device_information}
\alias{metal_device_information}
\title{List Available Metal Devices and Their Properties}
\description{
  Enumerate all Metal devices on the current system.
  Returns a list with one entry per device describing the device's properties.
}
\usage{
  metal_device_information()
}
\details{
  All devices on the current system conforming to the modern Metal specification will be enumerated.

  \code{\link{metal_is_available}} is available as a sentinel function that returns \code{TRUE} if this package's \code{configure} script was able to successfully compile a Metal test program.
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
    \item{\code{is_default}}{
      Logical: Indicating whether this device is the OS's default device.
    }
    \item{\code{has_unified_memory}}{
      Logical: Indicating whether this device used unified memory, i.e. an M4 or later Silicon device.
    }
    \item{\code{is_low_power}}{
      Logical: TBA.
    }
    \item{\code{is_removable}}{
      Logical: TBA.
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
      Integer: For now this is a sentinel NA, as this isn't a query-able quality in the Metal specification?
    }
    \item{\code{recommended_max_working_set_size_bytes}}{
      Numeric: Global memory size in bytes.
    }
    \item{\code{max_buffer_length_bytes}}{
      Numeric: Maximum work group size.
    }
    \item{\code{max_threads_per_threadgroup}}{
      Numeric: A vector indicating the max threads per threadgroup by dimension.
    }
    \item{\code{max_threadgroup_memory_length_bytes}}{
      Numeric: Indicates the largest single allocation the device permits.
    }
    \item{\code{registry_id}}{
      Numeric: device registry id.
    }
    \item{\code{location}}{
      Numeric: device location.
    }
    \item{\code{device_ptr}}{
      externalptr: pointer classed as \code{metal_device} wrapping the underlying Metal device handle.
    }
    \item{\code{framework}}{
      Character: In the case of this function, just populated with \code{metal}.
    }
  }
  If no OpenCL compliant devices are found, this function should return \code{NULL}.
}
\note{
  This R function is a relatively thin wrapper for the C-level \code{metal_available_devices}.
  If no Metal devices are detected on the system, i.e. the system is not Darwin, returns \code{NULL}
}
\examples{
  devices <- metal_device_information()
  str(devices)
}
\seealso{
  \code{\link{metal_is_available}}
}
