\name{cuda_device_information}
\alias{cuda_device_information}
\title{List Available CUDA Devices and Their Properties}
\description{
  Enumerate all visible CUDA devices on the current system.
  Returns a list with one entry per device describing the device's properties.
}
\usage{
  cuda_device_information()
}
\details{
  All devices visible to \code{cudaGetDeviceCount()}/\code{cudaGetDeviceProperties()} on the current system are enumerated.

  \code{\link{cuda_is_available}} is available as a sentinel function that returns \code{TRUE} if this package's \code{configure} script was able to successfully compile and link a CUDA test program.

  Unlike \code{\link{metal_device_information}}, there is no \code{device_ptr} field here, and no \code{is_default} field -- CUDA identifies devices by a plain integer ordinal rather than an opaque handle, and has no inherent default-device concept the way \code{MTLCreateSystemDefaultDevice()} does (\code{cudaGetDevice()} only reports whichever device happens to be current at the moment of the call, which is arbitrary at enumeration time, not a hardware property worth reporting).
}
\value{
  When available devices are present, an object of class \code{c("device_list", "list")}, where each element is itself of class \code{c("alternative_device", "list")} and describes a single CUDA compliant device with the following components:
  \describe{
    \item{\code{index}}{
      Integer: The device's ordinal, starting at 0, as reported by \code{cudaGetDeviceCount()}. Identical to \code{device_index}.
    }
    \item{\code{device_index}}{
      Integer: The device ordinal used to select this device elsewhere in the package, e.g. by \code{\link{cuda_make_context}}. Identical to \code{index}, provided under this name for consistency with the \code{device_ptr}-style field other frameworks expose.
    }
    \item{\code{name}}{
      Character: The device name, see \code{cudaDeviceProp::name}.
    }
    \item{\code{total_global_mem_bytes}}{
      Numeric: Total global memory size in bytes, see \code{cudaDeviceProp::totalGlobalMem}.
    }
    \item{\code{multiprocessor_count}}{
      Integer: The number of streaming multiprocessors on the device, see \code{cudaDeviceProp::multiProcessorCount}.
    }
    \item{\code{max_threads_per_block}}{
      Integer: The device-level ceiling on threads per block, see \code{cudaDeviceProp::maxThreadsPerBlock}. This is a per-\emph{device} value; the per-\emph{kernel} ceiling actually used for dispatch is queried and cached separately by \code{\link{cuda_make_kernelptr}}.
    }
    \item{\code{warp_size}}{
      Integer: The device's warp size, see \code{cudaDeviceProp::warpSize}.
    }
    \item{\code{compute_capability_major}}{
      Integer: Major compute capability version, see \code{cudaDeviceProp::major}.
    }
    \item{\code{compute_capability_minor}}{
      Integer: Minor compute capability version, see \code{cudaDeviceProp::minor}.
    }
    \item{\code{framework}}{
      Character: In the case of this function, just populated with \code{cuda}.
    }
    \item{\code{type}}{
      Character: Always \code{GPU} for CUDA devices.
    }
    \item{\code{max_compute_units}}{
      Integer: Populated from \code{multiprocessor_count}, provided under this name for consistency with the field \code{\link{opencl_device_information}}/\code{\link{metal_device_information}} expose.
    }
  }
  If no CUDA compliant devices are found, this function returns \code{NULL}.
}
\note{
  This R function is a relatively thin wrapper for the C-level \code{cuda_available_devices}.
  If no CUDA devices are detected on the system, i.e. \code{\link{cuda_is_available}} returns \code{FALSE}, this function returns \code{NULL}.
}
\examples{
  devices <- cuda_device_information()
  str(devices)
}
\seealso{
  \code{\link{cuda_is_available}},
  \code{\link{cuda_make_context}}
}
