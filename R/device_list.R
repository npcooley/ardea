###### -- methods for the device_list and alternative_device S3 class ---------


###### -- NOTES ---------------------------------------------------------------
# return a list where each position is itself a named list containing pertinent
# information about an openCL device that was successfully discovered

###### -- FUNCTIONS -----------------------------------------------------------

# byte printing helper
# this is analogous to:
# format(structure(bytes,
#                  class = "object_size"),
#        units = "auto",
#        standard = "IEC")
.format_bytes <- function(bytes) {
  if (is.null(bytes) ||
      is.na(bytes) ||
      bytes < 0) {
    return("NA")
  }
  units <- c("B",
             "KB",
             "MB",
             "GB",
             "TB")
  i <- 1L
  while (bytes >= 1024 && i < length(units)) {
    bytes <- bytes / 1024
    i <- i + 1L
  }
  sprintf("%.2f %s",
          bytes,
          units[i])
}

# print device information
print.alternative_device <- function(x, ...) {
  cat(sprintf("<%s device> %s\n",
              x$framework,
              x$name))
  cat(sprintf("  type:                 %s\n",
              x$type))
  cat(sprintf("  compute units:        %d\n",
              as.integer(x$max_compute_units)))
  if (x$framework == "opencl") {
    cat(sprintf("  global memory:        %s\n",
                .format_bytes(x$global_mem_size)))
    cat(sprintf("  max work-group size:  %.0f\n",
                x$max_work_group_size))
    cat(sprintf("  max work-item sizes:  %s\n",
                paste(x$max_work_item_sizes,
                      collapse = " x ")))
  } else if (x$framework == "metal") {
    cat(sprintf("  recommended max working set size bytes:        %s\n",
                .format_bytes(x$recommended_max_working_set_size_bytes)))
    cat(sprintf("  max buffer length bytes:                       %.0f\n",
                x$max_buffer_length_bytes))
    cat(sprintf("  max threads per threadgroup:                   %s\n",
                paste(x$max_threads_per_threadgroup,
                      collapse = " x ")))
  }
  
  invisible(x)
}

print.device_list <- function(x, ...) {
  n <- length(x)
  if (n == 0) {
    cat("<device_list: no devices found>\n")
    return(invisible(x))
  }
  cat(sprintf("<device_list: %d device%s>\n\n",
              n,
              if (n == 1) {
                ""
              } else {
                "s"
              }))
  for (i in seq_len(n)) {
    cat(sprintf("[[%d]]\n", i))
    print(x[[i]], ...)
    if (i < n) cat("\n")
  }
  invisible(x)
}
