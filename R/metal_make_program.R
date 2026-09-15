###### -- compile a .metal source file into a dispatch-ready library ----------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# two compilation paths, selected by whether 'metallib_file' is supplied:
#
# metallib_file == NULL (default):
#   compile at runtime via MTLDevice's newLibraryWithSource:options:error:.
#   'fast_math' and 'language_version' configure this path's MTLCompileOptions.
#
# metallib_file != NULL:
#   shell out to `xcrun metal` to compile 'metal_file' into a .metallib at
#   'metallib_file', then load that compiled library. this exists mainly
#   because newLibraryWithSource: compiles from an in-memory string with no
#   filesystem search-path context, so '#include' of separate header files
#   is not reliably supported there -- `xcrun metal`, compiling a real file
#   on disk, does not have that limitation. 'fast_math'/'language_version'
#   have no effect on this path -- xcrun's own flags would control that,
#   and this function doesn't currently expose them (see the 'xcrun flags'
#   discussion -- MTLCompileOptions itself only covers a narrow, fixed set
#   of settings, so there was little to gain by re-threading xcrun's much
#   larger flag surface through this function's argument list)
#
# both paths return the identical thing: an external pointer of class
# "metal_library". which compiler produced it is invisible past this
# function -- metal_make_kernelptr() works the same either way

###### -- FUNCTION ------------------------------------------------------------

metal_make_program <- function(metal_file,
                               context,
                               metallib_file = NULL,
                               xcrun_flags = NULL,
                               fast_math = FALSE,
                               language_version = NULL) {
  if (length(metal_file) != 1 || !is.character(metal_file)) {
    stop("'metal_file' must be a character vector of length 1")
  }
  if (!file.exists(metal_file)) {
    stop("'metal_file' must exist")
  }
  if (!is(object = context,
          class2 = "metal_context")) {
    stop("'context' must be a 'metal_context' object created by 'metal_make_context()'")
  }
  
  if (is.null(metallib_file)) {
    
    ###### -- path 1: compile from source via the Metal runtime API -------
    
    if (length(fast_math) != 1 || !is.logical(fast_math) || is.na(fast_math)) {
      stop("'fast_math' must be a single non-NA logical value")
    }
    if (!is.null(language_version) &&
        (length(language_version) != 1 || !is.character(language_version))) {
      stop("'language_version' must be NULL or a character vector of length 1")
    }
    
    res <- .Call("metal_program_from_source",
                 metal_file,
                 context,
                 fast_math,
                 language_version,
                 PACKAGE = "ardea")
    
  } else {
    
    ###### -- path 2: compile via 'xcrun metal', then load the result -----
    
    if (length(metallib_file) != 1 || !is.character(metallib_file)) {
      stop("'metallib_file' must be a character vector of length 1")
    }
    if (!identical(fast_math, FALSE) || !is.null(language_version)) {
      warning("'fast_math' and 'language_version' are MTLCompileOptions ",
              "settings and have no effect when 'metallib_file' is supplied ",
              "-- compilation is performed by 'xcrun metal' instead, whose ",
              "own flags aren't currently exposed by this function")
    }
    if (!is.null(xcrun_flags)) {
      if (!is.character(xcrun_flags) || anyNA(xcrun_flags)) {
        stop("'xcrun_flags' must be NULL or a character vector with no NA elements")
      }
      if (any(!nzchar(xcrun_flags))) {
        stop("'xcrun_flags' must not contain empty-string elements -- ",
             "omit an unused flag entirely rather than passing \"\"")
      }
      if (any(xcrun_flags == "-o")) {
        stop("'xcrun_flags' must not include '-o' -- this function already ",
             "supplies it from 'metallib_file'; a second '-o' would silently ",
             "compete with it rather than error, since the compiler honors ",
             "whichever '-o' appears last")
      }
    }
    
    # xcrun -sdk macosx metal <metal_file> -o <metallib_file>
    #
    # arguments are passed to system2() as separate, individually-quoted
    # elements rather than one paste()'d string. system2() still joins
    # them with spaces internally before handing the line to the shell, so
    # unquoted paths containing spaces or shell metacharacters would still
    # be misparsed by the shell if passed as one flat string -- shQuote()
    # on each path-bearing element is what actually protects against that,
    # not the vector structure alone
    xcrun_args <- c("-sdk",
                    "macosx",
                    "metal",
                    if (!is.null(xcrun_flags)) {
                      shQuote(xcrun_flags)
                    },
                    shQuote(metal_file),
                    "-o",
                    shQuote(metallib_file))
    
    # wait = TRUE (the default) matters here, not cosmetically: this blocks
    # R until xcrun exits, which we require -- proceeding to load
    # 'metallib_file' before the compiler has finished writing it would be
    # a race, not a "probably fine" shortcut
    exit_status <- system2(command = "xcrun",
                           args = xcrun_args,
                           wait = TRUE)
    if (!identical(exit_status, 0L) && !identical(exit_status, 0)) {
      stop("'xcrun metal' failed to compile '",
           metal_file,
           "' (exit status ",
           exit_status, ")")
    }
    if (!file.exists(metallib_file)) {
      stop("'xcrun metal' reported success but '",
           metallib_file,
           "' was not created")
    }
    
    res <- .Call("metal_program_from_metallib",
                 metallib_file,
                 context,
                 PACKAGE = "ardea")
  }
  
  return(res)
}

