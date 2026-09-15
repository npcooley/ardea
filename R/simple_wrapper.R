###### -- peform simple function dispatch for a selected specific framework ---
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------

# hypothetically this single R function should be able to power all three frameworks
# even though their verbiage is ... pretty different per framework, they should
# all rely on the same set of inputs to power function dispatch in the relatively
# narrow use case that this wrapper and the underlying runners target

# argument names here are intentionally set apart from framework specific
# vocabulary and are attempted to be named in a way that relates as closely
# as possible to the framework concepts that they propagate to
# i.e.
# problem_dims ->
#  opencl == x
#  metal == y
#  cuda == z
# group_dims ->
#  opencl == x
#  metal == y
#  cuda == z
# workers_per ->
#  opencl == x
#  metal == y
#  cuda == z

###### -- FUNCTION ------------------------------------------------------------

simple_wrapper <- function(framework = c("opencl",
                                         "metal",
                                         "cuda"),
                           context_ptr,
                           kernel_ptr,
                           arg_types,
                           arg_list,
                           problem_dims,
                           group_dims,
                           workers_per = NULL) {
  
  framework <- match.arg(framework)
  
  # specific overheads
  if (framework == "opencl") {
    if (!opencl_is_available()) {
      stop("OpenCL compliant devices must exist and be detectable.")
    }
    if(!is(object = context_ptr,
           class2 = "opencl_context")) {
      stop("'context_ptr' must be an 'opencl_context' created by 'opencl_make_context()'")
    }
    if(!is(object = kernel_ptr,
           class2 = "opencl_kernel")) {
      stop("'kernel_ptr' must be an 'opencl_kernel' created by 'opencl_make_kernelptr()'")
    }
  } else if (framework == "metal") {
    if (!metal_is_available()) {
      stop("metal compliant devices do not appear to be available")
    }
    if (!is(object = context_ptr,
            class2 = "metal_context")) {
      stop("'context_ptr' must be a 'metal_context' created by 'metal_make_context()'")
    }
    if (!is(object = kernel_ptr,
            class2 = "metal_pipeline")) {
      stop("'kernel_ptr' must be a 'metal_pipeline' created by 'metal_make_kernelptr()'")
    }
  } else if (framework == "cuda") {
    stop("cuda is not yet implemented")
  } else {
    stop("unrecognized framework")
  }
  # non-specific overheads
  if (length(arg_list) != length(arg_types)) {
    stop("'arg_list' and 'arg_types' must be the same length")
  }
  if (length(problem_dims) != 3) {
    stop("problem dims expects a vector of length three")
  }
  
  
  res <- switch(framework,
                opencl = .Call("opencl_simple_runner",
                               context_ptr,
                               kernel_ptr,
                               arg_types,
                               arg_list,
                               problem_dims,
                               group_dims,
                               workers_per,
                               PACKAGE = "ardea"),
                metal = .Call("metal_simple_runner",
                              context_ptr,
                              kernel_ptr,
                              arg_types,
                              arg_list,
                              problem_dims,
                              group_dims,
                              workers_per,
                              PACKAGE = "ardea"),
                cuda = stop("framework not implemented"))
  
  return(res)
}

