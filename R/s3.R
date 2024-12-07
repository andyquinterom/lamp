#' Create a New CUDA Device
#'
#' Initializes and returns a new CUDA device object for GPU programming.
#'
#' @param ordinal An integer specifying the ordinal index of the CUDA device.
#'
#' @return A CUDA device object.
#' @export
new_cuda <- function(ordinal) {
  res <- int_new_cuda(ordinal)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Create a New Tensor
#'
#' Constructs a tensor object using the specified array, dimensions, and device.
#'
#' @param array A numeric vector or array to initialize the tensor.
#' @param dims An integer vector specifying the dimensions of the tensor.
#' @param device A CUDA device object to associate with the tensor.
#'
#' @return A tensor object.
#' @export
new_tensor <- function(array, dims, device) {
  res <- int_new_tensor(array, dims, device)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Convert an Object to a Tensor
#'
#' Generic function to convert different data types to a tensor.
#'
#' @param array The input data to be converted to a tensor.
#' @param device A CUDA device object to associate with the tensor.
#'
#' @return A tensor object.
#' @export
as_tensor <- function(array, device) {
  UseMethod("as_tensor")
}

#' Convert a Matrix to a Tensor
#'
#' Converts a matrix to a tensor. Note: This performs an implicit transpose of the matrix.
#'
#' @param array A matrix to be converted to a tensor.
#' @param device A CUDA device object to associate with the tensor.
#'
#' @return A tensor object.
#' @export
as_tensor.matrix <- function(array, device) {
  # If the user has the ignore_implicit_transpose option
  if (!identical(getOption("lamp.ignore_implicit_transpose"), TRUE)) {
    rlang::warn("`as_tensor` does an implicit transpose of the matrix. It is recommended to use `new_tensor`")
  }
  res <- as_tensor_matrix(array, device)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Convert an Array to a Tensor
#'
#' Converts an n-dimensional array to a tensor.
#'
#' @param array An array to be converted to a tensor.
#' @param device A CUDA device object to associate with the tensor.
#'
#' @return A tensor object.
#' @export
as_tensor.array <- function(array, device) {
  dimensions <- length(dim(array))
  res <- switch(dimensions,
    as_tensor_numeric(array, device),
    as_tensor.matrix(array, device),
    as_tensor_matrix_3d(array, device)
  )
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Convert a Numeric Vector to a Tensor
#'
#' Converts a numeric vector to a tensor.
#'
#' @param array A numeric vector to be converted to a tensor.
#' @param device A CUDA device object to associate with the tensor.
#'
#' @return A tensor object.
#' @export
as_tensor.numeric <- function(array, device) {
  res <- as_tensor_numeric(array, device)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Print a Tensor Object
#'
#' Prints a human-readable representation of a tensor object.
#'
#' @param tensor A tensor object to be printed.
#' @param ... Additional arguments passed to the printing method.
#'
#' @export
print.Tensor <- function(tensor, ...) {
  print_tensor(tensor)
}

#' Add Two Tensors
#'
#' Performs element-wise addition of two tensors.
#'
#' @param lhs The left-hand side tensor.
#' @param rhs The right-hand side tensor.
#'
#' @return A tensor resulting from the addition.
#' @export
`+.Tensor` <- function(lhs, rhs) {
  if (!methods::is(rhs, "Tensor")) rhs <- as_tensor(rhs, get_device(lhs))
  res <- add_tensor(lhs, rhs)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Subtract Two Tensors
#'
#' Performs element-wise subtraction of two tensors.
#'
#' @param lhs The left-hand side tensor.
#' @param rhs The right-hand side tensor.
#'
#' @return A tensor resulting from the subtraction.
#' @export
`-.Tensor` <- function(lhs, rhs) {
  if (!methods::is(rhs, "Tensor")) rhs <- as_tensor(rhs, get_device(lhs))
  res <- sub_tensor(lhs, rhs)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Multiply Two Tensors
#'
#' Performs element-wise multiplication of two tensors.
#'
#' @param lhs The left-hand side tensor.
#' @param rhs The right-hand side tensor.
#'
#' @return A tensor resulting from the multiplication.
#' @export
`*.Tensor` <- function(lhs, rhs) {
  if (!methods::is(rhs, "Tensor")) rhs <- as_tensor(rhs, get_device(lhs))
  res <- mul_tensor(lhs, rhs)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Devide Two Tensors
#'
#' Performs element-wise division of two tensors.
#'
#' @param lhs The left-hand side tensor.
#' @param rhs The right-hand side tensor.
#'
#' @return A tensor resulting from the division
#' @export
`/.Tensor` <- function(lhs, rhs) {
  if (!methods::is(rhs, "Tensor")) rhs <- as_tensor(rhs, get_device(lhs))
  res <- div_tensor(lhs, rhs)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}


#' @export
`%*%.Tensor` <- function(lhs, rhs) {
  if (!methods::is(rhs, "Tensor")) rhs <- as_tensor(rhs, get_device(lhs))
  res <- dot_tensor(lhs, rhs)
  if (methods::is(res, "error")) rlang::abort(res$value)
  return(res)
}

#' Collect Tensor Data
#'
#' Converts the contents of a tensor back into an R array or matrix.
#'
#' @param tensor A tensor object to be collected.
#'
#' @return An R array, matrix, or vector depending on the tensor's dimensions.
#' @export
collect_tensor <- function(tensor) {
  res <- int_collect_tensor(tensor)
  if (methods::is(res, "error")) rlang::abort(res$value)

  dims <- attr(res, "dims")
  n_dims <- length(dims)

  if (n_dims == 0L) n_dims <- 1L # In R there is no 0 dimensions

  attr(res, "dims") <- NULL

  switch(n_dims,
    res, # 1 dimensions
    matrix(res, nrow = dims[1], ncol = dims[2], byrow = TRUE), # 2 dimensions
    array(res, dim = dims) # 3 dimensions
  )
}
