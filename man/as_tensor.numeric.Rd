% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/s3.R
\name{as_tensor.numeric}
\alias{as_tensor.numeric}
\title{Convert a Numeric Vector to a Tensor}
\usage{
\method{as_tensor}{numeric}(array, device)
}
\arguments{
\item{array}{A numeric vector to be converted to a tensor.}

\item{device}{A CUDA device object to associate with the tensor.}
}
\value{
A tensor object.
}
\description{
Converts a numeric vector to a tensor.
}
