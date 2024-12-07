options("lamp.ignore_implicit_transpose" = TRUE)

test_that("mul 2 dimensional tensors with scalar", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    dims = quickcheck::integer_bounded(1, 25, len = 2),
    scalar = quickcheck::double_(len = 1),
    property = function(dims, scalar) {
      DIM1 <- dims[1]
      DIM2 <- dims[2]

      values <- runif(DIM1 * DIM2)

      r_matrix <- matrix(values, nrow = DIM1, ncol = DIM2)

      lamp_matrix <- as_tensor(r_matrix, gpu)
      lamp_scalar <- as_tensor(scalar, gpu)


      testthat::expect_equal(
        r_matrix * scalar,
        collect_tensor(lamp_matrix * lamp_scalar)
      )
    }
  )
})

test_that("mul 2 dimensional tensors", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    dims = quickcheck::integer_bounded(1, 25, len = 2),
    property = function(dims) {
      DIM1 <- dims[1]
      DIM2 <- dims[2]

      values <- runif(DIM1 * DIM2)

      r_matrix <- matrix(values, nrow = DIM1, ncol = DIM2)

      lamp_matrix <- as_tensor(r_matrix, gpu)


      testthat::expect_equal(
        r_matrix * r_matrix,
        collect_tensor(lamp_matrix * lamp_matrix)
      )
    }
  )
})

test_that("mul 3 dimensional tensors", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    dims = quickcheck::integer_bounded(1, 25, len = 3),
    property = function(dims) {
      DIM1 <- dims[1]
      DIM2 <- dims[2]
      DIM3 <- dims[3]

      values <- runif(DIM1 * DIM2 * DIM3)

      r_matrix <- array(values, dim = c(DIM1, DIM2, DIM3))

      lamp_matrix <- as_tensor(r_matrix, gpu)

      testthat::expect_equal(
        r_matrix * r_matrix,
        collect_tensor(lamp_matrix * lamp_matrix)
      )
    }
  )
})

test_that("dot 2 dimensional tensors", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    size = quickcheck::integer_bounded(1, 25, len = 1),
    property = function(size) {
      values <- runif(size * size)

      r_matrix <- matrix(values, nrow = size, ncol = size)

      lamp_matrix <- as_tensor(r_matrix, gpu)


      testthat::expect_equal(
        r_matrix %*% r_matrix,
        collect_tensor(lamp_matrix %*% lamp_matrix)
      )
    }
  )
})


test_that("dot product works with matrices of different sizes", {
  gpu <- new_cuda(0)

  A <- matrix(c(1, 2, 3, 4, 5, 6), nrow = 2) # 2x3 matrix
  B <- matrix(c(7, 8, 9, 10, 11, 12), nrow = 3) # 3x2 matrix

  A_lamp <- as_tensor(A, gpu)
  B_lamp <- as_tensor(B, gpu)

  expected <- A %*% B
  result <- A_lamp %*% B_lamp

  expect_equal(collect_tensor(result), expected)
})

# Test: Dot product with square matrices
test_that("dot product works with square matrices", {
  gpu <- new_cuda(0)

  A <- matrix(c(1, 2, 3, 4), nrow = 2) # 2x2 matrix
  B <- matrix(c(5, 6, 7, 8), nrow = 2) # 2x2 matrix

  A_lamp <- as_tensor(A, gpu)
  B_lamp <- as_tensor(B, gpu)

  expected <- A %*% B
  result <- A_lamp %*% B_lamp

  expect_equal(collect_tensor(result), expected)
})

# Test: Dot product with larger matrices
test_that("dot product works with larger matrices", {
  gpu <- new_cuda(0)

  A <- matrix(runif(100), nrow = 10) # 10x10 matrix
  B <- matrix(runif(100), nrow = 10) # 10x10 matrix

  A_lamp <- as_tensor(A, gpu)
  B_lamp <- as_tensor(B, gpu)

  expected <- A %*% B
  result <- A_lamp %*% B_lamp

  expect_equal(collect_tensor(result), expected)
})

# Test: Dot product with single-row matrix (vector)
test_that("dot product works with single-row matrix", {
  gpu <- new_cuda(0)

  A <- matrix(c(1, 2, 3), nrow = 1) # 1x3 matrix (row vector)
  B <- matrix(c(4, 5, 6), nrow = 3) # 3x1 matrix (column vector)

  A_lamp <- as_tensor(A, gpu)
  B_lamp <- as_tensor(B, gpu)

  expected <- A %*% B
  result <- A_lamp %*% B_lamp

  expect_equal(collect_tensor(result), expected)
})

# Test: Dot product with mismatched dimensions
test_that("dot product throws error with mismatched dimensions", {
  gpu <- new_cuda(0)

  A <- matrix(c(1, 2, 3, 4), nrow = 2) # 2x2 matrix
  B <- matrix(c(5, 6, 7, 8, 9, 10), nrow = 3) # 3x2 matrix

  A_lamp <- as_tensor(A, gpu)
  B_lamp <- as_tensor(B, gpu)

  expect_error(A_lamp %*% B_lamp)
})

# Test: Dot product with scalars
test_that("dot product works with scalars", {
  gpu <- new_cuda(0)

  A <- matrix(2, nrow = 1) # 1x1 matrix (scalar)
  B <- matrix(3, nrow = 1) # 1x1 matrix (scalar)

  A_lamp <- as_tensor(A, gpu)
  B_lamp <- as_tensor(B, gpu)

  expected <- A %*% B
  result <- A_lamp %*% B_lamp

  expect_equal(collect_tensor(result), expected)
})

# Test: Dot product with identity matrix
test_that("dot product works with identity matrix", {
  gpu <- new_cuda(0)

  A <- diag(3) # 3x3 identity matrix
  B <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow = 3) # 3x3 matrix

  A_lamp <- as_tensor(A, gpu)
  B_lamp <- as_tensor(B, gpu)

  expected <- A %*% B
  result <- A_lamp %*% B_lamp

  expect_equal(collect_tensor(result), expected)
})
