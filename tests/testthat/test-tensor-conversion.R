options("lamp.ignore_implicit_transpose" = TRUE)

test_that("1 dimensional tensors", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    dims = quickcheck::integer_bounded(1, 1000, len = 1),
    property = function(dims) {
      DIM1 <- dims[1]

      values <- runif(DIM1)

      lamp_vector <- collect_tensor(as_tensor(values, gpu))

      testthat::expect_equal(values, lamp_vector)
    }
  )
})

test_that("2 dimensional tensors", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    dims = quickcheck::integer_bounded(1, 25, len = 2),
    property = function(dims) {
      DIM1 <- dims[1]
      DIM2 <- dims[2]

      values <- runif(DIM1 * DIM2)

      r_matrix <- matrix(values, nrow = DIM1, ncol = DIM2)

      lamp_matrix <- collect_tensor(as_tensor(r_matrix, gpu))

      testthat::expect_equal(r_matrix, lamp_matrix)
    }
  )
})

test_that("3 dimensional tensors", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    dims = quickcheck::integer_bounded(1, 25, len = 3),
    property = function(dims) {
      DIM1 <- dims[1]
      DIM2 <- dims[2]
      DIM3 <- dims[3]

      values <- runif(DIM1 * DIM2 * DIM3)

      r_matrix <- array(values, dim = c(DIM1, DIM2, DIM3))

      lamp_matrix <- collect_tensor(as_tensor(r_matrix, gpu))

      testthat::expect_equal(r_matrix, lamp_matrix)
    }
  )
})
