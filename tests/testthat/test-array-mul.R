options("lamp.ignore_implicit_transpose" = TRUE)

test_that("array multiplication (50 items)", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    array_1 = quickcheck::double_(len = 50),
    array_2 = quickcheck::double_(len = 50),
    property = function(array_1, array_2) {
      array_1_lamp <- as_tensor(array_1, gpu)
      array_2_lamp <- as_tensor(array_2, gpu)

      expected <- array_1 * array_2
      result <- collect_tensor(array_1_lamp * array_2_lamp)

      testthat::expect_equal(result, expected)
    }
  )
})

test_that("array multiplication with scalar", {
  gpu <- new_cuda(0)

  quickcheck::for_all(
    array_1 = quickcheck::double_(),
    scalar = quickcheck::double_(len = 1),
    property = function(array_1, scalar) {
      array_1_lamp <- as_tensor(array_1, gpu)
      scalar_lamp <- as_tensor(scalar, gpu)

      expected <- array_1 * scalar
      result <- collect_tensor(array_1_lamp * scalar_lamp)

      testthat::expect_equal(result, expected)
    }
  )
})
