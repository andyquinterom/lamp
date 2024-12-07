library(lamp)
library(microbenchmark)
library(assertthat)
library(ggplot2)
library(dplyr)

# Function for element-wise multiplication
matrix_mul <- function(array_2d) {
  array_2d * array_2d
}

# Function for element-wise addition
matrix_add <- function(array_2d) {
  array_2d + array_2d
}

# Function for element-wise division
matrix_div <- function(array_2d) {
  array_2d / (array_2d + 1e-5)  # Avoid division by zero
}

# Function for matrix dot product
matrix_dot <- function(array_2d) {
  array_2d %*% array_2d
}

# Fixed matrix size
SIZE <- 1000

# Create the matrix
array_2d <- array(runif(SIZE ^ 2), dim = c(SIZE, SIZE))

# GPU setup
gpu <- new_cuda(0)
array_2d_lamp <- as_tensor(array_2d, gpu)

# Assertions for correctness
assertthat::are_equal(
  collect_tensor(matrix_mul(array_2d_lamp)),
  matrix_mul(array_2d)
)
assertthat::are_equal(
  collect_tensor(matrix_add(array_2d_lamp)),
  matrix_add(array_2d)
)
assertthat::are_equal(
  collect_tensor(matrix_div(array_2d_lamp)),
  matrix_div(array_2d)
)
assertthat::are_equal(
  collect_tensor(matrix_dot(array_2d_lamp)),
  matrix_dot(array_2d)
)

# Benchmark for multiple operations
benchmark_results <- microbenchmark(
  times = 100L,
  base_mul = matrix_mul(array_2d),
  lamp_mul = matrix_mul(array_2d_lamp),
  base_add = matrix_add(array_2d),
  lamp_add = matrix_add(array_2d_lamp),
  base_div = matrix_div(array_2d),
  lamp_div = matrix_div(array_2d_lamp),
  base_dot = matrix_dot(array_2d),
  lamp_dot = matrix_dot(array_2d_lamp)
)

# Convert results to a data frame and calculate mean times
benchmark_df <- as.data.frame(benchmark_results) |>
  group_by(expr) |>
  summarize(mean_time = median(time) / 1e6)  # Convert to milliseconds

# Create a bar plot
ggplot(benchmark_df, aes(x = reorder(expr, mean_time), y = mean_time, fill = expr)) +
  geom_col() +
  labs(
    title = "Benchmark Results: Base vs Lamp",
    x = "Operation",
    y = "Median Execution Time (ms)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )
