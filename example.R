library(lamp)

gpu <- new_cuda(0)

# Create a 3D array in R and copy it to the GPU
array_3d <- array(runif(27), dim = c(3, 3, 3)) |>
  as_tensor(gpu)

# You can run as many operations over tensors as you
# want before collecting the tensor back into R

# Multiply the 3d array by itself
multiplication <- array_3d * array_3d

# Multiply by a scalar value
multiplication <- multiplication * 1.5

dot_product <- multiplication %*% array_3d

# If we are not going to do any additional operations
# we can copy the result from the GPU to out R process
result <- collect_tensor(dot_product)

# NOTE: Prints will be displayed differently as R
# stores matrices "col major" instead of "row major"
print(result)
