use candle_core::DType;
use extendr_api::{matrix::RMatrix, prelude::*};
use safetensors::tensor::View;

fn candle_error_to_r(error: candle_core::Error) -> extendr_api::Error {
    error.to_string().into()
}

struct Tensor(candle_core::Tensor);

#[extendr]
impl Tensor {}

struct Device(candle_core::Device);

#[extendr]
impl Device {}

#[extendr]
fn get_device(tensor: &Tensor) -> Device {
    Device(tensor.0.device().clone())
}

#[extendr]
fn int_new_cuda(ordinal: i32) -> Result<Device> {
    candle_core::Device::new_cuda(ordinal as usize)
        .map_err(candle_error_to_r)
        .map(Device)
}

#[extendr]
fn as_tensor_numeric(array: &[f64], device: &Device) -> Result<Tensor> {
    candle_core::Tensor::from_slice(array, array.len(), &device.0)
        .map_err(candle_error_to_r)
        .map(Tensor)
}

#[extendr]
fn as_tensor_matrix(matrix: RMatrix<f64>, device: &Device) -> Result<Tensor> {
    let dims = matrix.dim();
    candle_core::Tensor::from_slice(matrix.data(), (dims[1], dims[0]), &device.0)
        .map_err(candle_error_to_r)?
        .transpose(0, 1)
        .map_err(candle_error_to_r)
        .map(Tensor)
}

#[extendr]
fn as_tensor_matrix_3d(matrix: RArray<f64, [usize; 3]>, device: &Device) -> Result<Tensor> {
    let dims = matrix.dim();
    candle_core::Tensor::from_slice(matrix.data(), (dims[0], dims[1], dims[2]), &device.0)
        .map_err(candle_error_to_r)
        .map(Tensor)
}

#[extendr]
fn add_tensor(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.0.dims() == rhs.0.dims() {
        lhs.0.add(&rhs.0).map(Tensor).map_err(candle_error_to_r)
    } else {
        lhs.0
            .broadcast_add(&rhs.0)
            .map(Tensor)
            .map_err(candle_error_to_r)
    }
}

#[extendr]
fn sub_tensor(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.0.dims() == rhs.0.dims() {
        lhs.0.sub(&rhs.0).map(Tensor).map_err(candle_error_to_r)
    } else {
        lhs.0
            .broadcast_sub(&rhs.0)
            .map(Tensor)
            .map_err(candle_error_to_r)
    }
}

#[extendr]
fn mul_tensor(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.0.dims() == rhs.0.dims() {
        lhs.0.mul(&rhs.0).map(Tensor).map_err(candle_error_to_r)
    } else {
        lhs.0
            .broadcast_mul(&rhs.0)
            .map(Tensor)
            .map_err(candle_error_to_r)
    }
}

#[extendr]
fn div_tensor(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.0.dims() == rhs.0.dims() {
        lhs.0.div(&rhs.0).map(Tensor).map_err(candle_error_to_r)
    } else {
        lhs.0
            .broadcast_div(&rhs.0)
            .map(Tensor)
            .map_err(candle_error_to_r)
    }
}

#[extendr]
fn dot_tensor(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.0.dims() == rhs.0.dims() {
        lhs.0.matmul(&rhs.0).map(Tensor).map_err(candle_error_to_r)
    } else {
        lhs.0
            .broadcast_matmul(&rhs.0)
            .map(Tensor)
            .map_err(candle_error_to_r)
    }
}

#[extendr]
fn int_new_tensor(array: &[f64], dims: &[i32], device: &Device) -> Result<Tensor> {
    let dims = dims.iter().map(|&d| d as usize).collect::<Vec<_>>();
    candle_core::Tensor::from_slice(array, dims, &device.0)
        .map_err(candle_error_to_r)
        .map(Tensor)
}

#[extendr]
fn print_tensor(tensor: &Tensor) {
    println!("{}", tensor.0);
}

#[extendr]
fn int_collect_tensor(tensor: &Tensor) -> Result<Robj> {
    let dims = tensor
        .0
        .dims()
        .iter()
        .map(|&d| Rint::new(d as i32))
        .collect::<Integers>();
    let dtype = tensor.0.dtype();
    let data = tensor.0.data();
    let mut obj = match dtype {
        DType::F64 => data
            .chunks_exact(8)
            .map(|bytes| unsafe {
                let bytes_ptr = bytes.as_ptr() as *const [u8; 8];
                let bytes: [u8; 8] = *bytes_ptr;
                f64::from_le_bytes(bytes)
            })
            .collect::<Vec<_>>()
            .into_robj(),
        _ => todo!(),
    };
    obj.set_attrib("dims", dims)?;
    Ok(obj)
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod lamp;
    fn as_tensor_numeric;
    fn int_new_cuda;
    fn as_tensor_matrix;
    fn as_tensor_matrix_3d;
    fn print_tensor;
    fn add_tensor;
    fn mul_tensor;
    fn div_tensor;
    fn sub_tensor;
    fn int_new_tensor;
    fn int_collect_tensor;
    fn dot_tensor;
    fn get_device;
}
