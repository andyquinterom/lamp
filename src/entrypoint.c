// We need to forward routine registration from C to Rust
// to avoid the linker removing the static library.

void R_init_lamp_extendr(void *dll);

void R_init_lamp(void *dll) {
    R_init_lamp_extendr(dll);
}
