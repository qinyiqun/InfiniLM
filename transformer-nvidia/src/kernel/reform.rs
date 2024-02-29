﻿use crate::storage::DevMem;
use cuda::{bindings::CUdeviceptr, AsRaw, ContextGuard, KernelFn, Stream};
use std::ffi::c_void;
use tensor::{udim, Tensor};

pub struct Reform {
    f: KernelFn,
    block_size: udim,
    warp_size: udim,
}

impl Reform {
    pub fn new(block_size: usize, warp_size: usize, ctx: &ContextGuard) -> Self {
        assert_eq!(
            block_size % warp_size,
            0,
            "block_size must be a multiple of warp_size"
        );

        let name = "reform";

        const REFORM: &str = include_str!("reform.cuh");
        let code = format!(
            r#"{REFORM}

extern "C" __global__ void {name}(
    void       *__restrict__ dst,
    unsigned int const rsa,
    unsigned int const csa,
    void const *__restrict__ src,
    unsigned int const rsb,
    unsigned int const csb,
    unsigned int const ncols,
    unsigned int const bytes_per_thread
){{
    switch (bytes_per_thread) {{
        case  1: reform<uchar1 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  2: reform<uchar2 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  4: reform<float1 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case  8: reform<float2 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case 16: reform<float4 >(dst, rsa, csa, src, rsb, csb, ncols); break;
        case 32: reform<double4>(dst, rsa, csa, src, rsb, csb, ncols); break;
    }}
}}
"#
        );

        ctx.compile(code);
        Self {
            f: KernelFn::get(name).unwrap(),
            block_size: block_size as _,
            warp_size: warp_size as _,
        }
    }

    pub fn launch(&self, dst: &Tensor<DevMem>, src: &Tensor<DevMem>, stream: &Stream) {
        assert_eq!(dst.data_type(), src.data_type());
        assert_eq!(dst.shape(), src.shape());

        let &[r, c, b] = dst.shape() else {
            unreachable!()
        };
        let &[rsa, csa, 1] = dst.strides() else {
            unreachable!()
        };
        let &[rsb, csb, 1] = src.strides() else {
            unreachable!()
        };

        let contiguous_bytes = b * dst.data_type().size() as udim;
        assert_eq!(contiguous_bytes % self.warp_size, 0);
        let bytes_per_thread = contiguous_bytes / self.warp_size;
        assert!(bytes_per_thread <= 32 && bytes_per_thread.is_power_of_two());

        let dst_ptr =
            (unsafe { dst.physical().as_raw() } as isize + dst.bytes_offset()) as CUdeviceptr;
        let rsa = rsa as udim / b;
        let csa = csa as udim / b;
        let src_ptr =
            (unsafe { src.physical().as_raw() } as isize + src.bytes_offset()) as CUdeviceptr;
        let rsb = rsb as udim / b;
        let csb = csb as udim / b;
        let params: [*const c_void; 8] = [
            (&dst_ptr) as *const _ as _,
            (&rsa) as *const _ as _,
            (&csa) as *const _ as _,
            (&src_ptr) as *const _ as _,
            (&rsb) as *const _ as _,
            (&csb) as *const _ as _,
            (&c) as *const _ as _,
            (&bytes_per_thread) as *const _ as _,
        ];

        let max_warp_per_block = self.block_size / self.warp_size;
        let grid_dims = ((c + max_warp_per_block - 1) / max_warp_per_block, r);
        let block_dims = (self.warp_size, (c + grid_dims.0 - 1) / grid_dims.0);
        self.f
            .launch(grid_dims, block_dims, params.as_ptr(), 0, Some(stream));
    }
}