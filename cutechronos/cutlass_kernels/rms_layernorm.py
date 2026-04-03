"""CUTLASS/CuTeDSL RMSNorm kernel for Ampere-class inference workloads.

Derived from NVIDIA's CuTeDSL RMSNorm example and specialized for the
single-CTA reduction path used on SM80/SM86.
"""

from __future__ import annotations

import operator
from functools import lru_cache

import cuda.bindings.driver as cuda
import torch
import torch.nn as nn

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Boolean, Float32, Int32
from cutlass.cute.runtime import make_ptr

_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

_COMPILE_CACHE: dict[tuple[int, type[cutlass.Numeric], int], object] = {}


@lru_cache(maxsize=16)
def get_sm_version(device: int | str | torch.device | None = None) -> int:
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


@cute.jit
def _predicate_k(txcx: cute.Tensor, limit: int) -> cute.Tensor:
    txpx = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(txcx, mode=[0, 1]), cute.size(txcx, mode=[1]), cute.size(txcx, mode=[2])),
            stride=(cute.size(txcx, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(txpx.shape[0]):
        for rest_k in cutlass.range_constexpr(txpx.shape[2]):
            txpx[rest_v, 0, rest_k] = cute.elem_less(txcx[(0, rest_v), 0, rest_k][1], limit)
    return txpx


@cute.jit
def _block_reduce(
    val: Float32,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, operator.add)


@cute.jit
def _row_reduce(
    x: cute.TensorSSA,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    local_val = x.reduce(cute.ReductionOp.ADD, init_val=init_val, reduction_profile=0)
    warp_width = min(threads_per_row, 32)
    warp_val = cute.arch.warp_reduction(local_val, operator.add, threads_in_group=warp_width)
    warps_per_row = max(threads_per_row // 32, 1)
    if cutlass.const_expr(warps_per_row > 1):
        return _block_reduce(warp_val, reduction_buffer, init_val)
    return warp_val


class RMSNormConfig:
    COPY_BITS = 128

    def __init__(self, dtype: type[cutlass.Numeric], hidden_size: int):
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.vec_size = self.COPY_BITS // dtype.width
        self.threads_per_row = self._compute_threads_per_row(hidden_size)
        self.num_threads = self._compute_num_threads(hidden_size)
        self.num_vec_blocks = max(
            1,
            (hidden_size // self.vec_size + self.threads_per_row - 1) // self.threads_per_row,
        )
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

    @staticmethod
    def _compute_threads_per_row(hidden_size: int) -> int:
        if hidden_size <= 64:
            return 8
        if hidden_size <= 128:
            return 16
        if hidden_size <= 3072:
            return 32
        if hidden_size <= 6144:
            return 64
        if hidden_size <= 16384:
            return 128
        return 256

    @staticmethod
    def _compute_num_threads(hidden_size: int) -> int:
        return 128 if hidden_size <= 16384 else 256

    @staticmethod
    def _make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ) -> tuple:
        shape = ((threads_per_row, rows_per_block), (vec_size, num_vec_blocks))
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def smem_size_in_bytes(self) -> int:
        tile_bytes = self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
        reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        return tile_bytes + reduction_bytes


class _RMSNormKernel:
    def __init__(self, dtype: type[cutlass.Numeric], hidden_size: int):
        self.cfg = RMSNormConfig(dtype, hidden_size)

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        rows: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        mx = cute.make_tensor(x_ptr, cute.make_layout((rows, cfg.hidden_size), stride=(cfg.hidden_size, 1)))
        mw = cute.make_tensor(w_ptr, cute.make_layout((cfg.hidden_size,), stride=(1,)))
        mo = cute.make_tensor(o_ptr, cute.make_layout((rows, cfg.hidden_size), stride=(cfg.hidden_size, 1)))

        self.kernel(mx, mw, mo, eps).launch(
            grid=[cute.ceil_div(rows, cfg.rows_per_block), 1, 1],
            block=[cfg.num_threads, 1, 1],
            smem=cfg.smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mx: cute.Tensor,
        mw: cute.Tensor,
        mo: cute.Tensor,
        eps: Float32,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tv_shape, tv_stride = RMSNormConfig._make_tv_layout(
            cfg.threads_per_row,
            cfg.rows_per_block,
            cfg.vec_size,
            cfg.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (cfg.rows_per_block, cfg.cols_per_tile)

        smem = utils.SmemAllocator()
        sx = smem.allocate_tensor(
            mx.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((cfg.rows_per_block, cfg.warps_per_row)),
            byte_alignment=4,
        )

        idx = cute.make_identity_tensor(mx.shape)
        gx = cute.local_tile(mx, tiler_mn, (bidx, 0))
        go = cute.local_tile(mo, tiler_mn, (bidx, 0))
        cx = cute.local_tile(idx, tiler_mn, (bidx, 0))

        weight_layout = cute.prepend(mw.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        gw = cute.local_tile(cute.make_tensor(mw.iterator, weight_layout), tiler_mn, (0, 0))

        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mx.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )
        copy_atom_load_w = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mx.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mo.element_type,
            num_bits_per_copy=RMSNormConfig.COPY_BITS,
        )

        tiled_copy_load = cute.make_tiled_copy(copy_atom_load_async, tv_layout, tiler_mn)
        tiled_copy_w = cute.make_tiled_copy(copy_atom_load_w, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_x = tiled_copy_load.get_slice(tidx)
        thr_copy_w = tiled_copy_w.get_slice(tidx)
        thr_copy_o = tiled_copy_store.get_slice(tidx)

        txgx = thr_copy_x.partition_S(gx)
        txsx = thr_copy_x.partition_D(sx)
        txgo = thr_copy_o.partition_D(go)
        txcx = thr_copy_x.partition_S(cx)

        txrx = cute.make_fragment_like(txgx)
        txro = cute.make_fragment_like(txgo)
        twgw = thr_copy_w.partition_S(gw)
        twrw = cute.make_fragment_like(twgw)
        txrw = thr_copy_x.retile(twrw)

        txpx = _predicate_k(txcx, limit=cfg.hidden_size)
        row_coord = txcx[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < mx.shape[0]

        if row_in_bounds:
            cute.copy(copy_atom_load_async, txgx, txsx, pred=txpx)

        cute.arch.cp_async_commit_group()
        twpw = _predicate_k(thr_copy_w.partition_S(cx), limit=cfg.hidden_size)
        cute.copy(copy_atom_load_w, twgw, twrw, pred=twpw)
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(txsx, txrx)
        x = txrx.load().to(Float32)
        sum_sq = _row_reduce(x * x, cfg.threads_per_row, reduction_buffer, Float32(0.0))
        rstd = cute.math.rsqrt(sum_sq / cfg.hidden_size + eps, fastmath=True)

        cute.arch.barrier()

        cute.autovec_copy(txsx, txrx)
        x = txrx.load().to(Float32)
        w = txrw.load().to(Float32)
        y = x * rstd
        y = y * w
        txro.store(y.to(cfg.dtype))

        if row_in_bounds:
            cute.copy(copy_atom_store, txro, txgo, pred=txpx)


def _compiled_kernel(dtype: type[cutlass.Numeric], hidden_size: int, stream: cuda.CUstream):
    device_key = get_sm_version(torch.cuda.current_device())
    key = (device_key, dtype, hidden_size)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        kernel = _RMSNormKernel(dtype, hidden_size)
        compiled = cute.compile(
            kernel,
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
            Int32(1),
            Float32(1e-6),
            stream,
        )
        _COMPILE_CACHE[key] = compiled
    return compiled


def _rms_layernorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("CUTLASS RMSNorm requires CUDA tensors")
    if x.dtype != weight.dtype:
        raise ValueError(f"x and weight must share dtype, got {x.dtype} and {weight.dtype}")
    if x.dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise ValueError(f"Unsupported dtype for CUTLASS RMSNorm: {x.dtype}")
    if get_sm_version(x.device) < 80:
        raise ValueError("CUTLASS RMSNorm requires SM80 or newer")

    hidden_size = x.shape[-1]
    x_2d = x.reshape(-1, hidden_size).contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x_2d)

    cutlass_dtype = _TORCH_TO_CUTLASS_DTYPE[x.dtype]
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _compiled_kernel(cutlass_dtype, hidden_size, stream)
    compiled(
        make_ptr(cutlass_dtype, x_2d.data_ptr()),
        make_ptr(cutlass_dtype, weight.data_ptr()),
        make_ptr(cutlass_dtype, out.data_ptr()),
        Int32(x_2d.shape[0]),
        Float32(eps),
        stream,
    )
    return out.reshape(x.shape)


class _CutlassRMSNormAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return _rms_layernorm_forward(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        x_fp32 = x.float()
        rrms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
        normed = x_fp32 * rrms
        grad_weight = (grad_output.float() * normed).reshape(-1, x.shape[-1]).sum(0)
        d_normed = grad_output.float() * weight.float()
        grad_x = rrms * (d_normed - normed * (normed * d_normed).mean(-1, keepdim=True))
        return grad_x.to(x.dtype), grad_weight.to(weight.dtype), None


def rms_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _CutlassRMSNormAutograd.apply(x, weight, eps)


class CutlassRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)
