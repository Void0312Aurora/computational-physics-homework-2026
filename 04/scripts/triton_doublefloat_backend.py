from __future__ import annotations

import math

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


SQRT3_2 = math.sqrt(3.0) / 2.0


def _root_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [1.0 + 0.0j, complex(-0.5, SQRT3_2), complex(-0.5, -SQRT3_2)],
        dtype=torch.complex128,
        device=device,
    )


def downsample_tile(root_map: torch.Tensor, iters: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_h = root_map.shape[0] // factor
    out_w = root_map.shape[1] // factor

    reshaped_root = root_map.view(out_h, factor, out_w, factor)
    counts = torch.stack([(reshaped_root == idx).sum(dim=(1, 3)) for idx in range(3)], dim=0)
    converged_pixels = counts.sum(dim=0)
    dominant = counts.argmax(dim=0).to(torch.int8)
    small_root = torch.full((out_h, out_w), -1, dtype=torch.int8, device=root_map.device)
    small_root = torch.where(converged_pixels > 0, dominant, small_root)

    converged_mask = (root_map >= 0).view(out_h, factor, out_w, factor)
    iter_sum = (iters.float() * (root_map >= 0).float()).view(out_h, factor, out_w, factor).sum(dim=(1, 3))
    count = converged_mask.sum(dim=(1, 3))
    small_iters = torch.zeros((out_h, out_w), dtype=torch.float32, device=root_map.device)
    small_iters = torch.where(count > 0, iter_sum / count.clamp_min(1), small_iters)
    return small_root, small_iters


if TRITON_AVAILABLE:

    @triton.jit
    def ds_renorm(h, l):
        s = h + l
        e = l - (s - h)
        return s, e


    @triton.jit
    def ds_add_basic(ah, al, bh, bl):
        s = ah + bh
        v = s - ah
        t = ((bh - v) + (ah - (s - v))) + al + bl
        return ds_renorm(s, t)


    @triton.jit
    def ds_sub_basic(ah, al, bh, bl):
        return ds_add_basic(ah, al, -bh, -bl)


    @triton.jit
    def split_f32(a):
        c = 4097.0 * a
        ahi = c - (c - a)
        alo = a - ahi
        return ahi, alo


    @triton.jit
    def two_prod(a, b):
        p = a * b
        ahi, alo = split_f32(a)
        bhi, blo = split_f32(b)
        err = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
        return p, err


    @triton.jit
    def ds_mul_basic(ah, al, bh, bl):
        p, e = two_prod(ah, bh)
        e = e + ah * bl + al * bh
        return ds_renorm(p, e)


    @triton.jit
    def ds_square_basic(ah, al):
        return ds_mul_basic(ah, al, ah, al)


    @triton.jit
    def ds_mul_scalar_basic(ah, al, scalar):
        zero = tl.zeros_like(ah)
        scalar_vec = zero + scalar
        return ds_mul_basic(ah, al, scalar_vec, zero)


    @triton.jit
    def ds_reciprocal_real_basic(dh, dl):
        qh = 1.0 / dh
        ql = tl.zeros_like(qh)
        oneh = tl.zeros_like(qh) + 1.0
        onel = tl.zeros_like(qh)
        for _ in tl.static_range(0, 2):
            ph, pl = ds_mul_basic(dh, dl, qh, ql)
            eh, el = ds_sub_basic(oneh, onel, ph, pl)
            ch, cl = ds_mul_basic(qh, ql, eh, el)
            qh, ql = ds_add_basic(qh, ql, ch, cl)
        return qh, ql


    @triton.jit
    def newton_doublefloat_basic_kernel(
        zr_hi0_ptr,
        zr_lo0_ptr,
        zi_hi0_ptr,
        zi_lo0_ptr,
        zr_hi_out_ptr,
        zr_lo_out_ptr,
        zi_hi_out_ptr,
        zi_lo_out_ptr,
        root_ptr,
        iter_ptr,
        n_points,
        tol2,
        late_tol2,
        BLOCK_SIZE: tl.constexpr,
        MAX_ITER: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_points

        zr_hi = tl.load(zr_hi0_ptr + offs, mask=mask, other=0.0)
        zr_lo = tl.load(zr_lo0_ptr + offs, mask=mask, other=0.0)
        zi_hi = tl.load(zi_hi0_ptr + offs, mask=mask, other=0.0)
        zi_lo = tl.load(zi_lo0_ptr + offs, mask=mask, other=0.0)

        active = mask
        root = tl.full([BLOCK_SIZE], -1, tl.int32)
        iters = tl.zeros([BLOCK_SIZE], tl.int32)

        for step in tl.range(0, MAX_ITER):
            zr2_hi, zr2_lo = ds_square_basic(zr_hi, zr_lo)
            zi2_hi, zi2_lo = ds_square_basic(zi_hi, zi_lo)
            z2r_hi, z2r_lo = ds_sub_basic(zr2_hi, zr2_lo, zi2_hi, zi2_lo)

            zrzi_hi, zrzi_lo = ds_mul_basic(zr_hi, zr_lo, zi_hi, zi_lo)
            z2i_hi, z2i_lo = ds_mul_scalar_basic(zrzi_hi, zrzi_lo, 2.0)

            a2_hi, a2_lo = ds_square_basic(z2r_hi, z2r_lo)
            b2_hi, b2_lo = ds_square_basic(z2i_hi, z2i_lo)
            den_hi, den_lo = ds_add_basic(a2_hi, a2_lo, b2_hi, b2_lo)
            den_approx = den_hi + den_lo
            safe = active & (den_approx > 1.0e-14)

            safe_den_hi = tl.where(safe, den_hi, 1.0)
            safe_den_lo = tl.where(safe, den_lo, 0.0)
            recip_hi, recip_lo = ds_reciprocal_real_basic(safe_den_hi, safe_den_lo)

            invr_hi, invr_lo = ds_mul_basic(z2r_hi, z2r_lo, recip_hi, recip_lo)
            invr_hi, invr_lo = ds_mul_scalar_basic(invr_hi, invr_lo, 1.0 / 3.0)
            invi_hi, invi_lo = ds_mul_basic(z2i_hi, z2i_lo, recip_hi, recip_lo)
            invi_hi, invi_lo = ds_mul_scalar_basic(invi_hi, invi_lo, -1.0 / 3.0)

            invr_hi = tl.where(safe, invr_hi, 0.0)
            invr_lo = tl.where(safe, invr_lo, 0.0)
            invi_hi = tl.where(safe, invi_hi, 0.0)
            invi_lo = tl.where(safe, invi_lo, 0.0)

            lhs_r_hi, lhs_r_lo = ds_mul_scalar_basic(zr_hi, zr_lo, 2.0 / 3.0)
            lhs_i_hi, lhs_i_lo = ds_mul_scalar_basic(zi_hi, zi_lo, 2.0 / 3.0)

            next_r_hi, next_r_lo = ds_add_basic(lhs_r_hi, lhs_r_lo, invr_hi, invr_lo)
            next_i_hi, next_i_lo = ds_add_basic(lhs_i_hi, lhs_i_lo, invi_hi, invi_lo)

            zr_hi = tl.where(active, next_r_hi, zr_hi)
            zr_lo = tl.where(active, next_r_lo, zr_lo)
            zi_hi = tl.where(active, next_i_hi, zi_hi)
            zi_lo = tl.where(active, next_i_lo, zi_lo)

            zr = zr_hi + zr_lo
            zi = zi_hi + zi_lo
            d0 = (zr - 1.0) * (zr - 1.0) + zi * zi
            tmp = zr + 0.5
            d1i = zi - 0.8660254037844386
            d2i = zi + 0.8660254037844386
            d1 = tmp * tmp + d1i * d1i
            d2 = tmp * tmp + d2i * d2i

            root_idx = tl.zeros([BLOCK_SIZE], tl.int32)
            min_d = d0
            pick1 = d1 < min_d
            min_d = tl.where(pick1, d1, min_d)
            root_idx = tl.where(pick1, 1, root_idx)
            pick2 = d2 < min_d
            min_d = tl.where(pick2, d2, min_d)
            root_idx = tl.where(pick2, 2, root_idx)

            just = active & (min_d < tol2)
            root = tl.where(just, root_idx, root)
            iters = tl.where(just, step + 1, iters)
            active = active & ~just

        zr = zr_hi + zr_lo
        zi = zi_hi + zi_lo
        d0 = (zr - 1.0) * (zr - 1.0) + zi * zi
        tmp = zr + 0.5
        d1i = zi - 0.8660254037844386
        d2i = zi + 0.8660254037844386
        d1 = tmp * tmp + d1i * d1i
        d2 = tmp * tmp + d2i * d2i

        root_idx = tl.zeros([BLOCK_SIZE], tl.int32)
        min_d = d0
        pick1 = d1 < min_d
        min_d = tl.where(pick1, d1, min_d)
        root_idx = tl.where(pick1, 1, root_idx)
        pick2 = d2 < min_d
        min_d = tl.where(pick2, d2, min_d)
        root_idx = tl.where(pick2, 2, root_idx)

        late = active & (min_d < late_tol2)
        root = tl.where(late, root_idx, root)
        iters = tl.where(late, MAX_ITER, iters)

        tl.store(zr_hi_out_ptr + offs, zr_hi, mask=mask)
        tl.store(zr_lo_out_ptr + offs, zr_lo, mask=mask)
        tl.store(zi_hi_out_ptr + offs, zi_hi, mask=mask)
        tl.store(zi_lo_out_ptr + offs, zi_lo, mask=mask)
        tl.store(root_ptr + offs, root, mask=mask)
        tl.store(iter_ptr + offs, iters, mask=mask)


def build_boundary_mask(root_map: torch.Tensor, dilation: int) -> torch.Tensor:
    boundary = torch.zeros_like(root_map, dtype=torch.bool)
    boundary[1:, :] |= root_map[1:, :] != root_map[:-1, :]
    boundary[:-1, :] |= root_map[:-1, :] != root_map[1:, :]
    boundary[:, 1:] |= root_map[:, 1:] != root_map[:, :-1]
    boundary[:, :-1] |= root_map[:, :-1] != root_map[:, 1:]
    boundary |= root_map < 0
    if dilation <= 0:
        return boundary
    dilated = boundary.clone()
    frontier = boundary
    for _ in range(dilation):
        grown = frontier.clone()
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        dilated |= grown
        frontier = grown
    return dilated


@torch.no_grad()
def replay_reference_newton_selected(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    tol: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = mask.device
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        empty_idx = torch.empty((0,), dtype=torch.int64, device=device)
        empty_root = torch.empty((0,), dtype=torch.int16, device=device)
        empty_iter = torch.empty((0,), dtype=torch.int16, device=device)
        return empty_idx, empty_idx, empty_root, empty_iter

    row_idx = coords[:, 0]
    col_idx = coords[:, 1]
    x64 = x.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    z = torch.complex(x64[col_idx], y64[row_idx])
    roots = _root_tensor(device)
    converged = torch.zeros(z.shape, dtype=torch.bool, device=device)
    root_sel = torch.full(z.shape, -1, dtype=torch.int16, device=device)
    iter_sel = torch.zeros(z.shape, dtype=torch.int16, device=device)

    for step in range(1, max_iter + 1):
        z2 = z * z
        safe = torch.abs(z2) > 1.0e-28
        inv = torch.zeros_like(z)
        if bool(torch.any(safe).item()):
            inv[safe] = (1.0 / 3.0) / z2[safe]
        z_next = (2.0 / 3.0) * z + inv
        z = torch.where(converged, z, z_next)

        distances = torch.stack([torch.abs(z - root_val) for root_val in roots], dim=0)
        mins, indices = torch.min(distances, dim=0)
        just = (~converged) & (mins < tol)
        root_sel[just] = indices[just].to(torch.int16)
        iter_sel[just] = step
        converged |= just
        if bool(torch.all(converged).item()):
            break

    distances = torch.stack([torch.abs(z - root_val) for root_val in roots], dim=0)
    mins, indices = torch.min(distances, dim=0)
    late = (~converged) & (mins < 10.0 * tol)
    root_sel[late] = indices[late].to(torch.int16)
    iter_sel[late] = max_iter
    return row_idx, col_idx, root_sel, iter_sel


def split_doublefloat_grid(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x64 = x.to(dtype=torch.float64)
    y64 = y.to(dtype=torch.float64)
    x_hi = x64.to(torch.float32)
    x_lo = (x64 - x_hi.to(torch.float64)).to(torch.float32)
    y_hi = y64.to(torch.float32)
    y_lo = (y64 - y_hi.to(torch.float64)).to(torch.float32)

    zr_hi = x_hi[None, :].expand(y.shape[0], -1).contiguous().reshape(-1)
    zr_lo = x_lo[None, :].expand(y.shape[0], -1).contiguous().reshape(-1)
    zi_hi = y_hi[:, None].expand(-1, x.shape[0]).contiguous().reshape(-1)
    zi_lo = y_lo[:, None].expand(-1, x.shape[0]).contiguous().reshape(-1)
    return zr_hi, zr_lo, zi_hi, zi_lo


@torch.no_grad()
def run_tile_triton_doublefloat_newton(
    x: torch.Tensor,
    y: torch.Tensor,
    tol: float,
    max_iter: int,
    factor: int,
    block_size: int,
    num_warps: int,
    post_polish_steps: int = 2,
    iter_replay_threshold: int = 8,
    iter_replay_dilation: int = 1,
    return_downsample: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int, np.ndarray]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available in the current Python environment.")

    device = x.device
    zr_hi, zr_lo, zi_hi, zi_lo = split_doublefloat_grid(x, y)
    n_points = zr_hi.numel()
    zr_hi_out = torch.empty_like(zr_hi)
    zr_lo_out = torch.empty_like(zr_lo)
    zi_hi_out = torch.empty_like(zi_hi)
    zi_lo_out = torch.empty_like(zi_lo)
    root = torch.empty(n_points, dtype=torch.int32, device=device)
    iters = torch.empty(n_points, dtype=torch.int32, device=device)
    grid = lambda meta: (triton.cdiv(n_points, meta["BLOCK_SIZE"]),)
    newton_doublefloat_basic_kernel[grid](
        zr_hi,
        zr_lo,
        zi_hi,
        zi_lo,
        zr_hi_out,
        zr_lo_out,
        zi_hi_out,
        zi_lo_out,
        root,
        iters,
        n_points,
        tol * tol,
        (10.0 * tol) ** 2,
        BLOCK_SIZE=block_size,
        MAX_ITER=max_iter,
        num_warps=num_warps,
    )
    root_map = root.view(y.shape[0], x.shape[0]).to(torch.int16)
    iter_map = iters.view(y.shape[0], x.shape[0]).to(torch.int16)
    polished_root = root_map

    if post_polish_steps > 0:
        zr64 = zr_hi_out.view(y.shape[0], x.shape[0]).to(torch.float64) + zr_lo_out.view(y.shape[0], x.shape[0]).to(torch.float64)
        zi64 = zi_hi_out.view(y.shape[0], x.shape[0]).to(torch.float64) + zi_lo_out.view(y.shape[0], x.shape[0]).to(torch.float64)
        z = torch.complex(zr64, zi64)
        for _ in range(post_polish_steps):
            z2 = z * z
            safe = torch.abs(z2) > 1.0e-28
            inv = torch.zeros_like(z)
            if bool(torch.any(safe).item()):
                inv[safe] = (1.0 / 3.0) / z2[safe]
            z = (2.0 / 3.0) * z + inv

        roots = _root_tensor(device)
        distances = torch.stack([torch.abs(z - root_val) for root_val in roots], dim=0)
        mins, indices = torch.min(distances, dim=0)
        polished_root = torch.full_like(root_map, -1)
        polished_root = torch.where(mins < (10.0 * tol), indices.to(torch.int16), polished_root)
        polished_root = torch.where(root_map >= 0, indices.to(torch.int16), polished_root)

    if iter_replay_threshold > 0:
        suspect = build_boundary_mask(polished_root, iter_replay_dilation)
        suspect |= iter_map >= iter_replay_threshold
        suspect |= polished_root != root_map
        row_idx, col_idx, root_sel, iter_sel = replay_reference_newton_selected(x, y, suspect, tol, max_iter)
        if row_idx.numel() > 0:
            polished_root[row_idx, col_idx] = root_sel
            iter_map[row_idx, col_idx] = iter_sel

    root_map_out = polished_root.to(torch.int8)
    root_counts = np.array([(root_map_out == idx).sum().item() for idx in range(3)], dtype=np.int64)
    converged_total = int((root_map_out >= 0).sum().item())
    iter_total = int(iter_map[root_map_out >= 0].to(torch.int64).sum().item()) if converged_total else 0
    if not return_downsample:
        return None, None, converged_total, iter_total, root_counts
    small_root, small_iters = downsample_tile(root_map_out, iter_map, factor)
    return small_root, small_iters, converged_total, iter_total, root_counts
