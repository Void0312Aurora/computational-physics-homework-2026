from __future__ import annotations

import math
from typing import Any

import numpy as np


MASK64 = (1 << 64) - 1
UINT64_RANGE = 1 << 64
DOUBLE_UNIT = 1.0 / (1 << 53)
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def _rotl(x: int, k: int) -> int:
    return ((x << k) & MASK64) | (x >> (64 - k))


def _shape_from_size(size: int | tuple[int, ...]) -> tuple[int, ...]:
    return (size,) if isinstance(size, int) else tuple(size)


def _check_int_bounds(low: int, high: int) -> None:
    if low < INT64_MIN or high - 1 > INT64_MAX:
        raise ValueError("integer bounds must fit in signed int64")


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next_uint64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64


class Xoshiro256StarStar:
    """Small self-contained xoshiro256** PRNG with a NumPy-like subset."""

    name = "custom_xoshiro256ss"
    JUMP = (
        0x180EC6D33CFD0ABA,
        0xD5A61266F0C9392C,
        0xA9582618E03FC9AA,
        0x39ABDC4529B1661C,
    )
    LONG_JUMP = (
        0x76E15D3EFEFDCBBF,
        0xC5004E441C522FB3,
        0x77710069854EE241,
        0x39109BB02ACBE635,
    )

    def __init__(self, seed: int):
        sm = SplitMix64(seed)
        self.s = [sm.next_uint64() for _ in range(4)]
        if not any(self.s):
            self.s[0] = 1

    def next_uint64(self) -> int:
        s0, s1, s2, s3 = self.s
        result = (_rotl((s1 * 5) & MASK64, 7) * 9) & MASK64
        t = (s1 << 17) & MASK64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 45)
        self.s = [s0 & MASK64, s1 & MASK64, s2 & MASK64, s3 & MASK64]
        return result

    def random_raw(self, size: int | tuple[int, ...] | None = None) -> np.ndarray | int:
        if size is None:
            return self.next_uint64()
        shape = _shape_from_size(size)
        out = np.empty(shape, dtype=np.uint64)
        flat = out.ravel()
        for i in range(flat.size):
            flat[i] = self.next_uint64()
        return out

    def copy(self) -> "Xoshiro256StarStar":
        other = self.__class__.__new__(self.__class__)
        other.s = self.s.copy()
        return other

    def _apply_jump(self, constants: tuple[int, ...]) -> "Xoshiro256StarStar":
        jumped = [0, 0, 0, 0]
        for jump_constant in constants:
            for bit in range(64):
                if jump_constant & (1 << bit):
                    jumped = [value ^ state for value, state in zip(jumped, self.s)]
                self.next_uint64()
        self.s = [value & MASK64 for value in jumped]
        return self

    def jump(self) -> "Xoshiro256StarStar":
        return self._apply_jump(self.JUMP)

    def long_jump(self) -> "Xoshiro256StarStar":
        return self._apply_jump(self.LONG_JUMP)

    def jumped(self) -> "Xoshiro256StarStar":
        return self.copy().jump()

    def long_jumped(self) -> "Xoshiro256StarStar":
        return self.copy().long_jump()

    def _bounded_uint64(self, width: int) -> int:
        if width > UINT64_RANGE:
            raise ValueError("integer range is too wide for one uint64 draw")
        limit = (UINT64_RANGE // width) * width
        while True:
            value = self.next_uint64()
            if value < limit:
                return value % width

    def random(self, size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        if size is None:
            return ((self.next_uint64() >> 11) & ((1 << 53) - 1)) * DOUBLE_UNIT
        shape = _shape_from_size(size)
        total = math.prod(shape)
        data = np.fromiter(
            (((self.next_uint64() >> 11) & ((1 << 53) - 1)) * DOUBLE_UNIT for _ in range(total)),
            dtype=float,
            count=total,
        )
        return data.reshape(shape)

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | float:
        return low + (high - low) * self.random(size)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | int:
        if high is None:
            high = low
            low = 0
        width = high - low
        if width <= 0:
            raise ValueError("high must be greater than low")
        _check_int_bounds(low, high)
        if size is None:
            return low + self._bounded_uint64(width)
        shape = _shape_from_size(size)
        values = np.empty(shape, dtype=np.int64)
        flat = values.ravel()
        for i in range(flat.size):
            flat[i] = low + self._bounded_uint64(width)
        return values

    def choice(self, values: np.ndarray, size: int | tuple[int, ...] | None = None) -> np.ndarray | Any:
        idx = self.integers(0, len(values), size)
        return values[idx]


class ParkMiller:
    """HW/11-compatible Park-Miller minimal-standard LCG."""

    name = "custom_park_miller_lcg"
    modulus = 2147483647
    multiplier = 16807

    def __init__(self, seed: int):
        self.state = int(seed) % self.modulus
        if self.state <= 0:
            self.state += self.modulus - 1

    def next_uint31(self) -> int:
        self.state = (self.multiplier * self.state) % self.modulus
        return self.state

    def _bounded_int(self, width: int) -> int:
        raw_range = self.modulus - 1
        if width > raw_range:
            raise ValueError("integer range is too wide for Park-Miller output")
        limit = (raw_range // width) * width
        while True:
            value = self.next_uint31() - 1
            if value < limit:
                return value % width

    def random(self, size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        if size is None:
            return self.next_uint31() / self.modulus
        shape = _shape_from_size(size)
        out = np.empty(shape, dtype=float)
        flat = out.ravel()
        for i in range(flat.size):
            flat[i] = self.next_uint31() / self.modulus
        return out

    def uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | float:
        return low + (high - low) * self.random(size)

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
    ) -> np.ndarray | int:
        if high is None:
            high = low
            low = 0
        width = high - low
        if width <= 0:
            raise ValueError("high must be greater than low")
        _check_int_bounds(low, high)
        if size is None:
            return low + self._bounded_int(width)
        shape = _shape_from_size(size)
        values = np.empty(shape, dtype=np.int64)
        flat = values.ravel()
        for i in range(flat.size):
            flat[i] = low + self._bounded_int(width)
        return values

    def choice(self, values: np.ndarray, size: int | tuple[int, ...] | None = None) -> np.ndarray | Any:
        idx = self.integers(0, len(values), size)
        return values[idx]

def make_rng(kind: str, seed: int):
    if kind == "numpy_pcg64":
        return np.random.default_rng(seed)
    if kind == "xoshiro256ss":
        return Xoshiro256StarStar(seed)
    if kind == "park_miller":
        return ParkMiller(seed)
    raise ValueError(f"unknown RNG kind: {kind}")


def rng_label(kind: str) -> str:
    return {
        "numpy_pcg64": "NumPy PCG64",
        "xoshiro256ss": "custom xoshiro256**",
        "park_miller": "custom Park-Miller LCG",
    }[kind]
