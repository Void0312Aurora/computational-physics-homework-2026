#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "types.hpp"

namespace project2::nextgen_cpu {

class PageArena {
public:
    struct Marker {
        std::size_t block_cursor = 0;
    };

    explicit PageArena(std::size_t page_size_blocks, std::size_t reserve_pages = 1)
        : page_size_blocks_(std::max<std::size_t>(page_size_blocks, 1)) {
        storage_.resize(page_size_blocks_ * std::max<std::size_t>(reserve_pages, 1), 0);
    }

    [[nodiscard]] Marker marker() const {
        return Marker{block_cursor_};
    }

    void rewind(Marker marker) {
        if (marker.block_cursor > block_cursor_) {
            throw std::runtime_error("cannot rewind PageArena forward");
        }
        block_cursor_ = marker.block_cursor;
    }

    void reset() {
        block_cursor_ = 0;
    }

    [[nodiscard]] BigHandle allocate(std::size_t capacity_blocks, Domain domain = Domain::canonical) {
        const std::size_t requested = std::max<std::size_t>(capacity_blocks, 1);
        const std::size_t start = block_cursor_;
        ensure_capacity(start + requested);
        block_cursor_ += requested;
        return BigHandle{
            .page_id = start / page_size_blocks_,
            .offset_blocks = start % page_size_blocks_,
            .used_blocks = 0,
            .capacity_blocks = requested,
            .sign = Sign::zero,
            .domain = domain,
        };
    }

    void zero_fill(const BigHandle& handle) {
        std::fill_n(data(handle), handle.capacity_blocks, 0);
    }

    [[nodiscard]] block_t* data(const BigHandle& handle) {
        return storage_.data() + start_index(handle);
    }

    [[nodiscard]] const block_t* data(const BigHandle& handle) const {
        return storage_.data() + start_index(handle);
    }

    [[nodiscard]] std::size_t page_size_blocks() const {
        return page_size_blocks_;
    }

    [[nodiscard]] std::size_t blocks_in_use() const {
        return block_cursor_;
    }

    [[nodiscard]] std::size_t pages_in_use() const {
        return (block_cursor_ + page_size_blocks_ - 1) / page_size_blocks_;
    }

private:
    [[nodiscard]] std::size_t start_index(const BigHandle& handle) const {
        if (!handle.is_allocated()) {
            throw std::runtime_error("BigHandle is not allocated");
        }
        const std::size_t index = handle.page_id * page_size_blocks_ + handle.offset_blocks;
        if (index + handle.capacity_blocks > storage_.size()) {
            throw std::runtime_error("BigHandle points outside PageArena storage");
        }
        return index;
    }

    void ensure_capacity(std::size_t required_blocks) {
        if (required_blocks <= storage_.size()) {
            return;
        }
        const std::size_t required_pages = (required_blocks + page_size_blocks_ - 1) / page_size_blocks_;
        storage_.resize(required_pages * page_size_blocks_, 0);
    }

    std::size_t page_size_blocks_ = 0;
    std::size_t block_cursor_ = 0;
    std::vector<block_t> storage_;
};

}  // namespace project2::nextgen_cpu
