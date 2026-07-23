/**
 * @file
 * @brief Wrappers for multimem operations
 */

#pragma once

namespace kittens {

enum class reduce_op {
    ADD = 0,
    MIN = 1,
    MAX = 2
};

enum class memory_model {
    WEAK = 0,
    STRONG = 1
};

template <typename T>
struct multimem;

template <>
struct multimem<int> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(int &dst, const int *src) {
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.s32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(int *dst, const int &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.s32 [%0], %1;"
                :: "l"(dst), "r"(src) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.s32 [%0], %1;"
                :: "l"(dst), "r"(src) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(int *dst, const int &src) {
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        } else if constexpr (Op == reduce_op::MIN) {
            asm volatile("multimem.red.release.sys.global.min.s32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        } else if constexpr (Op == reduce_op::MAX) {
            asm volatile("multimem.red.release.sys.global.max.s32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        }
    }
};

template <>
struct multimem<uint> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(uint &dst, const uint *src) {
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.u32 %0, [%1];"
                    : "=r"(dst) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(uint *dst, const uint &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.u32 [%0], %1;"
                :: "l"(dst), "r"(src) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.u32 [%0], %1;"
                :: "l"(dst), "r"(src) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(uint *dst, const uint &src) {
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        } else if constexpr (Op == reduce_op::MIN) {
            asm volatile("multimem.red.release.sys.global.min.u32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        } else if constexpr (Op == reduce_op::MAX) {
            asm volatile("multimem.red.release.sys.global.max.u32 [%0], %1;"
                : : "l"(dst), "r"(src) : "memory");
        }
    }
};

template <>
struct multimem<float> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(float &dst, const float *src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f32 ld_reduce operations");
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.f32 %0, [%1];"
                    : "=f"(dst) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.f32 %0, [%1];"
                    : "=f"(dst) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(float *dst, const float &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.f32 [%0], %1;"
                :: "l"(dst), "f"(src) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.f32 [%0], %1;"
                :: "l"(dst), "f"(src) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(float *dst, const float &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f32 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.f32 [%0], %1;"
                : : "l"(dst), "f"(src) : "memory");
        }
    }
};


template <>
struct multimem<float2> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(float2 &dst, const float2 *src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f32 ld_reduce operations");
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.v2.f32 {%0, %1}, [%2];"
                    : "=f"(dst.x), "=f"(dst.y) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.v2.f32 {%0, %1}, [%2];"
                    : "=f"(dst.x), "=f"(dst.y) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(float2 *dst, const float2 &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.v2.f32 [%0], {%1, %2};"
                :: "l"(dst), "f"(src.x), "f"(src.y) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.v2.f32 [%0], {%1, %2};"
                :: "l"(dst), "f"(src.x), "f"(src.y) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(float2 *dst, const float2 &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f32 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.v2.f32 [%0], {%1, %2};"
                : : "l"(dst), "f"(src.x), "f"(src.y) : "memory");
        }
    }
};

template <>
struct multimem<bf16> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(bf16 &dst, const bf16 *src) {
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.bf16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.acc::f32.bf16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.bf16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.bf16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.bf16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.bf16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(bf16 *dst, const bf16 &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.bf16 [%0], %1;"
                :: "l"(dst), "h"(*reinterpret_cast<const uint16_t *>(&src)) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.bf16 [%0], %1;"
                :: "l"(dst), "h"(*reinterpret_cast<const uint16_t *>(&src)) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(bf16 *dst, const bf16 &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for bf16 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.bf16 [%0], %1;"
                : : "l"(dst), "h"(*reinterpret_cast<const uint16_t *>(&src)) : "memory");
        }
    }
};

template <>
struct multimem<bf16_2> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(bf16_2 &dst, const bf16_2 *src) {
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.bf16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.acc::f32.bf16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.bf16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.bf16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.bf16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.bf16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(bf16_2 *dst, const bf16_2 &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.bf16x2 [%0], %1;"
                :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.bf16x2 [%0], %1;"
                :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(bf16_2 *dst, const bf16_2 &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for bf16_2 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.bf16x2 [%0], %1;"
                : : "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        }
    }
};

template <>
struct multimem<half> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(half &dst, const half *src) {
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.acc::f32.f16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.f16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.f16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.f16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.f16 %0, [%1];"
                    : "=h"(*reinterpret_cast<uint16_t *>(&dst)) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(half *dst, const half &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.f16 [%0], %1;"
                :: "l"(dst), "h"(*reinterpret_cast<const uint16_t *>(&src)) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.f16 [%0], %1;"
                :: "l"(dst), "h"(*reinterpret_cast<const uint16_t *>(&src)) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(half *dst, const half &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f16 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.f16 [%0], %1;"
                : : "l"(dst), "h"(*reinterpret_cast<const uint16_t *>(&src)) : "memory");
        }
    }
};

template <>
struct multimem<half_2> {
    template <reduce_op Op, memory_model M = memory_model::WEAK>
    __device__ static inline void ld_reduce(half_2 &dst, const half_2 *src) {
        if constexpr (Op == reduce_op::ADD) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.add.acc::f32.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.add.acc::f32.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MIN) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.min.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.min.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        } else if constexpr (Op == reduce_op::MAX) {
            if constexpr (M == memory_model::WEAK) {
                asm volatile("multimem.ld_reduce.weak.global.max.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            } else if constexpr (M == memory_model::STRONG) {
                asm volatile("multimem.ld_reduce.acquire.sys.global.max.f16x2 %0, [%1];"
                    : "=r"(*reinterpret_cast<uint32_t *>(&dst)) : "l"(src) : "memory");
            }
        }
    }
    template <memory_model M = memory_model::WEAK>
    __device__ static inline void st(half_2 *dst, const half_2 &src) {
        if constexpr (M == memory_model::WEAK) {
            asm volatile("multimem.st.weak.global.f16x2 [%0], %1;"
                :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        } else if constexpr (M == memory_model::STRONG) {
            asm volatile("multimem.st.release.sys.global.f16x2 [%0], %1;"
                :: "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        }
    }
    template <reduce_op Op>
    __device__ static inline void red(half_2 *dst, const half_2 &src) {
        static_assert(Op == reduce_op::ADD, "MIN/MAX are not supported for f16_2 red operations");
        if constexpr (Op == reduce_op::ADD) {
            asm volatile("multimem.red.release.sys.global.add.f16x2 [%0], %1;"
                : : "l"(dst), "r"(*reinterpret_cast<const uint32_t *>(&src)) : "memory");
        }
    }
};

} // namespace kittens
