/*
 * Copyright (C) 2009 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_UI_PRIVATE_REGION_HELPER_H
#define ANDROID_UI_PRIVATE_REGION_HELPER_H

#include <limits>
#include <stdint.h>
#include <sys/types.h>

#include <limits>

namespace android {
// ----------------------------------------------------------------------------

template<typename RECT>
class region_operator
{
public:
    typedef typename RECT::value_type TYPE;
    static const TYPE max_value = std::numeric_limits<TYPE>::max();

    /*
     * Common boolean operations:
     * value is computed as 0b101 op 0b110
     *    other boolean operation are possible, simply compute
     *    their corresponding value with the above formulae and use
     *    it when instantiating a region_operator.
     */
    static const uint32_t LHS = 0x5;  // 0b101
    static const uint32_t RHS = 0x6;  // 0b110
    enum {
        op_nand = LHS & ~RHS,
        op_and  = LHS &  RHS,
        op_or   = LHS |  RHS,
        op_xor  = LHS ^  RHS
    };

    struct region {
        RECT const* rects;
        size_t count;
        TYPE dx;
        TYPE dy;
        inline region(const region& rhs) 
            : rects(rhs.rects), count(rhs.count), dx(rhs.dx), dy(rhs.dy) { }
        inline region(RECT const* _r, size_t _c)
            : rects(_r), count(_c), dx(), dy() { }
        inline region(RECT const* _r, size_t _c, TYPE _dx, TYPE _dy)
            : rects(_r), count(_c), dx(_dx), dy(_dy) { }
    };

    class region_rasterizer {
        friend class region_operator;
        virtual void operator()(const RECT& rect) = 0;
    public:
        virtual ~region_rasterizer() { }
    };
    
    inline region_operator(uint32_t op, const region& lhs, const region& rhs)
        : op_mask(op), spanner(lhs, rhs) 
    {
    }

    void operator()(region_rasterizer& rasterizer) {
        RECT current(Rect::EMPTY_RECT);
        do {
            SpannerInner spannerInner(spanner.lhs, spanner.rhs);
            int inside = spanner.next(current.top, current.bottom);
            spannerInner.prepare(inside);
            do {
                int inner_inside = spannerInner.next(current.left, current.right);
                if ((op_mask >> inner_inside) & 1) {
                    if (current.left < current.right && 
                            current.top < current.bottom) {
                        rasterizer(current);
                    }
                }
            } while(!spannerInner.isDone());
        } while(!spanner.isDone());
    }

private:    
    uint32_t op_mask;

    class SpannerBase
    {
    public:
        SpannerBase()
            : lhs_head(max_value), lhs_tail(max_value),
              rhs_head(max_value), rhs_tail(max_value) {
        }

        enum {
            lhs_before_rhs   = 0,
            lhs_after_rhs    = 1,
            lhs_coincide_rhs = 2
        };

    protected:
        TYPE lhs_head;
        TYPE lhs_tail;
        TYPE rhs_head;
        TYPE rhs_tail;

        inline int next(TYPE& head, TYPE& tail,
                bool& more_lhs, bool& more_rhs) 
        {
            int inside;
            more_lhs = false;
            more_rhs = false;
            if (lhs_head < rhs_head) {
                inside = lhs_before_rhs;
                head = lhs_head;
                if (lhs_tail <= rhs_head) {
                    tail = lhs_tail;
                    more_lhs = true;
                } else {
                    lhs_head = rhs_head;
                    tail = rhs_head;
                }
            } else if (rhs_head < lhs_head) {
                inside = lhs_after_rhs;
                head = rhs_head;
                if (rhs_tail <= lhs_head) {
                    tail = rhs_tail;
                    more_rhs = true;
                } else {
                    rhs_head = lhs_head;
                    tail = lhs_head;
                }
            } else {
                inside = lhs_coincide_rhs;
                head = lhs_head;
                if (lhs_tail <= rhs_tail) {
                    tail = rhs_head = lhs_tail;
                    more_lhs = true;
                }
                if (rhs_tail <= lhs_tail) {
                    tail = lhs_head = rhs_tail;
                    more_rhs = true;
                }
            }
            return inside;
        }
    };

    class Spanner : protected SpannerBase 
    {
        friend class region_operator;
        region lhs;
        region rhs;

    public:
        inline Spanner(const region& _lhs, const region& _rhs)
        : lhs(_lhs), rhs(_rhs)
        {
            if (lhs.count) {
                SpannerBase::lhs_head = lhs.rects->top      + lhs.dy;
                SpannerBase::lhs_tail = lhs.rects->bottom   + lhs.dy;
            }
            if (rhs.count) {
                SpannerBase::rhs_head = rhs.rects->top      + rhs.dy;
                SpannerBase::rhs_tail = rhs.rects->bottom   + rhs.dy;
            }
        }

        inline bool isDone() const {
            return !rhs.count && !lhs.count;
        }

        inline int next(TYPE& top, TYPE& bottom) 
        {
            bool more_lhs = false;
            bool more_rhs = false;
            int inside = SpannerBase::next(top, bottom, more_lhs, more_rhs);
            if (more_lhs) {
                advance(lhs, SpannerBase::lhs_head, SpannerBase::lhs_tail);
            }
            if (more_rhs) {
                advance(rhs, SpannerBase::rhs_head, SpannerBase::rhs_tail);
            }
            return inside;
        }

    private:
        static inline 
        void advance(region& reg, TYPE& aTop, TYPE& aBottom) {
            // got to next span
            size_t count = reg.count;
            RECT const * rects = reg.rects;
            RECT const * const end = rects + count;
            const int top = rects->top;
            while (rects != end && rects->top == top) {
                rects++;
                count--;
            }
            if (rects != end) {
                aTop    = rects->top    + reg.dy;
                aBottom = rects->bottom + reg.dy;
            } else {
                aTop    = max_value;
                aBottom = max_value;
            }
            reg.rects = rects;
            reg.count = count;
        }
    };

    class SpannerInner : protected SpannerBase 
    {
        region lhs;
        region rhs;
        
    public:
        inline SpannerInner(const region& _lhs, const region& _rhs)
            : lhs(_lhs), rhs(_rhs)
        {
        }

        inline void prepare(int inside) {
            if (inside == SpannerBase::lhs_before_rhs) {
                if (lhs.count) {
                    SpannerBase::lhs_head = lhs.rects->left  + lhs.dx;
                    SpannerBase::lhs_tail = lhs.rects->right + lhs.dx;
                }
                SpannerBase::rhs_head = max_value;
                SpannerBase::rhs_tail = max_value;
            } else if (inside == SpannerBase::lhs_after_rhs) {
                SpannerBase::lhs_head = max_value;
                SpannerBase::lhs_tail = max_value;
                if (rhs.count) {
                    SpannerBase::rhs_head = rhs.rects->left  + rhs.dx;
                    SpannerBase::rhs_tail = rhs.rects->right + rhs.dx;
                }
            } else {
                if (lhs.count) {
                    SpannerBase::lhs_head = lhs.rects->left  + lhs.dx;
                    SpannerBase::lhs_tail = lhs.rects->right + lhs.dx;
                }
                if (rhs.count) {
                    SpannerBase::rhs_head = rhs.rects->left  + rhs.dx;
                    SpannerBase::rhs_tail = rhs.rects->right + rhs.dx;
                }
            }
        }

        inline bool isDone() const {
            return SpannerBase::lhs_head == max_value && 
                   SpannerBase::rhs_head == max_value;
        }

        inline int next(TYPE& left, TYPE& right) 
        {
            bool more_lhs = false;
            bool more_rhs = false;
            int inside = SpannerBase::next(left, right, more_lhs, more_rhs);
            if (more_lhs) {
                advance(lhs, SpannerBase::lhs_head, SpannerBase::lhs_tail);
            }
            if (more_rhs) {
                advance(rhs, SpannerBase::rhs_head, SpannerBase::rhs_tail);
            }
            return inside;
        }

    private:
        static inline 
        void advance(region& reg, TYPE& left, TYPE& right) {
            if (reg.rects && reg.count) {
                const int cur_span_top = reg.rects->top;
                reg.rects++;
                reg.count--;
                if (!reg.count || reg.rects->top != cur_span_top) {
                    left  = max_value;
                    right = max_value;
                } else {
                    left  = reg.rects->left  + reg.dx;
                    right = reg.rects->right + reg.dx;
                }
            }
        }
    };

    Spanner spanner;
};

// ----------------------------------------------------------------------------
};

#endif /* ANDROID_UI_PRIVATE_REGION_HELPER_H */
