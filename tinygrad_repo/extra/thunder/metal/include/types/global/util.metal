#pragma once

namespace mittens {
namespace ducks {
namespace g {
    
    //template<int d> concept cdim = (d > 0); // represents a compile-time dimension
    //template<int d> concept rdim = (d == -1); // represents a runtime dimension
    
    template<int d>
    struct compiled_dim {
        static_assert(d > 0, "Invalid compile-time dimension value"); // Replace `cdim` concept check
        static constant constexpr uint32_t v = d;
        
        METAL_FUNC compiled_dim(thread const metal::nullptr_t &_) {}
        
        METAL_FUNC constexpr operator uint32_t() const { return v; }
    };
    
    struct runtime_dim {
        uint32_t v;
        METAL_FUNC runtime_dim(thread const uint32_t &_v) : v(_v) {}
        METAL_FUNC operator uint32_t() const { return v; }
    };
    
    template<int d> using make_dim_t = metal::conditional_t<d == -1, runtime_dim, compiled_dim<d>>;
    template<int d> using make_arg_t = metal::conditional_t<d == -1, size_t, metal::nullptr_t>; // we pass runtime dims as size_t, comptime dims as nullptr_t
    
}
}

struct coord { // essentially a named int4 for tensor coordinates.
    int b, d, r, c;
    METAL_FUNC coord(int _b, int _d, int _r, int _c) : b(_b), d(_d), r(_r), c(_c) {}
    METAL_FUNC coord(        int _d, int _r, int _c) : b( 0), d(_d), r(_r), c(_c) {}
    METAL_FUNC coord(                int _r, int _c) : b( 0), d( 0), r(_r), c(_c) {}
    METAL_FUNC coord(                        int _c) : b( 0), d( 0), r( 0), c(_c) {}
    METAL_FUNC coord(                              ) : b( 0), d( 0), r( 0), c( 0) {}
    METAL_FUNC coord(thread const coord &other) : b(other.b), d(other.d), r(other.r), c(other.c) {}
    METAL_FUNC coord(thread const int4 &other) : b(other.x), d(other.y), r(other.z), c(other.w) {}
    METAL_FUNC operator int4() const { return int4(b, d, r, c); }
};

}
