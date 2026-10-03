/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared vectors from and storing to global memory.
 */

/**
 * @brief Loads data from global memory into shared memory vector.
 *
 * This function loads data from a global memory location pointed to by `src` into a shared memory vector `dst`.
 * It calculates the number of elements that can be transferred in one operation based on the size ratio of `float4` to the data type of `SV`.
 * The function ensures coalesced memory access and efficient use of bandwidth by dividing the work among threads in a warp.
 *
 * @tparam SV Shared vector type, must satisfy ducks::sv::all concept.
 * @param dst Reference to the shared vector where the data will be loaded.
 * @param src Pointer to the global memory location from where the data will be loaded.
 */
template<typename SV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
load(threadgroup SV &dst, thread const GL &_src, thread const coord &idx, const int threadIdx) {
    using U = typename GL::dtype;
    using read_vector = ReadVector<1>;
    constexpr int elem_per_transfer = sizeof(read_vector) / sizeof(typename SV::dtype);
    constexpr int total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    device U *src = (device U*)&_src.template get<SV>(idx);
    
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length)
            *(threadgroup read_vector*)&dst[i*elem_per_transfer] = *(device read_vector*)&src[i*elem_per_transfer];
    }
}
 
/**
 * @brief Stores data from a shared memory vector to global memory.
 *
 * This function stores data from a shared memory vector `src` to a global memory location pointed to by `dst`.
 * Similar to the load function, it calculates the number of elements that can be transferred in one operation based on the size ratio of `float4` to the data type of `SV`.
 * The function ensures coalesced memory access and efficient use of bandwidth by dividing the work among threads in a warp.
 *
 * @tparam SV Shared vector type, must satisfy ducks::sv::all concept.
 * @param dst Pointer to the global memory location where the data will be stored.
 * @param src Reference to the shared vector from where the data will be stored.
 */
template<typename SV, typename GL>
METAL_FUNC static typename metal::enable_if<ducks::is_shared_vector<SV>(), void>::type
store(thread const GL &_dst, threadgroup const SV &src, thread const coord &idx, const int threadIdx) {
    using read_vector = ReadVector<1>;
    using U = typename GL::dtype;
    constexpr int elem_per_transfer = sizeof(read_vector) / sizeof(typename SV::dtype);
    constexpr int total_calls = SV::length / elem_per_transfer; // guaranteed to divide
    device U *dst = (device U*)&_dst.template get<SV>(idx);
    
    metal::simdgroup_barrier(metal::mem_flags::mem_none);
    #pragma clang loop unroll(full)
    for(int i = laneid(threadIdx); i < total_calls; i+= GROUP_THREADS) {
        if(i * elem_per_transfer < src.length)
            *(device read_vector*)&dst[i*elem_per_transfer] = *(threadgroup read_vector*)&src[i*elem_per_transfer]; // lmao it's identical
    }
}
