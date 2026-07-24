/**
 * @file Group-level tcgen05 MMA operations.
*/

template<int trans_a, int n_trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    if(laneid() == 0) ::kittens::mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}

// no sem versions


template<int trans_a, int n_trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    if(laneid() == 0) ::kittens::mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}