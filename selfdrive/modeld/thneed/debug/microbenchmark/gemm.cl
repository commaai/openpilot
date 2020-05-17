__kernel void myGEMM1(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
  const int globalRow = get_global_id(0); // Row ID of C (0..M)
  const int globalCol = get_global_id(1); // Col ID of C (0..N)

  float acc = 0.0f;
  for (int k=0; k<K; k++) {
    acc += A[k*M + globalRow] * B[globalCol*K + k];
  }

  C[globalCol*M + globalRow] = acc;
}

