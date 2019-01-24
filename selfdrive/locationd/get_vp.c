int get_intersections(double *lines, double *intersections, long long n) {
  double D, Dx, Dy;
  double x, y;
  double *L1, *L2;
  int k = 0;
  for (int i=0; i < n; i++) {
    for (int j=0; j < n; j++) {
      L1 = lines + i*3;
      L2 = lines + j*3;
      D = L1[0] * L2[1] - L1[1] * L2[0];
      Dx = L1[2] * L2[1] - L1[1] * L2[2];
      Dy = L1[0] * L2[2] - L1[2] * L2[0];
      // only intersect lines from different quadrants and only left-right crossing
      if ((D != 0) && (L1[0]*L2[0]*L1[1]*L2[1] < 0) && (L1[1]*L2[1] < 0)){
        x = Dx / D;
        y = Dy / D;
        if ((0 < x) &&
            (x < W) &&
            (0 < y) &&
            (y < H)){
          intersections[k*2 + 0] = x;
          intersections[k*2 + 1] = y;
          k++;
        }
      }
    }
  }
  return k;
}

void increment_grid(double *grid, double *lines, long long n) {
  double *intersections = (double*) malloc(n*n*2*sizeof(double));
  int y, x, k;
  k = get_intersections(lines, intersections, n);
  for (int i=0; i < k; i++) {
    x = (int) (intersections[i*2 + 0] + 0.5);
    y = (int) (intersections[i*2 + 1] + 0.5);
    grid[y*(W+1) + x] += 1.;
  }
  free(intersections);
}
