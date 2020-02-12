bool sane(double track [K + 1][5]) {
  double diffs_x [K-1];
  double diffs_y [K-1];
  int i;
  for (i = 0; i < K-1; i++) {
    diffs_x[i] = fabs(track[i+2][2] - track[i+1][2]);
    diffs_y[i] = fabs(track[i+2][3] - track[i+1][3]);
  }
  for (i = 1; i < K-1; i++) {
    if (((diffs_x[i] > 0.05 or diffs_x[i-1] > 0.05) and
         (diffs_x[i] > 2*diffs_x[i-1] or
          diffs_x[i] < .5*diffs_x[i-1])) or
        ((diffs_y[i] > 0.05 or diffs_y[i-1] > 0.05) and
	 (diffs_y[i] > 2*diffs_y[i-1] or
          diffs_y[i] < .5*diffs_y[i-1]))){
      return false;
    } 
  }
  return true;
}

void merge_features(double *tracks, double *features, long long *empty_idxs) {
  double feature_arr [3000][5];
  memcpy(feature_arr, features, 3000 * 5 * sizeof(double));
  double track_arr [6000][K + 1][5];
  memcpy(track_arr, tracks, (K+1) * 6000 * 5 * sizeof(double));
  int match;
  int empty_idx = 0;
  int idx;
  for (int i = 0; i < 3000; i++) {
    match = feature_arr[i][4];
    if (track_arr[match][0][1] == match and track_arr[match][0][2] == 0){
      track_arr[match][0][0] = track_arr[match][0][0] + 1;
      track_arr[match][0][1] = feature_arr[i][1];
      track_arr[match][0][2] = 1;
      idx = track_arr[match][0][0];
      memcpy(track_arr[match][idx], feature_arr[i], 5 * sizeof(double));
      if (idx == K){
        // label complete
        track_arr[match][0][3] = 1;
	if (sane(track_arr[match])){
          // label valid
          track_arr[match][0][4] = 1; 
	}
      }		
    } else {
      // gen new track with this feature
      track_arr[empty_idxs[empty_idx]][0][0] = 1;
      track_arr[empty_idxs[empty_idx]][0][1] = feature_arr[i][1];
      track_arr[empty_idxs[empty_idx]][0][2] = 1;
      memcpy(track_arr[empty_idxs[empty_idx]][1], feature_arr[i], 5 * sizeof(double));
      empty_idx = empty_idx + 1;
    } 
  }
  memcpy(tracks, track_arr, (K+1) * 6000 * 5 * sizeof(double));
}
