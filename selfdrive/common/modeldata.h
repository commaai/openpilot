#ifndef MODELDATA_H
#define MODELDATA_H

#define MODEL_PATH_DISTANCE 192
#define POLYFIT_DEGREE 4
#define SPEED_PERCENTILES 10
#define DESIRE_PRED_SIZE 32
#define OTHER_META_SIZE 4

typedef struct PathData {
  float points[MODEL_PATH_DISTANCE];
  float prob;
  float std;
  float stds[MODEL_PATH_DISTANCE];
  float poly[POLYFIT_DEGREE];
  float validLen;
} PathData;

typedef struct LeadData {
  float dist;
  float prob;
  float std;
  float rel_y;
  float rel_y_std;
  float rel_v;
  float rel_v_std;
  float rel_a;
  float rel_a_std;
} LeadData;

typedef struct ModelData {
  PathData path;
  PathData left_lane;
  PathData right_lane;
  LeadData lead;
  LeadData lead_future;
  float meta[OTHER_META_SIZE + DESIRE_PRED_SIZE];
  float speed[SPEED_PERCENTILES];
} ModelData;

#endif
