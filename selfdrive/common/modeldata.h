#ifndef MODELDATA_H
#define MODELDATA_H

#define MODEL_PATH_DISTANCE 50

typedef struct PathData {
  float points[MODEL_PATH_DISTANCE];
  float prob;
  float std;
} PathData;

typedef struct LeadData {
  float dist;
  float prob;
  float std;
} LeadData;

typedef struct ModelData {
  PathData path;
  PathData left_lane;
  PathData right_lane;
  LeadData lead;
} ModelData;

#endif
