#ifndef MODELDATA_H
#define MODELDATA_H

typedef struct PathData {
  float points[50];
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
