#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define ORBD_KEYPOINTS 3000
#define ORBD_DESCRIPTOR_LENGTH 32
#define ORBD_HEIGHT 874
#define ORBD_WIDTH 1164
#define ORBD_FOCAL 910

// matches OrbFeatures from log.capnp
struct orb_features {
  // align this
  uint16_t n_corners;
  uint16_t xy[ORBD_KEYPOINTS][2];
  uint8_t octave[ORBD_KEYPOINTS];
  uint8_t des[ORBD_KEYPOINTS][ORBD_DESCRIPTOR_LENGTH];
  int16_t matches[ORBD_KEYPOINTS];
};

// forward declare this
struct pyramid;

// manage the pyramids in extractor.c
void init_gpyrs();
int extract_and_match_gpyrs(const uint8_t *img, struct orb_features *);
int extract_and_match(const uint8_t *img, struct pyramid *pyrs, struct pyramid *prev_pyrs, struct orb_features *);

#ifdef __cplusplus
}
#endif

#endif // EXTRACTOR_H
