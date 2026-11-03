"""Camera-checked labels of stop-circuit episodes: painted STOP lines vs traffic lights (base replays, 20 Hz).

Each entry is (replay name, first frame, last frame) of a stop-circuit episode (u69+u248+u201+u239 > 0.3).
Checked on the viewer thumbnails (every 10 frames), contact sheets in $WEIGHT_LAB_DATA/exp/stop/*.jpg.
STOP = a painted STOP word or STOP sign is in view before the line. RED = a red light is in view before the stop.
Episodes that are only a lead car, a construction flagger or unclear are in neither list.
"""

STOP = [
  # 000005fe (night)
  ('fe_0', 274, 352), ('fe_7', 745, 914), ('fe_8', 93, 278), ('fe_8', 650, 886),
  # 000005fd (night)
  ('fd_0', 545, 672), ('fd_0', 869, 1010), ('fd_4', 844, 976), ('fd_5', 215, 293),
  # 000005f7 (night)
  ('f7_10', 644, 790), ('f7_11', 493, 651), ('f7_11', 1079, 1199), ('f7_12', 408, 554), ('f7_12', 724, 885),
  ('f7_12', 1094, 1199), ('f7_13', 234, 362), ('f7_13', 1013, 1130), ('f7_14', 129, 282), ('f7_19', 405, 598),
  ('f7_19', 960, 1102), ('f7_20', 318, 451), ('f7_20', 858, 987), ('f7_21', 162, 289), ('f7_21', 690, 812),
  # 00000608 (night)
  ('s608_0', 1135, 1199), ('s608_14', 419, 565), ('s608_16', 6, 308), ('s608_21', 189, 287), ('s608_26', 675, 796),
  ('s608_27', 229, 388),
  # 00000606 (night): STOP painted before a parking gate
  ('s606_11', 298, 513),
]

RED = [
  ('fe_0', 527, 1140), ('fe_2', 447, 620), ('fe_6', 458, 664),
  ('fd_1', 640, 916), ('fd_2', 9, 216), ('fd_2', 418, 527), ('fd_3', 393, 671), ('fd_5', 619, 1188), ('fd_7', 1132, 1199),
  ('fd_8', 9, 289),
  ('f7_5', 113, 262), ('f7_8', 509, 704), ('f7_9', 397, 624), ('f7_14', 677, 1085), ('f7_16', 174, 327),
  ('f7_16', 920, 1067),
  ('s606_1', 853, 1033), ('s606_4', 328, 561), ('s606_4', 920, 1199), ('s606_9', 1024, 1199), ('s606_10', 483, 679),
  ('s608_6', 504, 687), ('s608_9', 787, 1073), ('s608_11', 319, 627), ('s608_17', 227, 763), ('s608_17', 1070, 1199),
  ('s608_19', 249, 531), ('s608_24', 1080, 1199),
  ('id4_6', 561, 839),
]
