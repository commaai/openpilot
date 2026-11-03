"""Camera-checked labels of launches at traffic lights (base replays, 20 Hz).

Found by exp_green.py: the car is slow (< 4 m/s) with the stop circuit on in the 10 s before, then it speeds up by
> 3 m/s. Checked on the viewer thumbnails (every 10 frames), contact sheets in $WEIGHT_LAB_DATA/exp/green/*.jpg.

GREEN: (replay name, frame of the first thumbnail with a green light ahead, launch frame). The light was red
before. The real change is up to 10 frames before the given frame. 'lead' = a car waits in front.
RTOR: the light ahead stays red and the car goes anyway (turn on red, done by the driver). (name, launch frame).
STOP_EXTRA: STOP signs that are not in exp_stop_labels.STOP. (name, first frame, last frame).
"""

GREEN = [
  ('f7_5', 240, 248),      # creeping at 0.5 m/s
  ('f7_8', 680, 686),
  ('f7_14', 1020, 1032),
  ('fd_2', 210, 204),      # car moves at about the same time as the green thumbnail
  ('fd_2', 770, 785),
  ('fd_3', 660, 667),
  ('fd_5', 1170, 1178),
  ('fd_8', 260, 272),
  ('fe_2', 600, 612),      # creeping at 0.4 m/s
  ('fe_6', 640, 648),
  ('s606_6', 540, 559),
  ('s606_10', 10, 45),     # lead; green is between s606_9 f1190 and s606_10 f10
  ('s606_10', 1010, 1040),  # lead
  ('s608_6', 670, 665),    # creeping at 0.05 m/s
  ('s608_17', 730, 736),   # lead
  ('s608_18', 1090, 1112),
  ('s608_19', 490, 524),
  ('s608_25', 990, 999),
  ('id4_6', 780, 800),     # lead, day
]

RTOR = [('f7_9', 597), ('f7_16', 317), ('f7_16', 1054), ('fd_1', 893), ('s606_1', 1012), ('s606_4', 554), ('s606_8', 283),
        ('s606_8', 992)]

STOP_EXTRA = [('s606_7', 450, 600), ('s608_14', 0, 140)]

# Red-light waits that are not in exp_stop_labels.RED (found on a sheet of all standstills >= 1 s, exp/green/stand_*.jpg).
# All frames are before the green. s606_5: railroad gate down with red lights.
RED_EXTRA = [('fd_2', 526, 765), ('s606_1', 9, 290), ('s606_4', 1119, 1199), ('s606_5', 0, 1199), ('s606_6', 0, 530),
             ('s606_10', 663, 1000), ('s608_18', 799, 1085), ('s608_25', 11, 89), ('s608_25', 217, 289),
             ('s608_25', 339, 803), ('s608_25', 851, 985)]
