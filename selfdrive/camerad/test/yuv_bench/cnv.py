import numpy as np
import cv2  # pylint: disable=import-error

# img_bgr = np.zeros((874, 1164, 3), dtype=np.uint8)
# for y in range(874):
#   for k in range(1164*3):
#     img_bgr[y, k//3, k%3] = k ^ y

# cv2.imwrite("img_rgb.png", img_bgr)


cl = np.fromstring(open("out_cl.bin", "rb").read(), dtype=np.uint8)

cl_r = cl.reshape(874 * 3 // 2, -1)

cv2.imwrite("out_y.png", cl_r[:874])

cl_bgr = cv2.cvtColor(cl_r, cv2.COLOR_YUV2BGR_I420)
cv2.imwrite("out_cl.png", cl_bgr)
