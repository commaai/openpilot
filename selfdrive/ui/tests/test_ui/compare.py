import cv2


def pixel_by_pixel_comparison(image1_path, image2_path, threshold_value=30):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1.shape == img2.shape:
        # Computing the absolute difference between the images
        difference = cv2.absdiff(img1, img2)

        # Converting the difference image to grayscale
        gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

        # Applying a threshold to the grayscale difference image
        _, thresholded = cv2.threshold(gray_difference, threshold_value, 255, cv2.THRESH_BINARY)

        # Finding contours based on the threshold
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Drawing the contours on the images
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    raise
