import cv2
import numpy as np

image_path = "img\src\\43.png"

def box_extraction(original_img_path, cropped_dir_path):

    print("Reading image...")
    print("Image source path: ",original_img_path)
    # Read the image
    img = cv2.imread(original_img_path, 0)
    # Now, thresholding the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Now, invert the image
    img_bin = 255-img_bin

    print("Storing binary image to img/output/image_bin.jpg")
    cv2.imwrite("img/output/image_bin.jpg",img_bin)

    # Vertical kernel
    vertical_kernel_length=np.array(img).shape[1]//20
    print("vertical kernel length : ",vertical_kernel_length)
    # A verticle kernel of (1 X vertical_kernel_length), which will detect all the vertical lines from the image.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))
    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(img_temp1, vertical_kernel, iterations=3)
    cv2.imwrite("img/output/vertical_lines.jpg",vertical_lines_img)

    # Horizontal kernel
    hori_kernel_length=np.array(img).shape[0]//55
    print("horizontal kernel length : ",hori_kernel_length)
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hori_kernel_length, 1))
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("img/output/horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters for combination
    alpha = 0.5                 # percentage of vertical lines
    beta = 1.0 - alpha          # percentage of horizontal lines
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # img_final_bin = 255-img_final_bin
    # See vertical and horizontal lines in the image which is used to find boxes
    print("Image which only contains boxes: img/output/img_final_bin.jpg")
    cv2.imwrite("img/output/img_final_bin.jpg",img_final_bin)

box_extraction(image_path,"./img/output/")