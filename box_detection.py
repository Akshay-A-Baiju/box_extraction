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

box_extraction(image_path,"./img/output/")