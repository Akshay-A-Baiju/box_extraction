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

box_extraction(image_path,"./img/output/")