import os
import re
import numpy as np
import cv2


def show_image(label, image):
    height, width = image.shape[:2]
    new_width = int(width / 2)
    new_height = int(height / 2)
    show = cv2.resize(image, (new_width, new_height))
    cv2.imshow(label, show)


def validate(image_path, image, largest_contour):
    show_mask = cv2.drawContours(np.ones_like(image), [largest_contour], -1, (100, 100, 100), thickness=cv2.FILLED)
    alpha = 0.99
    mask_alpha = cv2.addWeighted(image, 1, show_mask, alpha, 0)
    cv2.drawContours(mask_alpha, [largest_contour], -1, (0, 0, 0), 2)  # Draw on the blank canvas

    show_image(image_path, mask_alpha)
    while 1 == 1:
        key = cv2.waitKey(0)
        if key == -1:
            quit()
        if key in (13, 27):
            cv2.destroyAllWindows()
            return key != 27


def get_largest_contour(base_image):
    image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(image, (5, 5), 0)


    adaptive_thresh = cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=19,  # Size of the pixel neighborhood
        C=3  # Constant subtracted from the mean
    )

    edges = cv2.Canny(adaptive_thresh, threshold1=50, threshold2=150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)

    mask = cv2.drawContours(np.zeros_like(base_image[:, :, 0]), [largest], -1, 255, thickness=cv2.FILLED)
    edges = cv2.Canny(mask, threshold1=50, threshold2=150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)

    return largest


def get_pixel_count(image, largest_contour):
    mask = cv2.drawContours(np.zeros_like(image[:, :, 0]), [largest_contour], -1, 255, thickness=cv2.FILLED)

    return cv2.countNonZero(mask)


def process_image(image_path):
    image = cv2.imread(image_path)
    largest_contour = get_largest_contour(image)

    pixel_count = get_pixel_count(image, largest_contour)
    is_valid = validate(image_path, image, largest_contour)

    print(format_csv(image_path, pixel_count, is_valid))


def get_files():
    base_dir = 'wing_photos'

    # List to store file paths
    image_files = []
    # Traverse all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file is an image (you can customize extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                image_files.append(os.path.join(root, file))
    return image_files


def format_csv(image_path, pixels, valid):
    return ""
    match = re.search(r"(\w{1,2})-(20|60|100)-(N?D)-(R[123])[\\/](\d+)(M|F)(R?)\.jpg", image_path)
    data = [match[1], match[2], match[3], match[4], match[6], match[5] + match[6], match[7] or "L", pixels, pixels * 0,
            valid]
    return ', '.join(str(x) for x in data)


for file in get_files():
    process_image(file)
#
# process_image("wing_photos/WC-20-ND-R1/11F.jpg");
