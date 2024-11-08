import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib2 import Path

# Function to rotate the image without cropping
def rotate_image(image, angle):
    # Get image size
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w / 2, h / 2)

    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the sine and cosine of the rotation angle
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # Compute the new bounding dimensions of the image
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to account for the new dimensions
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)

    return rotated

# Function to crop the image to the bounding box of the edges
def crop_image_to_edges(image, edge_mask):

    crop_threshold = np.max(edge_mask) * 0.6
    crop_edge_mask = edge_mask > crop_threshold

    # Find the coordinates of the non-zero values in the edge mask
    coords = np.column_stack(np.where(crop_edge_mask))

    # Get the bounding box of the edges
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    # Crop the image based on the bounding box
    cropped_image = image[x_min:x_max, y_min:y_max]

    return cropped_image

# Function to crop the ROI of the image ( roi is the number and suit of the card)
def card_roi(cropped_img):
    # apply binary thresholding
    _, thresh = cv2.threshold(cropped_img, 70, 255, cv2.THRESH_BINARY)

    # crop the left top corner of the image
    number_roi = thresh[50:250, 20:170]
    suit_roi = thresh[260:440, 30:170]
    return number_roi, suit_roi

def template_matching(roi, templates, template_names):
    # Ensure ROI is of the right data type (uint8)
    roi = roi.astype(np.uint8)
    
    best_match_name = None
    best_match_score = -1
    matched_template = None
    
    for template, name in zip(templates, template_names):
        # Ensure the template is also uint8
        template = template.astype(np.uint8)
        
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_match_score:
            best_match_score = max_val
            best_match_name = name
            matched_template = template

    return best_match_name, best_match_score, matched_template


def load_templates(template_paths):
    templates = []
    template_names = []
    for path in template_paths:
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates.append(template)
            template_names.append(Path(path).stem)
    return templates, template_names

def plot_images(img, cropped_img, number_roi, suit_roi):
    plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")

    plt.subplot(2,2,2)
    plt.imshow(cropped_img, cmap='gray')
    plt.title("Rotated & Cropped Image")

    plt.subplot(2,2,3)
    plt.imshow(number_roi, cmap='gray')
    plt.title("ROI of the card number")

    plt.subplot(2,2,4)
    plt.imshow(suit_roi, cmap='gray')
    plt.title("ROI of the card suit")

    plt.tight_layout()
    plt.show()

# Main function to load, process, rotate, and crop the image
def main():
    # Ask user for image without the file extension
    name = input("Enter the name of the image: ")
    image_path = f"{name}"

    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the grayscale image
    gaussian_blur = np.array([[0, 0, 1, 2, 1, 0, 0],
                              [0, 3, 13, 22, 13, 3, 0],
                              [1, 13, 59, 97, 59, 13, 1],
                              [2, 22, 97, 159, 97, 22, 2],
                              [1, 13, 59, 97, 59, 13, 1],
                              [0, 3, 13, 22, 13, 3, 0],
                              [0, 0, 1, 2, 1, 0, 0]], dtype=np.float32) / 1003

    img_blurred = cv2.filter2D(img, cv2.CV_32F, gaussian_blur)

    # Sobel kernels
    gx = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    gy = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    # Apply Sobel filters to the blurred image
    sobel_x = cv2.filter2D(img_blurred, cv2.CV_32F, gx)
    sobel_y = cv2.filter2D(img_blurred, cv2.CV_32F, gy)

    # Calculate gradient magnitude and angle
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Convert radians to degrees and adjust angle range to [0, 180)
    gradient_angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi % 180

    # Threshold the gradient magnitude to keep significant edges
    mag_threshold = np.max(gradient_magnitude) * 0.3  # Adjust this value as needed
    edge_mask = gradient_magnitude > mag_threshold

    # Get edge gradient angles
    edge_angles = gradient_angle[edge_mask]

    # Create histogram of edge angles
    hist, bins = np.histogram(edge_angles, bins=180, range=(0, 180))

    # Find the dominant edge angle
    dominant_angle_index = np.argmax(hist)
    dominant_angle = (bins[dominant_angle_index] + bins[dominant_angle_index + 1]) / 2

    # Since gradient_angle corresponds to edge orientation, calculate rotation angle directly
    rotation_angle = 90 - dominant_angle

    # Check if the rotation angle is negative and adjust it
    if rotation_angle < 0:
        rotation_angle += 180

    # Rotate the image
    rotated_img = rotate_image(img, rotation_angle)

    # Sobel filters for the rotated image
    crop_sobel_x = cv2.filter2D(rotated_img, cv2.CV_32F, gx)
    crop_sobel_y = cv2.filter2D(rotated_img, cv2.CV_32F, gy)

    # Rotated image magnitude and angle
    crop_rotated_mag = np.sqrt(crop_sobel_x**2 + crop_sobel_y**2)

    # Crop the image
    cropped_img = crop_image_to_edges(rotated_img, crop_rotated_mag)

    number_roi, suit_roi = card_roi(cropped_img)
    
    # Load number and suit templates
    number_template_paths = ["templates/numbers/A.jpeg", "templates/numbers/2.jpeg", "templates/numbers/3.jpeg",
                             "templates/numbers/4.jpeg", "templates/numbers/5.jpeg", "templates/numbers/6.jpeg",
                             "templates/numbers/7.jpeg", "templates/numbers/8.jpeg", "templates/numbers/9.jpeg",
                             "templates/numbers/10.jpeg", "templates/numbers/J.jpeg", "templates/numbers/Q.jpeg",
                             "templates/numbers/K.jpeg"]
    suit_template_paths = ["templates/suits/club.jpeg", "templates/suits/diamond.jpeg", "templates/suits/heart.jpeg", "templates/suits/spade.jpeg"] 
    
    number_templates, number_names = load_templates(number_template_paths)
    suit_templates, suit_names = load_templates(suit_template_paths)

    # Apply template matching to number ROI
    best_number, number_score, matched_number_template = template_matching(number_roi, number_templates, number_names)
    
    # Apply template matching to suit ROI
    best_suit, suit_score, matched_suit_template = template_matching(suit_roi, suit_templates, suit_names)

    # Display results
    print(f"Best matched number: {best_number} with score: {number_score}")
    print(f"Best matched suit: {best_suit} with score: {suit_score}")

    plot_images(img, cropped_img, number_roi, suit_roi)

    if matched_number_template is not None:
        plt.figure(figsize=(5, 5))
        plt.title(f"Matched Number Template: {best_number}")
        plt.imshow(matched_number_template, cmap='gray')
        plt.show()

    if matched_suit_template is not None:
        plt.figure(figsize=(5, 5))
        plt.title(f"Matched Suit Template: {best_suit}")
        plt.imshow(matched_suit_template, cmap='gray')
        plt.show()


if __name__ == "__main__":
    main()
