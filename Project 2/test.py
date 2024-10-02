import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    return rotated

# Function to crop the image to the bounding box of the edges
def crop_image_to_edges(image, edge_mask):
    # Find the coordinates of the non-zero values in the edge mask
    coords = np.column_stack(np.where(edge_mask > 0))

    # Get the bounding box of the edges
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    # Crop the image based on the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image

# Main function to load, process, rotate, and crop the image
def main():
    # Load the image in grayscale
    img = cv2.imread('Testimage1.tif', cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to the grayscale image
    gaussian_blur = np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]]) / 273

    img_blurred = cv2.filter2D(img, -1, gaussian_blur)

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
    print(f"Dominant edge angle: {dominant_angle} degrees")

    # Since gradient_angle corresponds to edge orientation, calculate rotation angle directly
    rotation_angle = 90 - dominant_angle
    print(f"Rotation angle: {rotation_angle} degrees")

    if rotation_angle < 0:
        rotation_angle += 180

    # Rotate the image
    rotated_img = rotate_image(img, rotation_angle)

    # Display the rotated
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(rotated_img, cmap='gray')
    plt.title('Rotated Image')

    # Sobel filters for the rotated image
    sobel_x = cv2.filter2D(rotated_img, cv2.CV_32F, gx)
    sobel_y = cv2.filter2D(rotated_img, cv2.CV_32F, gy)

    # Rotated image magnitude and angle
    rotated_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Display the rotated image magnitude
    plt.subplot(132)
    plt.imshow(rotated_mag, cmap='gray')
    plt.title('Rotated Image Magnitude')

    # Crop the image
    cropped_img = crop_image_to_edges(rotated_img, rotated_mag)

    # Display the cropped image
    plt.subplot(133)
    plt.imshow(cropped_img, cmap='gray')
    plt.title('Cropped Image')
    plt.show()


if __name__ == "__main__":
    main()
