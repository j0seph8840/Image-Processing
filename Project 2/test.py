import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image in grayscale
img = cv2.imread('Testimage3.tif', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to the grayscale image
gaussian_blur = np.array([[1, 4, 7, 4, 1],
                          [4, 16, 26, 16, 4],
                          [7, 26, 41, 26, 7],
                          [4, 16, 26, 16, 4],
                          [1, 4, 7, 4, 1]]) / 273
img_blurred = cv2.filter2D(img, -1, gaussian_blur)

# Sobel kernels (transposed)
gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]).T
gy = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]).T

# Apply Sobel filters to the blurred image
sobel_x = cv2.filter2D(img_blurred, cv2.CV_32F, gx)
sobel_y = cv2.filter2D(img_blurred, cv2.CV_32F, gy)

# Calculate gradient magnitude and angle
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_angle = np.arctan2(sobel_y, sobel_x)

# Convert radians to degrees and adjust angle range to [0, 180)
gradient_angle = np.degrees(gradient_angle) % 180

# Threshold the gradient magnitude to keep significant edges
mag_threshold = 50  # Adjust this value as needed
significant_mask = gradient_magnitude > mag_threshold

# Get significant gradient angles
significant_angles = gradient_angle[significant_mask]

# Compute histogram of significant angles
hist_bins = 180  # One bin per degree
hist_range = (0, 180)
hist, bins = np.histogram(significant_angles, bins=hist_bins, range=hist_range)

# Find the dominant angle
dominant_angle_index = np.argmax(hist)
dominant_angle = (bins[dominant_angle_index] + bins[dominant_angle_index + 1]) / 2
print(f"Dominant angle: {dominant_angle} degrees")

# Since gradient_angle corresponds to edge orientation, calculate rotation angle directly
rotation_angle = 90 - dominant_angle
print(f"Initial rotation angle: {rotation_angle} degrees")

# Adjust rotation angle to be within [-90, 90] for minimal rotation
if rotation_angle < -90:
    rotation_angle += 180
elif rotation_angle > 90:
    rotation_angle -= 180
print(f"Adjusted rotation angle: {rotation_angle} degrees")

# Load the original image in color (or grayscale if preferred)
img_original = cv2.imread('Testimage1.tif')

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

# Rotate the image
rotated_img = rotate_image(img_original, rotation_angle)

# Display the rotated image
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')
plt.axis('off')
plt.show()
