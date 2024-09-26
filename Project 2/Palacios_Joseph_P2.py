import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

def main():
    # name = input("Enter the name of the image: ")
    # image_path = f"{name}.tif"
    image = cv2.imread("Testimage3.tif")

    align_image(image)

def align_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Average blur kernel 3x3
    # blur_kernel = np.ones((5, 5)) / 25

    #gausian blue 5x5 kernel
    blur_kernel = np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]]) / 256
    
    # Apply blurring
    blurred = cv2.filter2D(binary_image, cv2.CV_32F, blur_kernel)

    # Apply Sobel filters
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Apply convolution with the Sobel X and Y filters
    grad_x = cv2.filter2D(blurred, cv2.CV_32F, sobel_x)
    grad_y = cv2.filter2D(blurred, cv2.CV_32F, sobel_y)

    # Compute the magnitude of the gradient
    # magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.abs(grad_x) + np.abs(grad_y)

    # Compute the angle of the gradient
    angle_radians = np.arctan2(grad_y, grad_x)

    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Create histogram of angles, ignore 0-degree pixels
    hist, bins = np.histogram(angle_radians, bins=180, range=(1, 180))
    
    # Find the dominant angle, finds the degree with the highest frequency
    dominant_angle = bins[np.argmax(hist)]
    print(f"Dominant angle is: {dominant_angle}")

    # Rotate the image based on the dominant angle
    angle_to_rotate = 90 - dominant_angle
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle_to_rotate, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Display original image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title('Original Image')

    # Display the binary image
    plt.figure(figsize=(12, 8))
    plt.imshow(binary_image)
    plt.title('Binary Image')

    # Display the blurred image
    plt.figure(figsize=(12, 8))
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    
    # Display magnitude of the gradient
    plt.figure(figsize=(12, 8))
    plt.imshow(magnitude, cmap='gray')
    plt.title('Magnitude of the Gradient')

    # Display angle of the gradient
    plt.figure(figsize=(12, 8))
    plt.imshow(angle_degrees, cmap='gray')
    plt.title('Angle of the Gradient')

    # Display rotated image
    plt.figure(figsize=(12, 8))
    plt.imshow(rotated_image)
    plt.title('Rotated Image')

    # Display the histogram of the angle
    plt.figure(figsize=(12, 8))
    plt.plot(hist)
    plt.title('Histogram of Angles')

    plt.show()

if __name__ == "__main__":
    main()