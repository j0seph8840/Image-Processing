import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os

def main():
    name = input("Enter the name of the image: ")
    image_path = f"{name}.tif"
    image = cv2.imread(image_path)

    align_image(image)

def align_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Average blur kernel 3x3
    blur_kernel = np.ones((5, 5)) / 25
    
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
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # Compute the gradient direction in radians (-π to π)
    radians = np.arctan2(grad_y, grad_x)

    # Convert gradient direction to degrees (-180 to 180)
    degrees = np.degrees(radians)

    # Display original image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title('Original Image')
    
    # Display magnitude of the gradient
    plt.figure(figsize=(12, 8))
    plt.imshow(magnitude, cmap='gray')
    plt.title('Magnitude of the Gradient')
    
    # Plot the histogram of gradient directions
    plt.figure(figsize=(10, 6))
    plt.hist(degrees.ravel(), bins=360, range=(-180, 180), alpha=1)
    plt.title('Histogram of Gradient Directions')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()