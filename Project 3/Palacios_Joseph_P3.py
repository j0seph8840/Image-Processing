import numpy as np
import matplotlib.pyplot as plt
import cv2

# Ask user for image without the file extension
# name = input("Enter the name of the image: ")
# image_path = f"{name}.tif"

# Load the image in grayscale
img = cv2.imread("Proj3.tif", cv2.IMREAD_GRAYSCALE)

# Convert the image to float32 to avoid clipping issues
img_float = img.astype(np.float32)

# Apply Gaussian blur to estimate the background illumination
blurred = cv2.GaussianBlur(img_float, (51, 51), 0)

# Subtract the blurred image from the original image
illumination_corrected = img_float - blurred

# Normalize the result to the range [0, 255]
illumination_corrected = cv2.normalize(illumination_corrected, None, 0, 255, cv2.NORM_MINMAX)

# Convert back to uint8
illumination_corrected = illumination_corrected.astype(np.uint8)

# Find the Fourier Transform of the image
illumination_corrected_fourier = np.log(1 + np.abs(np.fft.fftshift(np.fft.fft2(illumination_corrected))))

# Create a Butterworth low-pass filter
def butterworth_lowpass_filter(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    return H

# Apply the Butterworth low-pass filter
cutoff = 30  # Cutoff frequency
order = 2    # Order of the filter
butterworth_filter = butterworth_lowpass_filter(illumination_corrected.shape, cutoff, order)
filtered_fourier = illumination_corrected_fourier * butterworth_filter

# Perform the inverse Fourier transform to get the filtered image
filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_fourier))
filtered_image = np.abs(filtered_image)

# Normalize the filtered image to the range [0, 255]
filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)
filtered_image = filtered_image.astype(np.uint8)

# Display the original and corrected images
plt.figure(figsize=(12, 6))
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.figure(figsize=(12, 6))
plt.title('Illumination Corrected Image')
plt.imshow(illumination_corrected, cmap='gray')

plt.figure(figsize=(12, 6))
plt.title('Spectrum F(u)')
plt.imshow(illumination_corrected_fourier, cmap='gray')

plt.figure(figsize=(12, 6))
plt.title('Filtered Spectrum H(u)F(u)')
plt.imshow(filtered_fourier, cmap='gray')

plt.show()