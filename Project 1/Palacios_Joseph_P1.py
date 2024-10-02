import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np


def main():
    # Load images

    path = input("Enter the path to the image set: ")

    directory = glob.glob(f"{path}/*.JPG")

    images = [cv2.imread(file) for file in directory]

    hsv_method(images)

def rgb_method(images):
    for image in images:

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Split the channels
        r, g, b = cv2.split(rgb)
        # If red, green, and blue all have the same value then we got a gray pixel
        gray_pixels = np.sum((r == g) & (g == b))

        total_pixels = image.shape[0] * image.shape[1]

# Determine whether day or night and display image

        # Check if there are more gray pixels than anything else
        is_night = gray_pixels > total_pixels * .06495

        if is_night:
            title = 'Night!'
        else:
            title = 'Day!'  

        plt.figure(figsize=(12,8))

        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Plotting histogram. Value x Frequency

        # # Create histograms for each channel
        # plt.figure(figsize=(12,8))

        # # RGB
        # # Plot histogram for Red
        # plt.subplot(3, 1, 1)
        # plt.hist(r.ravel(), bins=256, range=[0, 255], color='red', alpha=0.7)
        # plt.title('Red Histogram')
        # plt.xlabel('Red value')
        # plt.ylabel('Frequency')
        # # Plot histogram for Green
        # plt.subplot(3, 1, 2)
        # plt.hist(g.ravel(), bins=256, range=[0, 255], color='green', alpha=0.7)
        # plt.title('Green Histogram')
        # plt.xlabel('Green value')
        # plt.ylabel('Frequency')
        # # Plot histogram for Blue
        # plt.subplot(3, 1, 3)
        # plt.hist(b.ravel(), bins=256, range=[0, 255], color='blue', alpha=0.7)
        # plt.title('Blue Histogram')
        # plt.xlabel('Blue value)')
        # plt.ylabel('Frequency')

        # # Adjust the layout and display
        # plt.tight_layout()
        # plt.show()


def hsv_method(images):
    for image in images:

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Split the channels
        h, s, v = cv2.split(hsv)
        # If hue and saturation are both 0 then we got a gray pixel
        gray_pixels = np.sum((h == 0) & (s == 0))
        total_pixels = image.shape[0] * hsv.shape[1]

# Determine whether day or night and display image

        # Check if there are more gray pixels than anything else
        is_night = gray_pixels > total_pixels * .06495

        if is_night:
            title = 'Night!'
        else:
            title = 'Day!'  

        plt.figure(figsize=(12,8))

        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Plotting histogram. Value x Frequency

        # # Create histograms for each channel
        # plt.figure(figsize=(12,8))

        # # HSV
        # # Plot histogram for Hue
        # plt.subplot(3, 1, 1)
        # plt.hist(h.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
        # plt.title('Hue Histogram')
        # plt.xlabel('Hue value')
        # plt.ylabel('Frequency')
        # # Plot histogram for Saturation
        # plt.subplot(3, 1, 2)
        # plt.hist(s.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
        # plt.title('Saturation Histogram')
        # plt.xlabel('Saturation value')
        # plt.ylabel('Frequency')
        # # Plot histogram for Value
        # plt.subplot(3, 1, 3)
        # plt.hist(v.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        # plt.title('Value Histogram')
        # plt.xlabel('Value (Brightness)')
        # plt.ylabel('Frequency')

    # # Adjust the layout and display
    #     plt.tight_layout()
    #     plt.show()

main()