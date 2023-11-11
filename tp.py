# Code for removing noise

# import cv2

# def preprocess_image(image_path, target_size):
#     # Load the image
#     image = cv2.imread("img.jpeg")

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Remove noise using Gaussian blur
#     denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

#     # Adjust contrast using histogram equalization
#     equalized_image = cv2.equalizeHist(denoised_image)

#     # Resize the image to the target size
#     resized_image = cv2.resize(equalized_image, target_size)

#     return resized_image

# # Example usage:
# image_path = 'img.jpeg'
# target_size = (800, 600)  # Specify the desired size

# preprocessed_image = preprocess_image(image_path, target_size)

# # Display the preprocessed image
# cv2.imshow("Preprocessed Image", preprocessed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Code for better clearity
import cv2
import numpy as np

def enhance_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply non-local means denoising
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Apply histogram equalization for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(denoised_image)

    return enhanced_image

# Example usage:
image_path = 'img.jpeg'

# Enhance the image
enhanced_image = enhance_image(image_path)

# Display the enhanced image
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()