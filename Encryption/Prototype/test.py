import OpenEXR
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calculate_entropy(data):
    hist, _ = np.histogram(data, bins=256, range=(0, 1))
    hist = hist[hist > 0]  # Ignore zero entries
    probabilities = hist / hist.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def analyze_exr_file(filename, reference=None):
    file = OpenEXR.InputFile(filename)
    header = file.header()
    dw = header['dataWindow']
    width = dw.max.x + 1
    height = dw.max.y + 1

    channels = []
    for channel_name in header['channels']:
        channel_data = np.frombuffer(file.channel(channel_name), dtype=np.float32).reshape((height, width))
        channels.append(channel_data)

    for i, channel in enumerate(channels):
        print(f"Channel {i + 1} Analysis:")
        mean = np.mean(channel)
        std_dev = np.std(channel)
        min_val = np.min(channel)
        max_val = np.max(channel)
        variance = std_dev ** 2
        entropy = calculate_entropy(channel.flatten())

        print(f"  - Mean: {mean}")
        print(f"  - Standard Deviation: {std_dev}")
        print(f"  - Variance: {variance}")
        print(f"  - Min: {min_val}")
        print(f"  - Max: {max_val}")
        print(f"  - Entropy: {entropy}")

        plt.hist(channel.flatten(), bins=256, range=(0, 1))
        plt.title(f"Histogram of Channel {i + 1}")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

        # If a reference image is provided, calculate PSNR and SSIM
        if reference is not None:
            # Assuming reference has the same dimensions and number of channels
            psnr = 20 * np.log10(np.max(reference) / np.sqrt(np.mean((reference - channel) ** 2)))
            ssim_index = ssim(reference, channel, data_range=channel.max() - channel.min())
            print(f"  - PSNR: {psnr:.2f} dB")
            print(f"  - SSIM: {ssim_index:.4f}")

        # Correlation Coefficient
        if len(channels) > 1:
            corr_matrix = np.corrcoef([channel.flatten() for channel in channels])
            print(f"  - Correlation Matrix:\n{corr_matrix}")

    file.close()

# Example usage:
filename = r"/Encryption/Prototype\encrypted1.exr"
# Optionally, load a reference image for PSNR and SSIM calculations
# reference_filename = r"path_to_reference_image.exr"
# reference = analyze_exr_file(reference_filename)
analyze_exr_file(filename)
#
#
# # import numpy as np
# # import OpenEXR
# # import Imath
# #
# #
# # def load_exr_image(image_path):
# #     exr_file = OpenEXR.InputFile(image_path)
# #     header = exr_file.header()
# #     dw = header['dataWindow']
# #     width, height = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
# #
# #     # Read RGB channels if available, or fallback to Y channel
# #     if all(c in header["channels"] for c in ("R", "G", "B")):
# #         channels = ["R", "G", "B"]
# #         image = []
# #         for channel in channels:
# #             channel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
# #             channel_array = np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)
# #             image.append(channel_array)
# #         image = np.stack(image, axis=-1)  # Combine channels into one 3D array
# #     elif "Y" in header["channels"]:
# #         channel_data = exr_file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
# #         image = np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)
# #     else:
# #         raise ValueError("EXR file must contain either RGB or Y channel.")
# #
# #     return image
# #
# #
# # def calculate_psnr(image1, image2):
# #     if image1.shape != image2.shape:
# #         raise ValueError("Images must have the same dimensions")
# #
# #     mse = np.mean((image1 - image2) ** 2)
# #     if mse == 0:
# #         return float('inf')
# #
# #     max_pixel_value = 1.0
# #     psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
# #     return psnr
# # def calculate_npcr(image1, image2):
# #     if image1.shape != image2.shape:
# #         raise ValueError("Images must have the same dimensions")
# #
# #     # Calculate the number of differing pixels
# #     differing_pixels = np.sum(image1 != image2)
# #
# #     # Calculate NPCR as the percentage of differing pixels
# #     total_pixels = image1.size
# #     npcr = (differing_pixels / total_pixels) * 100
# #     return npcr
# # # Paths to the two .exr images
# # image_path1 = 'encrypted1.exr'
# # image_path2 = 'encrypted2.exr'
# #
# # # Load images
# # image1 = load_exr_image(image_path1)
# # image2 = load_exr_image(image_path2)
# #
# # # Calculate PSNR
# # psnr_value = calculate_psnr(image1, image2)
# # npcr_value = calculate_npcr(image1, image2)
# # print(f"PSNR: {psnr_value:.2f} dB")
# # print(f"NPCR: {npcr_value:.2f}%")

# import numpy as np
# import OpenEXR
# import Imath
# import cv2
#
# def load_exr_image(image_path):
#     exr_file = OpenEXR.InputFile(image_path)
#     header = exr_file.header()
#     dw = header['dataWindow']
#     width, height = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#
#     # Read RGB channels if available, or fallback to Y channel
#     if all(c in header["channels"] for c in ("R", "G", "B")):
#         channels = ["R", "G", "B"]
#         image = []
#         for channel in channels:
#             channel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
#             channel_array = np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)
#             image.append(channel_array)
#         image = np.stack(image, axis=-1)  # Combine channels into one 3D array
#     elif "Y" in header["channels"]:
#         channel_data = exr_file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT))
#         image = np.frombuffer(channel_data, dtype=np.float32).reshape(height, width)
#     else:
#         raise ValueError("EXR file must contain either RGB or Y channel.")
#
#     return image
#
# def calculate_entropy(image):
#     histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
#     histogram = histogram[histogram > 0]  # Ignore zero entries
#     probabilities = histogram / histogram.sum()
#     entropy = -np.sum(probabilities * np.log2(probabilities))
#     return entropy
#
# def calculate_correlation(image):
#     if image.ndim == 3:
#         channels = [image[:, :, i] for i in range(image.shape[2])]
#         correlation_matrix = np.corrcoef([c.flatten() for c in channels])
#         return correlation_matrix
#     return None
#
# def calculate_standard_deviation(image):
#     return np.std(image)
#
# def assess_encryption(image_path):
#     image = load_exr_image(image_path)
#
#     # Calculate entropy
#     entropy = calculate_entropy(image)
#     print(f"Entropy: {entropy}")
#
#     # Calculate correlation
#     correlation_matrix = calculate_correlation(image)
#     print("Correlation Matrix:")
#     print(correlation_matrix)
#
#     # Calculate standard deviation
#     std_dev = calculate_standard_deviation(image)
#     print(f"Standard Deviation: {std_dev}")
#
#     # Criteria for a good encryption score
#     entropy_threshold = 7.5
#     correlation_threshold = 0.1  # values should be low for strong encryption
#     std_dev_threshold = 0.5  # adjust based on the data range
#
#     # Check conditions
#     score = 0
#     if entropy >= entropy_threshold:
#         score += 30  # Good entropy score
#     if np.all(np.abs(correlation_matrix) < correlation_threshold):
#         score += 30  # Low correlation score
#     if std_dev >= std_dev_threshold:
#         score += 30  # Good standard deviation score
#
#     # Additional score for randomness
#     patterns, _ = extract_patterns(image)
#     if np.sum(patterns) > 0:  # If some patterns are found
#         score += 10
#
#     # Final score
#     print(f"Encryption Score: {score}/100")
#     return score
#
# def extract_patterns(image, threshold=0.5):
#     # Convert to grayscale if the image is multi-channel
#     if image.ndim == 3:
#         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray_image = image
#
#     # Apply thresholding to highlight patterns
#     _, thresh_image = cv2.threshold(gray_image, threshold, 1.0, cv2.THRESH_BINARY)
#
#     # Use edge detection (Canny) to find edges
#     edges = cv2.Canny((thresh_image * 255).astype(np.uint8), 100, 200)
#
#     # Find contours to identify patterns
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Draw contours on the original image for visualization
#     pattern_image = np.zeros_like(image)
#     cv2.drawContours(pattern_image, contours, -1, (1, 0, 0), 1)  # Red contours
#
#     return pattern_image, contours
#
# # Example usage
# image_path = r'C:\Users\viraj\PycharmProjects\Research\Encryption\Prototype\encrypted2.exr'  # Replace with your .exr file path
# assess_encryption(image_path)


import cv2
import numpy as np
import OpenEXR
import Imath


def read_exr(file_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)

    # Get the header to retrieve the size of the image
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the RGB channels
    channels = exr_file.channels(['R', 'G', 'B'])
    # Convert channels to a NumPy array
    img = np.zeros((size[1], size[0], 3), dtype=np.float32)
    img[..., 0] = np.frombuffer(channels[0], np.float32).reshape(size[1], size[0])  # Red
    img[..., 1] = np.frombuffer(channels[1], np.float32).reshape(size[1], size[0])  # Green
    img[..., 2] = np.frombuffer(channels[2], np.float32).reshape(size[1], size[0])  # Blue

    return img


def calculate_mse(imageA, imageB):
    # Calculate the Mean Squared Error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def calculate_psnr(imageA, imageB):
    # Calculate the Peak Signal-to-Noise Ratio between two images
    mse = calculate_mse(imageA, imageB)
    if mse == 0:
        return 100  # Perfect match
    max_pixel = 1.0  # Since we're dealing with floating point images [0,1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def rate_encryption(original_image_path, encrypted_image_path):
    # Read the original PNG image
    original = cv2.imread(original_image_path)

    # Read the encrypted EXR image using OpenEXR
    encrypted = read_exr(encrypted_image_path)

    # Check if the original image is in the range [0, 255]
    if original.max() > 255:
        original = (original / 255.0).astype(np.float32)  # Normalize if it's in float format

    # Ensure both images are the same size
    if original.shape[0:2] != encrypted.shape[0:2]:
        raise ValueError("Images must be of the same dimensions for comparison.")

    # Calculate PSNR
    psnr_value = calculate_psnr(original, encrypted)

    # Rate the encryption based on PSNR
    if psnr_value >= 40:
        rating = 100
    elif psnr_value >= 30:
        rating = 75
    elif psnr_value >= 20:
        rating = 50
    else:
        rating = 25

    return psnr_value, rating


original_image_path = r'/Encryption/Prototype\Compilation\Imagess\wtff.png'  # Update with your image path
encrypted_image_path = r'/Encryption/Prototype\wtf.exr'  # Update with your image path
psnr_value, encryption_rating = rate_encryption(original_image_path, encrypted_image_path)

print(f"PSNR Value: {psnr_value:.2f} dB")
print(f"Encryption Rating: {encryption_rating}/100")
