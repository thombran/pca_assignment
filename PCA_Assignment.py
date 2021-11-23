import imageio
from matplotlib.colors import Colormap
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import os
import sys
"""
@Authors: Brandon Thomas & Daniel Floyd
@Date: 11/22/2021
CIS 365 - Artificial Intelligence
Professor Denton Bobeldyk
Assignment 12 - Principal Component Analysis
"""

VECTORS_TO_KEEP = [15, 100, 200] # Amt of vectors the program will run through

# Make sure image is provided and exists in file system
def load_image():
    if len(sys.argv) != 2:
        print('Please supply an image file\nUsage: python penny.py example.jpg')
        exit(1)

    image = sys.argv[1]

    if not os.path.exists(image):
        print('Image file is invalid')
        exit(1)
    return imageio.imread(image)

# Calc total amount of pixels and divide sum of total values
def find_mean(image):
    sum = 0
    total_pixels = image.shape[0] * image.shape[1]
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            sum += image[row][col]
    return sum / total_pixels

# Subtract the value of mean from each pixel in image (value cannot be negative)
def subtract_mean(mean, image):
    new_img = [[pixel for pixel in range(1080)] for pixel in range(1080)]
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            new_img[row][col] = image[row][col] - mean
    return new_img

def get_covariance(image):
    return np.cov(image)

# Returns eigenvectors and eigenvalues as a tuple, where vals[i] corresponds to vectors[:, i]
def find_eig_vals_vect(img_cov):
    vals, vectors = lg.eig(img_cov)
    return vals, vectors

# Calculate the variance among the selected eigenvectors 
def calc_perc_variance(eig_vals, amt_vectors):
    eig_sum = 0
    total = 0
    for num in range(amt_vectors): # Sum up eigenvalues in selected range
        eig_sum += eig_vals[num]
    for value in eig_vals: # Sum up all eigenvalues
        total += value
    return eig_sum / total


# Driver function
def run():
    fig = plt.figure(figsize=(10,10)) # Figure where images will be shown
    ROWS = 2 # Rows and cols for matplotlib figure at end of program completion
    COLS = 2
    pos = 1
    for amt_vectors in VECTORS_TO_KEEP: # Loop through vectors and create lossy image for each
        img = load_image()
        m = find_mean(img)
        img_sub_mean = subtract_mean(m, img)
        cov = get_covariance(img_sub_mean)

        eig_vals, eigVects = find_eig_vals_vect(cov) # eig_vals[i] corresponds to eigVects[:, i]
        percent_variance = calc_perc_variance(eig_vals, amt_vectors) * 100
        print(str(percent_variance) + '% variance for ' + str(amt_vectors) + ' vectors')

        eigenVectorsToKeep = np.transpose(eigVects[0:amt_vectors]) # Format vectors to correct dimensions and return
        compressedImage = np.matmul(img_sub_mean, eigenVectorsToKeep)
        lossyUnCompressedImage = np.matmul(compressedImage, np.transpose(eigenVectorsToKeep)) + m

        fig.add_subplot(ROWS, COLS, pos)
        pos += 1

        plt.imshow(lossyUnCompressedImage, cmap='gray')
        plt.axis('off')
        plt.title('Vectors kept: ' + str(amt_vectors))
    plt.show()

run()