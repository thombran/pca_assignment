import imageio
from matplotlib.colors import Colormap
import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import os
import sys

VECTORS_TO_KEEP = [15, 100, 200]

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

# To work according to assignment code posted, needed to swap matrix axis
# i.e. cols became rows and rows became cols since (n, k) * (k, m) --> (n, m)
def format_vectors(vectors):
    final_matrix = [[pixel for pixel in range(len(vectors))] for pixel in range(len(vectors[0]))] # temp matrix
    row_pos = 0
    col_pos = 0
    for vector in vectors:
        for val in vector:
            final_matrix[col_pos][row_pos] = val #Place val in [row][col] pos into [col][row] pos
            col_pos += 1
        col_pos = 0
        row_pos += 1
    return final_matrix

# Driver function
def run():
    img = load_image()
    m = find_mean(img)
    img_sub_mean = subtract_mean(m, img)
    cov = get_covariance(img_sub_mean)
    eig_vals, eigVects = find_eig_vals_vect(cov) # eig_vals[i] corresponds to eigVects[:, i]
    percent_variance = calc_perc_variance(eig_vals, VECTORS_TO_KEEP[0]) * 100
    print(str(percent_variance) + '% variance')
    eigenVectorsToKeep = format_vectors(eigVects[0: VECTORS_TO_KEEP[0]]) # Format vectors to correct dimensions and return
    compressedImage = np.matmul(img_sub_mean, eigenVectorsToKeep)
    lossyUnCompressedImage = np.matmul(compressedImage, np.transpose(eigenVectorsToKeep)) + m
    result = plt.imshow(compressedImage, cmap='gray')
    plt.show()


run()