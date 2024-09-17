import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.stats import chisquare
import zlib
from scipy.stats import entropy

def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    length = len(data)
    freq_map = Counter(data)
    for count in freq_map.values():
        p_x = count / length
        entropy += - p_x * math.log2(p_x)
    return entropy

def plot_histogram(data, title="Byte Frequency Distribution", filename="histogram.png"):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(256), alpha=0.75, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Byte value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(filename)  # Save the histogram to a file
    plt.close()  # Close the plot to free up memory

def calculate_correlation_coefficient(data1, data2):
        # Ensure both data sets are the same length
        if len(data1) != len(data2):
            raise ValueError("Data sets must have the same number of elements")

        # Calculate the correlation coefficient
        correlation_matrix = np.corrcoef(data1, data2)
        correlation_coefficient = correlation_matrix[0, 1]
        return correlation_coefficient

def chi_squared_test(data):
    observed_freq = np.bincount(data, minlength=256)
    expected_freq = np.full(256, len(data) / 256)
    chi_stat, p_value = chisquare(observed_freq, f_exp=expected_freq)
    return chi_stat, p_value

def calculate_hamming_distance(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("Data sets must have the same length to calculate Hamming distance")
    return sum(b1 != b2 for b1, b2 in zip(data1, data2))

def calculate_compression_ratio(data):
    compressed_data = zlib.compress(data)
    return len(data) / len(compressed_data)

def calculate_frequency_distribution(data):
    distribution = Counter(data)
    return distribution

def calculate_kl_divergence(p, q):
    return sum(p[i] * math.log(p[i] / q[i]) for i in range(len(p)))

