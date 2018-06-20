####################################################################################################
# Steerable-Pyramid Code for the SCO
# Code by Chrysa Papadaniil <chrysa@nyu.edu> and Noah C. Benson <nben@nyu.edu>

import scipy
import os
import math
import numpy as np

pi=np.pi


def log0(x, base=None, zero_val=None):
    x = np.asarray(x) 
    y = np.full(x.shape, 0.0 if zero_val is None else zero_val, dtype=np.float)
    ii = (x > 0)
    y[ii] = np.log(x[ii]) / (1 if base is None else np.log(base))
    return y

def log_raised_cos(r, cpd, bandwidth):
    rarg = (pi / bandwidth) * log0(pi / cpd * r, 2, bandwidth)
    y = np.sqrt(0.5 * (np.cos(rarg) + 1))  
    y[np.where(rarg >= pi)] =0
    y[np.where(rarg <= -pi)] = 0  
    return y

def freqspace(dim):    
    'Equivalent of Matlab freqspace, frequency spacing for frequency response.'
    return (np.linspace(-dim,     dim - 2, dim) / float(dim)
            if dim % 2 == 0 else
            np.linspace(-dim + 1, dim - 1, dim) / float(dim))

def make_steer_fr(dims, orientation, rpp, bandwidth, num_orientations):
    '''
    Makes the frequency response of a steerable pyramid filter for a particular orientation and 
    central frequency
    
    Arguments:
    dims -- size of each image
    orientation --  a single real number, the orientation of the filter
    rpp -- spatial frequency of the filter, in radians (of cycle) per pixel angle
    bandwidth -- spatial frequency bandwidth in octaves

    Returns 2-D array that contains the frequency response
    '''
    
    p = num_orientations - 1
    const = math.sqrt((2.0**(2*p) * math.factorial(p)**2) / float(math.factorial(2*p) * (p+1)))
    f1 = freqspace(dims[0])
    f2 = freqspace(dims[1])
    wx, wy = np.meshgrid(f1, f2)
    r = np.sqrt(wx**2 + wy**2)
    theta = np.arctan2(wy, wx) 

    freq_resp = -const*1j * np.cos(theta - orientation)**p * log_raised_cos(r, rpp, bandwidth)

    return np.fft.ifftshift(freq_resp)

def build_steer_band(im, freq_resp):
    '''
    Builds subband of steerable pyramid transform of one multiscale for a particular orientation and 
    central frequency
    
    Arguments:
    im -- an array of grayscale images
    freq_resp -- filter frequency response returned by make_steer_fr
    
    Returns 3D array that contains a steerable pyramid subband for the input images
    '''
    fourier = np.fft.fft2(im)
    subband = np.fft.ifft2(np.multiply(fourier, freq_resp)).real
    return subband

def spyr_filter(im, orientation, cpp, bandwidth, num_orientations):
    '''
    Makes energy contrast images based on quadrature pairs of a steerable pyramid filter of a 
    particular spatial frequency and orientation
    
    Arguments:
    im -- a (p x m x n) array of grayscale images, where p is the number of images and m, n are the 
    images dimensions
    orientation --  a single real number, the orientation of the filter
    cpp -- spatial frequency of the filter, in cycles per image pixel
    bandwidth -- spatial frequency bandwidth in octaves
    
    Returns a (p x m x n) array of energy contrast images, the result of filtering the array of images with a 
    quadrature pair of steerable filters with a particular central frequency and orientation. The result is 
    the squared sum of the quadrature subbands.
    '''
    # the frequency expected in the functions below is in radians / pixel
    rpp = cpp * 2*pi
    dims = im.shape    
    freq_resp_imag = make_steer_fr((dims[1], dims[2]), orientation, rpp, bandwidth, num_orientations)
    #freq_resp_real = abs(make_steer_fr((dims[1], dims[2]), orientation, rpp, bandwidth, num_orientations))
    freq_resp_real = np.abs(freq_resp_imag)

    pyr_imag = build_steer_band(im, freq_resp_imag)
    pyr_real = build_steer_band(im, freq_resp_real)
    #contrast_pyr = (pyr_real**2 + pyr_imag**2)  
    return pyr_real + 1j*pyr_imag
