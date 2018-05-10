####################################################################################################
# Steerable-Pyramid Code for the SCO
# Code by Chrysa Papadaniil <chrysa@nyu.edu>

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
    """Equivalent of Matlab freqspace, frequency spacing for frequency response"""
    f1 = []                   
    if dim % 2 == 0:   
        for i in range(-dim, dim-1, 2):
            ft = float(i) / float(dim)
            f1.append(ft)    
    else:                  
        for i in range(-dim+1, dim, 2):
            ft = float(i) / float(dim)
            f1.append(ft) 
    return f1

def make_steer_fr(dims, orientation, cpd, bandwidth, num_orientations):
    """Makes the frequency response of a steerable pyramid filter for a particular orientation and 
       central frequency
    
    Arguments:
    dims -- size of each image
    orientation --  a single real number, the orientation of the filter
    cpd -- spatial frequency of the filter, in cycles per degree of visual angle
    bandwidth -- spatial frequency bandwidth in octaves

    Returns 2-D array that contains the frequency response
    """
    
    p = num_orientations -1
    const = math.sqrt(float(math.pow(2,(2*p))*math.pow(math.factorial(p),2))/float(math.factorial(2*p)*(p+1)))
    f1 = freqspace(dims[0])
    f2 = freqspace(dims[1])
    wx, wy = np.meshgrid(f1, f2)
    r = np.sqrt(wx**2 + wy**2)
    theta = np.arctan2(wy, wx) 
   
    freq_resp = const*1j*np.cos(theta - orientation)**p * log_raised_cos(r, cpd, bandwidth)
    return freq_resp

def build_steer_band(im, freq_resp, orientation, cpd):
    """Builds subband of steerable pyramid transform of one multiscale for a particular orientation and 
    central frequency
    
    Arguments:
    im -- an array of grayscale images
    freq_resp -- filter frequency response returned by make_steer_fr
    orientation --  a single real number, the orientation of the filter
    cpd -- spatial frequency of the filter, in cycles per degree of visual angle
    
    Returns 3D array that contains a steerable pyramid subband for the input images
    """
    
    fourier = np.fft.fftshift(np.fft.fft2(im))
    subband = np.fft.ifft2(np.fft.fftshift(np.multiply(fourier, freq_resp))).real
    return subband

def spyr_filter(im, orientation, cpd, bandwidth, num_orientations):
    """ Makes energy contrast images based on quadrature pairs of a steerable pyramid filter of a 
        particular spatial frequency and orientation
    
    Arguments:
    im -- a (p x m x n) array of grayscale images, where p is the number of images and m, n are the 
    images dimensions
    orientation --  a single real number, the orientation of the filter
    cpd -- spatial frequency of the filter, in cycles per degree of visual angle
    bandwidth -- spatial frequency bandwidth in octaves
    
    Returns a (p x m x n) array of energy contrast images, the result of filtering the array of images with a 
    quadrature pair of steerable filters with a particular central frequency and orientation. The result is 
    the squared sum of the quadrature subbands.
    """
    dims = im.shape    
    freq_resp_imag = make_steer_fr((dims[1], dims[2]), orientation, cpd, bandwidth, num_orientations)
    freq_resp_real = abs(make_steer_fr((dims[1], dims[2]), orientation, cpd, bandwidth, num_orientations))

    pyr_imag = build_steer_band(im, freq_resp_imag, orientation, cpd)
    pyr_real = build_steer_band(im, freq_resp_real, orientation, cpd)
    contrast_pyr = (pyr_real**2 + pyr_imag**2)  
    return contrast_pyr
