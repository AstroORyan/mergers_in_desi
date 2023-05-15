'''
Author: David O'Ryan
Date: 09/05/2023

Script to calculate the gini coefficients of all the mergers in my sample. This must be done on the HEC as there are approximately 200k of them.
'''

## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from zipfile import ZipFile
tqdm.pandas()
import os

from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from shapely.geometry import Polygon, Point
import cv2 as cv

import sys
sys.path.insert(0, '/mmfs1/storage/users/oryan/torch-packages')

## Functions
def getting_correct_contours(contours):
    length = 0
        
    for i in contours:
        if len(i) > length:
            correct_contours = i
            length = len(i)
    
    return correct_contours

def conts_to_list(contours):
    contour_list = []
    for i in range(len(contours)):
        row = contours[i][0]
        contour_list.append([row[0], row[1]])
    return contour_list

def get_galaxy(cutout):
    cutout_int = cutout.data.copy()
    
    cut = np.percentile(cutout.data,90)
    cutout_int[cutout_int <= cut] = 0
    cutout_int[cutout_int > cut] = 1
    cutout_int = cutout_int.astype(int)
    
    contours, _ = cv.findContours(cutout_int, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_NONE)
    
    contours_nested_list = getting_correct_contours(contours)
    
    extracted_contour_list = conts_to_list(contours_nested_list)
    
    contour_arr = np.zeros([len(extracted_contour_list),2])
    for i in range(len(extracted_contour_list)):
        contour_arr[i,0] = extracted_contour_list[i][0]
        contour_arr[i,1] = extracted_contour_list[i][1]
        
    pl = Polygon(contour_arr)
    
    pixels_mask = np.zeros(cutout.data.shape).astype(bool)
    for i in range(cutout.data.shape[0]):
        for j in range(cutout.data.shape[1]):
            pt = Point(i,j)
            if pl.contains(pt):
                pixels_mask[i,j] = True
    pixels_mask = pixels_mask.T
    
    reduced_cutout = cutout.data[pixels_mask]
    
    return reduced_cutout

def calc_gini_func(pixels):
    mean_flux = np.mean(abs(pixels))
    ordered_pixels = np.sort(pixels)
    n = len(ordered_pixels)
    
    gini = (((2 * np.arange(1, n + 1)) - n) - 1)*np.abs(ordered_pixels)
        
    normalization =  (mean_flux * n * (n - 1))
    
    return np.sum(gini) / normalization

def calc_gini(path, petro_50, ra, dec): 
    if np.isnan(petro_50):
        return 'Failed'
    
    try:
        data = fits.getdata(path)
    except:
        return 'corrupted'
    
    try:
        header = fits.getheader(path)
    except:
        return 'corrupted'
    
    w = WCS(header, naxis = 2)
    
    size = u.Quantity((10*petro_50, 10*petro_50), u.arcsec)
    coord = SkyCoord(ra = ra * u.deg, dec = dec * u.deg, frame = 'icrs')
    try:
        cutout = Cutout2D(data[1,:,:], coord, size, wcs = w, mode='strict')
    except:
        return 'partial-overlap'
    
    if np.sum(cutout.data) == 0:
        return 'empty-image'
    
    reduced_cutout = get_galaxy(cutout)
    
    gini = calc_gini_func(reduced_cutout)
    
    return gini
    
## Main Function
def main():
    print('Reading in CSV...')
    df = pd.read_csv('/mmfs1/home/users/oryan/galaxy-zoo-desi/data/mergers-hec-manifest.csv', index_col = 0)
    print('Successful.')

    print('Creating HEC Names...')
    df_hec = (
        df
        .assign(hec_path = df.id_str.apply(lambda x: f'/mmfs1/scratch/hpc/60/oryan/desi-mergers/{x}-cutout.fits'))
    )
    print('Successful.')

    assert os.path.exists(df_hec.hec_path.iloc[0])

    print('Beginning Gini Calculations...')
    df_gini = (
        df_hec.
        assign(gini_r = df_hec.apply(lambda row: calc_gini(row.hec_path, row.est_petro_th50, row.ra, row.dec), axis = 1))
    )
    print('Completed.')

    print('Saving...')
    df_gini.to_csv('/mmfs1/home/users/oryan/galaxy-zoo-desi/results/ginis-mergers.csv')
    print('Successfully saved.')

    print('Algorithm Completed.')

## Initialization
if __name__ == '__main__':

    zfile_path = '/mmfs1/scratch/hpc/60/oryan/mergers.zip'
    if os.path.exists(zfile_path):
        print('Extracting mergers...')
        with ZipFile(zfile_path, 'r') as zfile:
            zfile.extractall('/mmfs1/scratch/hpc/60/oryan/desi-mergers/')
        print('Completed.')
        os.remove(zfile_path)

    main()