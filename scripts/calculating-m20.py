'''
Author: David O'Ryan
Date: 26/04/2023

Script to calculate the M20 coefficients from Galaxy Zoo DESI.
'''
# imports
import glob
import os
from tqdm import tqdm
tqdm.pandas()
from zipfile import ZipFile

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
import astropy.units as u

from shapely.geometry import Polygon, Point
import cv2 as cv


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
    
    cut = np.percentile(cutout.data,65)
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

def calc_m20(row):
    
    path = row.hec_path
    petro_50 = row.est_petro_th50
    ra = row.ra
    dec = row.dec
    
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

    # mtot = np.inf
    # for i in range(cutout.shape[0]):
    #     for j in range(cutout.shape[1]):

    #         if not pixels_mask[i,j]:
    #             continue

    #         m_tmp = 0

    #         for p in range(cutout.shape[0]):
    #             for q in range(cutout.shape[1]):

    #                 if not pixels_mask[p,q]:
    #                     continue

    #                 m_tmp += cutout.data[p,q] * ((p - i)**2 + (q - j)**2)

    #         if m_tmp < mtot:
    #             mtot = m_tmp.copy()
    #             center = [i,j]

    gal_pixels_list = np.argwhere(pixels_mask).tolist()
    gal_pixels_arr = np.asarray(gal_pixels_list)

    mtot = np.inf
    for i in gal_pixels_list:
        
        mtmp = np.sum(cutout.data[gal_pixels_arr[:,0], gal_pixels_arr[:,1]] * ((gal_pixels_arr[:,0] - i[0])**2 + (gal_pixels_arr[:,1] - i[1])**2))
        
        if mtmp < mtot:
            mtot = mtmp.copy()
            center = i.copy()

    f_tot = np.sum(cutout.data[pixels_mask])
    
    sum_f = 0
    cutout_array = cutout.data.copy()
    pixels = []

    sum_f = 0
    cutout_array = cutout.data.copy()
    cutout_array[np.invert(pixels_mask)] = 0
    pixels = []

    while sum_f < 0.20 * f_tot:
        arr_max = np.max(cutout_array)
        indices = np.where(cutout_array == arr_max)
        x = indices[0][0]
        y = indices[1][0]

        pixels.append([x,y])

        sum_f += arr_max
        cutout_array[x,y] = 0
        
    m_i = []
    for i in pixels:
        x = i[0]
        y = i[1]
        f = cutout.data[x,y]

        m_i.append(f * ((x - center[0])**2 + (y - center[1])**2))
        
    m_20 = np.log10(np.sum(m_i) / mtot)
    
    return m_20

## Main Function
def main():
    print('Reading in CSV...')
    df_ginis = pd.read_csv('/mmfs1/home/users/oryan/galaxy-zoo-desi/data/mergers-hec-manifest.csv', index_col = 0)
    print('Successful.')

    print('Creating HEC Names...')
    df_hec = (
        df_ginis
        .assign(hec_path = df_ginis.id_str.apply(lambda x: f'/mmfs1/scratch/hpc/60/oryan/desi-mergers/{x}-cutout.fits'))
    )
    print('Successful.')

    assert os.path.exists(df_hec.hec_path.iloc[0])

    print('Beginning M20 Calculation...')
    df_m20 = (
        df_hec
        .assign(m20 = df_hec.apply(lambda row: calc_m20(row), axis = 1))
    )
    print('Successful.')

    print('Saving...')
    df_m20.to_csv('/mmfs1/home/users/oryan/galaxy-zoo-desi/results/m20s-merger.csv')
    print('Successfully saved.')

    print('Algorithm Completed.')

## initialization
if __name__ == '__main__':

    zfile_path = '/mmfs1/scratch/hpc/60/oryan/mergers.zip'
    if os.path.exists(zfile_path):
        print('Extracting mergers...')
        with ZipFile(zfile_path, 'r') as zfile:
            zfile.extractall('/mmfs1/scratch/hpc/60/oryan/desi-mergers/')
        print('Completed.')
        os.remove(zfile_path)

    main()