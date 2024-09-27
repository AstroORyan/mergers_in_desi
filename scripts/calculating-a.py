'''
Author: David O'Ryan
Date: 07/06/2023

This script will calculate all three CAS parameters, starting with finding the Asymmetry paramter. From this parameter, we also gain the image centre. This centre is used to calculate the remaining parameters.
'''
## Imports
import logging
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2 as cv
from shapely.geometry import Polygon, Point
from PIL import Image

from astropy.io import fits

def getting_correct_contours(contours, cen_x, cen_y):
    point = Point(cen_x, cen_y)        
    for i in contours:
        cont_arr = conts_to_arr(i)
        if len(cont_arr) > 2:
            polygon = Polygon(cont_arr)
            if polygon.contains(point):
                return cont_arr
            else:
                continue
        else:
            continue
        
    return 'failed'

def conts_to_arr(nested_list):
    contour_arr = np.zeros([len(nested_list),2])
    for i in range(len(nested_list)):
        row = nested_list[i][0]
        contour_arr[i,0] = row[0]
        contour_arr[i,1] = row[1]
    
    return contour_arr

## Functions
def calc_a(im_path: str) -> list:
    ## Loading Data
    data = fits.getdata(im_path)[1,:,:]

    if np.sum(data) == 0.0:
        return 'empty-image'
    
    ## Finding Correct Contour
    cutout_int = data.copy()
    cut = np.percentile(data,80)
    cutout_int[cutout_int <= cut] = 0
    cutout_int[cutout_int > cut] = 1
    cutout_int = cutout_int.astype(int)
    
    try:
        contours, _ = cv.findContours(cutout_int, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_NONE)
    except:
        return 'failed'
    contour_arr = getting_correct_contours(contours, int(data.shape[0]/2), int(data.shape[1]/2))

    del contours, cutout_int

    if contour_arr == "failed":
        return 'no-galaxy'
    
    ## Defining Non Galaxy Pixels
    pl = Polygon(contour_arr)

    del contour_arr

    pixels_mask = np.zeros(data.shape).astype(bool)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pt = Point(i,j)
            if pl.contains(pt):
                pixels_mask[i,j] = True
    pixels_mask = pixels_mask.T
    
    gal_im = data.copy()
    gal_im[~pixels_mask] = 0
    
    gal_pixels_arr = np.asarray(np.argwhere(pixels_mask).tolist())
    
    min_A = np.inf
    centre = np.array([0,0])

    for i in gal_pixels_arr:
        im = Image.fromarray(gal_im)
        rot_im = im.rotate(180, center = (i[0], i[1]))
        gal_rot = np.asarray(rot_im)
        im.close()
        rot_im.close()

        del rot_im, im

        gal_rot[~pixels_mask] = 0

        A = np.sum(abs(gal_im[pixels_mask] - gal_rot[pixels_mask])) / np.sum(abs(gal_im[pixels_mask]))

        if A < min_A:
            min_A = A
            centre = i

        del gal_rot
            
    return [min_A, centre]

## Main Function
def main():
    gal_type = 'major'
    df = pd.read_csv(f'/mmfs1/home/users/oryan/galaxy-zoo-desi/data/{gal_type}-hec-manifest.csv', index_col = 0)

    df_paths = (
        df
        .assign(hec_paths = df.id_str.apply(lambda x: f'/mmfs1/scratch/hpc/60/oryan/desi-{gal_type}/{x}-cutout.fits'))
    )

    a_dict = {i.replace('-cutouts.fits', '') : calc_a(i) for i in tqdm(list(df_paths.hec_paths))}

    df_a = pd.DataFrame.from_dict(a_dict, orient = 'index').reset_index().rename(columns = {'index' : 'id_str', 0 : 'asym', 1 : 'centre'})

    df_a.to_csv('/mmfs1/home/users/oryan/galaxy-zoo-desi/results/asym-df.csv')

## Initialization
if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s %(levelname)s: %(message)s',
                        filename = '/mmfs1/home/users/oryan/galaxy-zoo-desi/cas-log.log',
                        filemode = 'a')

    main()