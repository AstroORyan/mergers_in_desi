'''
Author: David O'Ryan
Date: 02/05/2023

This script will be used to parallelise getting FITS images across 16 areas on the HEC. It will make network calls to the Legacy Survey and download them.
'''

## Imports
import pandas as pd
from joblib import Parallel, delayed
import time
import requests
import os
import sys
from tqdm import tqdm
tqdm.pandas()

import glob
import zipfile

## Functions
def get_fits(row, category = 'mergers'):
    ra = row[0]
    dec = row[1]
    id_str = row[2]
    
    save_dir = f'/mmfs1/scratch/hpc/60/oryan/{category}/{id_str}-cutout.fits'
    if os.path.exists(save_dir):
        return save_dir
    
    url = f'http://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&layer=ls-dr10&pixscale=0.262'
    
    for i in range(5):
        try:
            r = requests.get(url)
        except:
            time.sleep(1)
            continue
        
        if r.status_code == 200:
            break
        else:
            time.sleep(1)
    
    if i >= 4:
        sys.exit()
    
    with open(save_dir, 'wb') as f:
        f.write(r.content)
    
    return save_dir

## Main Function
def main():
    print('Import data...')
    df = pd.read_csv('/mmfs1/home/users/oryan/galaxy-zoo-desi/data/mergers-for-hec.csv', index_col = 0)
    print('Completed.')

    print('Beginning file downloads...')
    # results = Parallel(n_jobs = 8)(delayed(get_fits)(i) for i in tqdm(zip(df['ra'], df['dec'], df['id_str'])))
    df_export = (
        df
        .assign(hec_paths = df.progress_apply(lambda row: get_fits([row.ra, row.dec, row.id_str], 'mergers'), axis = 1))
        )
    print('Completed.')

    # df_export = df.assign(hec_paths = results)

    print('Saving new csv...')
    df_export.to_csv('/mmfs1/home/users/oryan/galaxy-zoo-desi/data/mergers-dowloaded.csv')
    print('Completed.')

    print(df_export.hec_paths.value_counts())

    files = glob.glob('/mmfs1/scratch/hpc/60/oryan/mergers/*.fits')

    print(f'Number of files downloaded: {len(files)}')

    print('Adding FITS images to a ZipFile.')

    with zipfile.ZipFile('/mmfs1/scratch/hpc/60/oryan/mergers/mergers-fits.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel = 6) as zipf:
        for file in tqdm(files):
            zipf.write(file, os.path.basename(file))

    print('Completed.')

    print('Removing excess Files.')
    for i in tqdm(files):
        if os.path.exists(i):
            os.remove(i)
        else:
            continue
    print('Completed.')

    print('Algorithm Completed.')

## Initialization
if __name__ == '__main__':
    main()