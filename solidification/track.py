import os

import imageio
import numpy as np
from skimage import (
    img_as_float, exposure, filters, measure, morphology, restoration
)


def load_am_img(
    img_n, 
    data_path='data/nickel_solidification.tif', 
    crop_tup=(175, 60, 250, 110)
):
    """Load an image corresponding to the image number image_n in dataset 
     located in directory data_dir.

    Parameters
    ----------
    img_n : int
        Image number for image in dataset data_dir that will be loaded.
    data_path : str, optional
        Directory in which image datset is located. Defaults to 'data', where 
        example subset of images is located.
    crop_tup : (4, ) array-like, optional
        Array-like containing integers for cropping the image corresponding to 
        (x1, y1, x2, y2). Defaults to (175, 60, 250, 110).

    Returns
    -------
    img : array 
        Single image loaded from the stack of images located at data_path.
    """
    
    collection = []
    reader = imageio.get_reader(data_path)
    for img_slice in reader:
        collection.append(img_slice)
    img = collection[img_n]
    # Convert image to float
    img = img_as_float(img)
    
    if crop_tup is not None:
        # Define top left corner (x1, y1) & bottom right corner (x2, y2) of
        # image crop from crop_tup
        x1, y1, x2, y2 = crop_tup
        # Crop image
        img = img[y1:y2, x1:x2]
    
    return img

def sub_img(img, img_to_sub_n, img_load_func=load_am_img):
    """Subtract an image from another image.

    Parameters
    ----------
    img : array
        Input image.
    img_to_sub_n : int 
        Image number corresponding to the image to subtract from `img`.
    img_load_func : function, optional
        Function used to load the image that will be subtracted. Defaults to 
        load_am_img.

    Returns
    -------
    img : array
        Array representing the subtracted image.
    """
    
    # Smooth image
    img = filters.gaussian(img)

    # Load image that will be subtracted
    img_pre = img_load_func(img_to_sub_n)
    img_pre = filters.gaussian(img_pre)
    
    img = img - img_pre

    return img

def clip(img, proc_func=None, low=5, high=95):
    """Clip the top and bottom intensities from an image and replace the 
    intensities below percentile `low` with the intensity value at percentile 
    `low` and replace the intensities above percentile `high` with the 
    intensity at percentile `high`.

    Parameters
    ----------
    img : array
        Input image.
    proc_func : function, optional
        Processing function taking an input image and returning a processed 
        image. Defaults to None.
    low : int, optional
        Lower intensity threshold in percentile. Defaults to 5.
    high : int, optional
        Upper intensity threshold in percentile. Defaults to 95.

    Returns
    -------
    img : array
        Clipped image.
    """
    
    if proc_func is not None:
        # Preprocess image
        img = proc_func(img)
    
    # Clip top and bottom image intensities 
    # (assign low/high for all values below/above low/high)
    p_low, p_high = np.percentile(img, [low, high])
    img -= p_low
    img[(img < 0.0)] = 0.0
    img = img / p_high
    img[(img > 1.0)] = 1.0
    
    return img

def invert_denoise(img, denoise_weight=0.15):
    """Invert and denoise an image.

    Parameters
    ----------
    img : array
        Input image.
    denoise_weight : float, optional
        Weight variable to be passed to denoising algorithm. Details avaible at 
        https://scikit-image.org/docs/stable/api/skimage.restoration
        .html#skimage.restoration.denoise_tv_chambolle. Defaults to 0.15.

    Returns
    -------
    img : array
        Inverted and denoised image.
    """
    
    # Invert image
    img = 1.0 - img

    # Denoise image
    img = restoration.denoise_tv_chambolle(img, weight=denoise_weight)
    
    return img

def threshold(img, thresh_val=0.4, thresh_greater=True):
    """Threshold and create binary image.

    Parameters
    ----------
    img : array
        Input image.
    thresh_val : float, optional
        Thresholding value. Defaults to 0.4.
    thresh_greater : bool, optional
        If True, set values above thresholding value to 1. If False, use values 
        below thresholding value. Defaults to True.

    Returns
    -------
    img : array
        Binary thresholded image.
    """
    
    # Binarize image
    img_mask = np.zeros(img.shape, dtype=np.int)
    if thresh_greater:
        img_mask[(img > thresh_val)] = 1
    else:
        img_mask[(img < thresh_val)] = 1

    return img_mask
    
def filter_region_size(mask, min_region_size=100, return_bbox=False):
    """Filter the regions in a binary image/mask so that only the regions with 
    a pixel area larger than min_region_size will remain.

    Parameters
    ----------
    mask : array
        Array representing a binary image/mask from which to extract larger 
        regions.
    min_region_size : int, optional
        Minimum area in pixels for which the regions will be excluded if they 
        are below this number. Defaults to 100.
    return_bbox : bool, optional
        If true, the tuple representing the bounding box will be returned 
        instead of an array with the areas smaller than min_region_size 
        excluded. Defaults to False.

    Returns
    -------
    If return_bbox is False:
    img : array
        A binary image/mask representing the regions of img that are larger 
        than min_region_size.
    If return_bbox is True:
    bbox : tuple
        A 4-tuple corresponding to the bounding box (min_row, min_col, max_row, 
        max_col) of the conglomeration of regions larger than min_region_size.
    """

    labels = morphology.label(mask, connectivity=2)
    regions = measure.regionprops(labels)

    # Make lists to hold all the bounding box (bbox) values in case multiple 
    # regions are greate than min_region_size
    minr_list, minc_list, maxr_list, maxc_list = [], [], [], []
    filtered_mask = np.zeros(labels.shape, dtype=np.int)
    for region in regions:
        if region.area > min_region_size:
            filtered_mask[(labels == region.label)] = 1
            minr_list.append(region.bbox[0]) 
            minc_list.append(region.bbox[1]) 
            maxr_list.append(region.bbox[2]) 
            maxc_list.append(region.bbox[3]) 

    minr = min(minr_list)
    minc = min(minc_list)
    maxr = max(maxr_list)
    maxc = max(maxc_list)

    if return_bbox:
        return minr, minc, maxr, maxc

    return filtered_mask
