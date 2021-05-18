# Standard library imports
import os

# Third-party imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from skimage import measure, morphology


def plot_am_imgs(
    img_n_list,
    img_load_func,
    row_title=None,
    show_region_bbox=False,
    cmap='viridis', 
    show_axis=False, 
    tight_layout=True, 
    figsize=(10, 6)
):
    """Plot a series of cropped images.
    
    Parameters
    ----------
    img_dir_path : str
        Path to directory containing raw images.
    img_load_func : function, optional
        Function used to load the image that will be subtracted. Defaults to 
        load_am_img.
    row_title : str, optional
        Y-axis label to give far left image to differentiate this set of 
        plotted images from any other rows of images that may also be plotted 
        from another call to this function.
    cmap : str, optional
        Colormap to use on plotted images. Must be a matplotlib colormap. 
        Defaults to 'viridis'.
    show_axis : bool, optional
        If False, hides the ticks of the x and y axes. Defaults to False.
    tight_layout : bool, optional
        If True, uses matplotlib.pyplot.Figure.tight_layout() to reduce 
        whitespace in figure. Defaults to True.
    figsize : (2, ) tuple
        Sets max figure size (in inches) for matplotlib figure. Defaults to 
        (10, 6).
    
    Returns
    -------
    fig, axes : matplotlib.pyplot.Figure, array
        Matplotlib figure and axes containing 
        plotted images.
    """
    fig, axes = plt.subplots(ncols=len(img_n_list), figsize=figsize)
    ax = axes.ravel()

    for i, img_n in enumerate(img_n_list):
        
        img = img_load_func(img_n)

        ax[i].imshow(img, cmap=cmap, interpolation='nearest')
        ax[i].set_title(f'Image {img_n}')
        
        if row_title is not None:
            ax[0].set_ylabel(row_title)
        
        # Unless show_axis is True, remove the axis from the image in a way 
        # that retains the y label
        if not show_axis:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        
        # When show_region_bbox is true, plot bbox of regions determined by 
        # skimage.morphology.label
        if show_region_bbox:
            # Label connected areas in masked image
            labels = morphology.label(img, connectivity=2)
            # Turn labels into regions
            regions = measure.regionprops(labels)

            mask = np.zeros(labels.shape, dtype=np.int)
            for region in regions:
                if (region.area > 100):
                    mask[(labels == region.label)] = 1
                    minr, minc, maxr, maxc = region.bbox
                    rect = mpatches.Rectangle(
                        (minc, minr), maxc - minc, maxr - minr, 
                        fill=False, edgecolor='red', linewidth=2
                    )
                    ax[i].add_patch(rect)
        
        # Cut down on white space in figure in matplotlib's automatic way
        if tight_layout:
            fig.tight_layout()
    
    return fig, axes
    
def plot_bbox(
    img_n_list,
    img_load_func,
    region_gen_func,
    row_title=None,
    cmap='viridis', 
    show_axis=False,
    tight_layout=True, 
    figsize=(10, 6)
):
    """Generate a bounding box (bbox) from a processing routine and plot the 
    resulting bbox on a loaded image.
    
    Parameters
    ----------
    img_n_list : array-like
        A list or array of image numbers for which bounding boxes of regions of 
        interest will be plotted.
    img_load_func : function
        An image returning function that takes an image number as an argument 
        to return an image on which the bounding box of the determined region 
        of interest will be plotted.
    region_gen_func : function
        A bounding box returning function that take an image number as an 
        argument, loads the corresponding image, and performs a series of 
        processing functions on that image to generate a region of interest of 
        which a bounding box will be returned.
    row_title : str, optional
        A title to be applied on the y-axis of the left-most image in case 
        multiple iterations of plot_bbox() want to be called e.g. to compare 
        different region_gen_funcs or img_load_funcs passed for a given set of 
        image numbers.
    cmap : str, optional
        Colormap to use on plotted images. Must be a matplotlib colormap. 
        Defaults to 'viridis'.
    show_axis : bool, optional
        If False, hides the ticks of the x and y axes. Defaults to False.
    tight_layout : bool, optional
        If True, uses matplotlib.pyplot.Figure.tight_layout() to reduce 
        whitespace in figure. Defaults to True.
    figsize : (2, ) array-like, optional
        Sets max figure size (in inches) for matplotlib figure. Defaults to 
        (10, 6).
    
    Returns
    -------
    fig, axes : matplotlib.pyplot.Figure, array
        Matplotlib figure and axes containing plotted images.
    """

    fig, axes = plt.subplots(ncols=len(img_n_list), figsize=figsize)
    ax = axes.ravel()

    for i, img_n in enumerate(img_n_list):
        
        img = img_load_func(img_n)

        bbox_minr, bbox_minc, bbox_maxr, bbox_maxc = region_gen_func(img_n)

        ax[i].imshow(img, cmap=cmap, interpolation='nearest')
        ax[i].set_title(f'Image {img_n}')
        
        if row_title is not None:
            ax[0].set_ylabel(row_title)
        
        # Unless show_axis is True, remove the axis from the image in a way 
        # that retains the y label
        if not show_axis:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        
        # Label connected areas in masked image
        labels = morphology.label(img, connectivity=2)
        # Turn labels into regions
        regions = measure.regionprops(labels)

        # Plot bbox as rectangle on image
        rect = mpatches.Rectangle(
            (bbox_minc, bbox_minr), 
            bbox_maxc - bbox_minc, 
            bbox_maxr - bbox_minr, 
            fill=False, edgecolor='red', linewidth=2
        )
        ax[i].add_patch(rect)
        
        # Cut down on white space in figure in matplotlib's automatic way
        if tight_layout:
            fig.tight_layout()

    return fig, axes
