# Standard library imports
import os

# Third-party imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from skimage import measure, morphology


def plot_imgs(
    img_list,
    row_title=None,
    cmap='viridis', 
    show_axis=False, 
    tight_layout=True, 
    figsize=(10, 6)
):
    """Plot a series of cropped images.
    
    Parameters
    ----------
    img_list : str
        A list of images to be plotted.
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
    fig, axes = plt.subplots(ncols=len(img_list), figsize=figsize)
    ax = axes.ravel()

    for i, img in enumerate(img_list):
        
        ax[i].imshow(img, cmap=cmap, interpolation='nearest')
        
        if row_title is not None:
            ax[0].set_ylabel(row_title)
        
        # Unless show_axis is True, remove the axis from the image in a way 
        # that retains the y label
        if not show_axis:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        
        # Cut down on white space in figure in matplotlib's automatic way
        if tight_layout:
            fig.tight_layout()
    
    return fig, axes
    
def plot_bbox(
    img_list,
    bbox_list,
    row_title=None,
    cmap='viridis', 
    show_axis=False,
    tight_layout=True, 
    figsize=(10, 6)
):
    """Plot bounding boxes on a loaded image.
    
    Parameters
    ----------
    img_list : list
        A list of images to be plotted.
    bbox_list : list
        A list of 4-tuples (minr, minc, maxr, maxc) to be plotted on top of the 
        images in img_list.
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
    fig, axes = plt.subplots(ncols=len(img_list), figsize=figsize)
    ax = axes.ravel()

    for i, (img, bbox) in enumerate(zip(img_list, bbox_list)):
        
        ax[i].imshow(img, cmap=cmap, interpolation='nearest')
        
        if row_title is not None:
            ax[0].set_ylabel(row_title)
        
        # Unless show_axis is True, remove the axis from the image in a way 
        # that retains the y label
        if not show_axis:
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        minr, minc, maxr, maxc = bbox
        # Plot bbox as rectangle on image
        rect = mpatches.Rectangle(
            (minc, minr), 
            maxc - minc, 
            maxr - minr, 
            fill=False, edgecolor='red', linewidth=2
        )
        ax[i].add_patch(rect)
        
        # Cut down on white space in figure in matplotlib's automatic way
        if tight_layout:
            fig.tight_layout()

    return fig, axes
