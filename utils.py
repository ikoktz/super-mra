import sys
import cv2
import numpy as np
import scipy.ndimage as ndi
import os
from PIL import Image
import tensorflow as tf
from scipy.signal import convolve2d
import time
import multiprocessing
from joblib import Parallel, delayed


def cv2_clipped_zoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    :param img: image array to zoom and clip
    :param zoom_factor: zoom factor
    :return: zoomed image
    """
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='edge')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def print_prog_bar(icur, imax, bar_width):
    """
    prints a progress bar
    :param icur: current index in progress bar
    :param imax: max index length of progress bar
    :param bar_width: width of progress bar
    :return: none
    """
    frac = icur / imax
    filled_progbar = round(frac * bar_width)
    print('\r', '#' * filled_progbar + '-' * (bar_width - filled_progbar), '[{:>7.2%}]'.format(frac), end='')


def augment_patches(patchvolume, augmentfactor, seed, rotrange, shiftrange, minsigscale):
    """
    function that augments patches through random shifting, scaling, and rotations
    :param patchvolume: array that contains patches; has size [# patches,patch nrows, patch ncols,1]
    :param augmentfactor: number of times to augment data
    :param seed: seed for random data augmentation
    :param rotrange: data augmentation random rotation angle range (+/-) in degrees
    :param shiftrange: data augmentation random shift range (+/-) in pixels
    :param minsigscale: minimum value to randomly scale patches during augmentation
    :return: augmented patchvolume array of size [#patches * augmentfactor,patch nrows, patch ncols,1]
    """
    np.random.seed(seed)
    # create augmented output variable
    output = np.zeros(
        [patchvolume.shape[0] * augmentfactor, patchvolume.shape[1], patchvolume.shape[2], patchvolume.shape[3]],
        dtype='float16')
    tmp = np.zeros([patchvolume.shape[1], patchvolume.shape[2]], dtype='float32')
    # loop over patches to augment
    print("augmenting data " + str(augmentfactor) + "-fold ")
    for iPatch in range(0, patchvolume.shape[0]):
        if iPatch % 3 == 0:
            print_prog_bar(iPatch, patchvolume.shape[0], 40)
        patch = np.squeeze(patchvolume[iPatch, :, :, 0])
        # loop over augmentation steps
        for iAug in range(0, augmentfactor):
            # copy original patch to output variable
            if iAug == 0:
                output[iPatch * augmentfactor + iAug, :, :, 0] = patch
            # perform data augmentation using random rotations and translations
            else:
                randangle = np.random.uniform(-rotrange, rotrange)
                randshftx = np.random.uniform(-shiftrange, shiftrange)
                randshfty = np.random.uniform(-shiftrange, shiftrange)
                randzoom = np.random.uniform(0.8, 1.2)
                randsiscale = np.random.uniform(minsigscale, 1.0)

                tmp = randsiscale * np.float32(patch)
                # print('ang',randangle,'randshiftx',randshftx,'randshifty',randshfty,'randzoom',randzoom)
                tmp = ndi.interpolation.shift(tmp, (randshftx, randshfty), output=None, order=3, mode='nearest',
                                              cval=0.0, prefilter=True)
                tmp = cv2_clipped_zoom(tmp, randzoom)

                tmpimg = np.float16(
                    ndi.interpolation.rotate(tmp, randangle, axes=(1, 0), reshape=False, output=None, order=3,
                                             mode='nearest', cval=0.0, prefilter=True))
                output[iPatch * augmentfactor + iAug, :, :, 0] = tmpimg
    print()
    return output


def augment_patches_2p5d(patchvolume, augmentfactor, seed, rotrange, shiftrange, minsigscale):
    """
    function that augments patches through random shifting, scaling, and rotations
    :param patchvolume: array that contains patches; has size [# patches,patch nrows, patch ncols, input_ch]
    :param augmentfactor: number of times to augment data
    :param seed: seed for random data augmentation
    :param rotrange: data augmentation random rotation angle range (+/-) in degrees
    :param shiftrange: data augmentation random shift range (+/-) in pixels (tuple or integer)
    :param minsigscale: minimum value to randomly scale patches during augmentation
    :param input_ch: number of channels in input data for each patch
    :return: augmented patchvolume array of size [#patches * augmentfactor,patch nrows, patch ncols, input_ch]
    """
    if not isinstance(shiftrange, tuple):  # convert to tuple
        shiftrange = (shiftrange, shiftrange)

    np.random.seed(seed)
    # create augmented output variable
    output = np.zeros(
        [patchvolume.shape[0] * augmentfactor, patchvolume.shape[1], patchvolume.shape[2], patchvolume.shape[3]],
        dtype='float16')
    tmp = np.zeros([patchvolume.shape[1], patchvolume.shape[2], patchvolume.shape[3]], dtype='float32')
    # loop over patches to augment
    print("augmenting data " + str(augmentfactor) + "-fold ")
    for iPatch in range(0, patchvolume.shape[0]):
        if iPatch % 3 == 0:
            print_prog_bar(iPatch, patchvolume.shape[0], 40)
        patch = np.squeeze(patchvolume[iPatch, :, :, :])
        # loop over augmentation steps
        for iAug in range(0, augmentfactor):
            # copy original patch to output variable
            if iAug == 0:
                output[iPatch * augmentfactor + iAug, :, :, :] = patch
            # perform data augmentation using random rotations and translations
            else:
                randangle = np.random.uniform(-rotrange, rotrange)
                randshftx = np.random.uniform(-shiftrange[0], shiftrange[0])
                randshfty = np.random.uniform(-shiftrange[1], shiftrange[1])
                randzoom = np.random.uniform(0.8, 1.2)
                randsiscale = np.random.uniform(minsigscale, 1.0)

                tmp = randsiscale * np.float32(patch)
                # print('ang',randangle,'randshiftx',randshftx,'randshifty',randshfty,'randzoom',randzoom)
                tmp = ndi.interpolation.shift(tmp, (randshftx, randshfty, 0), output=None, order=3, mode='nearest',
                                              cval=0.0, prefilter=True)
                for chs in range(patchvolume.shape[3]):
                    tmp[:, :, chs] = cv2_clipped_zoom(tmp[:, :, chs], randzoom)

                tmpimg = np.float16(
                    ndi.interpolation.rotate(tmp, randangle, axes=(1, 0), reshape=False, output=None, order=3,
                                             mode='nearest', cval=0.0, prefilter=True))
                output[iPatch * augmentfactor + iAug, :, :, :] = tmpimg
    print()
    return output


# removes values below zero and above one
def remove_values_above_one_below_zero(arr):
    """
    removes values >1 and <0 from numpy array
    :param arr: numpy array
    :return: numpy array with values >1 and <0 set to 1 and 0 respectively
    """
    arr[np.where(arr > 1)] = 1
    arr[np.where(arr < 0)] = 0
    return arr


# helper function that loads a tiff file to a numpy array
def load_tiff_to_numpy_vol(path, submode, minslc, maxslc):
    """
    function that loads a tiff file to a numpy array
    :param path:  file name of tiff stack
    :param submode: subset mode flag (True/False)
    :param minslc: minimum slice for subset mode
    :param maxslc: maximum slice for subset mode
    :return: tiff stack loaded in numpy array
    """
    # open and load tiff
    img = Image.open(path)
    img.load()
    # extract individual slices
    if submode:  # loading only a portion of the tiff (i.e. a subset of slices)
        volume = np.zeros([img.height, img.width, min(maxslc, img.n_frames) - (minslc - 1)], dtype='float16')
        for n in range(minslc, min(maxslc, img.n_frames)):
            img.seek(n)
            volume[:, :, n - minslc] = np.float32(np.asarray(img.copy()))
    else:  # loading the entire tiff (i.e. all slices)
        volume = np.zeros([img.height, img.width, img.n_frames], dtype='float16')
        for n in range(0, img.n_frames):
            img.seek(n)
            volume[:, :, n] = np.float32(np.asarray(img.copy()))
    # return volume to process further
    return volume


# helper function that counts the number of slices in a given tiff file
def count_tiff_slices(path, submode, minslc, maxslc, raw_projection=0):
    """
    function that counts the number of slices in a given tiff file
    :param path: file name of tiff stack
    :param submode: subset mode flag
    :param minslc: minimum slice for subset mode
    :param maxslc: maximum slice for subset mode
    :param raw_projection: projection direction to read tiff along
    :return: number of slice in stack
    """
    img = Image.open(path)  # open tiff
    img.load()  # load tiff
    # print('img shape: ', img.size)
    if submode:  # extract a subset of slices
        nslices = min(maxslc, img.n_frames) - (minslc - 1)
        if raw_projection > 0:
            nslices = min(maxslc, img.size[raw_projection - 1]) - (minslc - 1)
    else:  # extract all slice
        nslices = img.n_frames
        if raw_projection > 0:
            nslices = img.size[raw_projection - 1]
    return nslices


# helper function that returns source and target tiff file names
def get_source_and_target_files(dirsource, dirtarget):
    """
    # helper function that returns tiff files to process
    inputs:
    dirsource  folder containing input tiff files
    dirtarget  folder containing output tiff files

    outputs:   srcfiles, tgtfiles
    """
    srcfiles = []
    for fname in os.listdir(dirsource):
        if fname.endswith('.tif') > 0 or fname.endswith('.tiff'):
            srcfiles.append(fname)
    tgtfiles = []
    for fname in os.listdir(dirtarget):
        if fname.endswith('.tif') > 0 or fname.endswith('.tiff'):
            tgtfiles.append(fname)
    return srcfiles, tgtfiles


# helper function that loads tiff and scales signals between 0 and 1
def load_tiff_volume_and_scale_si(in_dir, in_fname, crop_x, crop_y, blksz, raw_projection, subset_train_mode,
                                  subset_train_minslc, subset_train_maxslc):
    """
    :param in_dir: input directory
    :param in_fname: input filename
    :param crop_x: crop factor in x direction
    :param crop_y: crop factor in y direction
    :param blksz: patch size (i.e. the length or height of patch, either integer or tuple)
    :param raw_projection: projection direction
    :param subset_train_mode: boolean subset mode flag
    :param subset_train_minslc: if loading a subset, min slice index to load
    :param subset_train_maxslc: if loading a subset, max slice index to load
    :return:    vol (tiff volume scaled between 0 and 1), maxsi (maximum signal intensity of original volume)
    """
    t = time.time()
    if not isinstance(blksz, tuple):  # create tuple if not
        blksz = (blksz, blksz)
    vol = load_tiff_to_numpy_vol(os.path.join(in_dir, in_fname), subset_train_mode, subset_train_minslc,
                                 subset_train_maxslc)
    elapsed = time.time() - t
    print('total time to load source tiff was', elapsed, 'seconds')
    # adjust x and y dimensions of volume to divide evenly into blksz
    # while cropping using 'crop_train' to increase speed and avoid non-pertinent regions
    t = time.time()
    vol = crop_volume_in_xy_and_reproject(vol, crop_x, crop_y, blksz, raw_projection)
    vol = np.float32(vol)
    if len(np.argwhere(np.isinf(vol))) > 0:
        for xyz in np.argwhere(np.isinf(vol)):
            vol[xyz[0], xyz[1], xyz[2]] = 0
    print('max signal value is', np.amax(vol))
    print('min signal value is', np.amin(vol))
    # normalize volumes to have range of 0 to 1
    maxsi = np.amax(vol)
    vol = np.float32(vol / np.amax(vol))  # volume1 is source
    elapsed = time.time() - t
    print('total time to convert to float32 and replace undefined values was', elapsed, 'seconds')
    return vol, maxsi


# helper function that crops (in the x and y directions) and then reprojects the volume
def crop_volume_in_xy_and_reproject(volume, cropfactor_x, cropfactor_y, blksz, raw_projection):
    """
    function that crops the volume used for training in the x and y directions
    :param volume: input volume to crop
    :param cropfactor_x: crop factor (between 0 to 1) in x direction
    :param cropfactor_y: crop factor (between 0 to 1) in y direction
    :param blksz: patch size (i.e. the length or height of patch, either integer or tuple)
    :return: cropped and reprojected output volume
    """
    # crop volume
    print('crop_volume_in_xy_and_reproject => initial shape is', volume.shape)

    if not isinstance(blksz, tuple):  # create tuple if not
        blksz = (blksz, blksz)
    xcropspan = (int(volume.shape[0] * cropfactor_x) // blksz[0]) * blksz[0]
    ycropspan = (int(volume.shape[1] * cropfactor_y) // blksz[1]) * blksz[1]
    xcropmini = int((volume.shape[0] - xcropspan) / 2)
    ycropmini = int((volume.shape[1] - ycropspan) / 2)
    volume = volume[xcropmini:xcropmini + xcropspan, ycropmini:ycropmini + ycropspan, :]
    print('crop_volume_in_xy_and_reproject => cropped shape is', volume.shape)

    # reproject volume
    if raw_projection > 0:
        if raw_projection == 1:
            volume = np.moveaxis(volume, -1, 1)  # sagittal projection
        elif raw_projection == 2:
            volume = np.moveaxis(volume, -1, 0)  # coronal projection
        else:
            raise ValueError('raw_projection is invalid, quit out')
        print('crop_volume_in_xy_and_reproject => reprojected shape is', volume.shape)

    return volume


# helper function that undoes a prior reprojection
def undo_reproject(volume, raw_projection):
    """
    helper function that undoes a prior reprojection
    :param volume: volume to undo reprojection on
    :param raw_projection: direction of prior reprojection
    :return: volume with reprojection undone
    """
    if raw_projection > 0:  # switch back to x,y,z order
        print('undo_reproject => initial shape is', volume.shape)
        if raw_projection == 1:
            volume = np.moveaxis(volume, -1, 1)
        elif raw_projection == 2:
            volume = np.moveaxis(volume, 0, -1)
        else:
            raise ValueError('raw_projection is invalid, quit out')
        print('undo_reproject => output shape is', volume.shape)
    return volume


# helper function that gets patches from a source and target image
from skimage.feature import peak_local_max


def get_local_max_patches_from_image_unaugmented(img_target, img_source, blksz, n_highsig=12, n_lowsig=0, input_ch=1,
                                                 dif_mode=0):
    """
    # helper function that obtains patches (based on local maxima) from a source and target image; size is [nrows x ncols x 1] or [nrows x ncols x n]
    :param img_target: target image to predict
    :param img_source: source image to predict target from
    :param blksz: patch size in rows x columns (integer or tuple)
    :param n_highsig: number of high signal intensity patches
    :param n_lowsig: number of low signal intensity patches
    :param input_ch: input of input channels for the neural network
    :param dif_mode: mode for selecting the 'high' and 'low' signal patches; 0: normal mode based purely on signal intensity; 1: based on the difference of the (source and target signals) * sqrt(target signal intensity)
    :return: a dtype='float16' numpy matrix of size [n_highsig + n_lowsig, blksz, blksz, srcslice+tgtslice]
    """
    srcslice = 1 if len(img_source.shape) == 2 else img_source.shape[2]  # number of slices
    tgtslice = 1 if len(img_target.shape) == 2 else img_target.shape[2]

    if not isinstance(blksz, tuple):  # convert to tuple
        blksz = (blksz, blksz)

    # get the central target slice
    if tgtslice == 1:
        img_target_mid = img_target
    else:
        mid_slice = np.int(np.ceil(tgtslice / 2)) - 1
        # print('tgtshape, tgtslice, mid_slice = ', tgtshape, ', ', tgtslice, ', ', mid_slice)
        img_target_mid = np.squeeze(img_target[:, :, mid_slice])

    # compute metric image used to select patches
    if dif_mode == 0:
        coordinates = peak_local_max(img_target_mid, min_distance=10)
    else:
        if srcslice == 1:
            img_dif = abs(img_target_mid - np.squeeze(img_source)) * np.sqrt(img_target_mid)
        else:
            slc_before_mid = img_source.shape[2] // 2
            img_dif = abs(img_target_mid - np.squeeze(img_source[:, :, slc_before_mid])) * np.sqrt(img_target_mid)
        coordinates = peak_local_max(ndi.gaussian_filter(img_dif, 2), min_distance=10)

    sigs = img_target_mid[coordinates[:, 0], coordinates[:, 1]]
    idx = np.argsort(sigs)
    highidx = idx[-n_highsig:]
    coordi = coordinates[highidx, :]

    # create matrix that will be returned
    output = np.zeros([n_highsig + n_lowsig, blksz[0], blksz[1], srcslice + tgtslice], dtype='float16')
    if tgtslice == 1:
        img_tgt = np.squeeze(img_target[:, :])
    else:
        img_tgt = np.squeeze(img_target[:, :, :])

    if srcslice > 1:
        img_src = np.squeeze(img_source[:, :, :])
    else:
        img_src = np.squeeze(img_source[:, :])

    for i in range(0, n_highsig):
        # save image patches around these vessels
        miny = min(max(coordi[i, 0] - blksz[0] // 2, 0), img_tgt.shape[0] - blksz[0] - 1)
        maxy = miny + blksz[0]
        minx = min(max(coordi[i, 1] - blksz[1] // 2, 0), img_tgt.shape[1] - blksz[1] - 1)
        maxx = minx + blksz[1]
        if tgtslice == 1:
            output[i, :, :, 0] = img_tgt[miny:maxy, minx:maxx]
        else:
            output[i, :, :, 0:tgtslice] = img_tgt[miny:maxy, minx:maxx, :]
        if srcslice == 1:
            output[i, :, :, tgtslice] = img_src[miny:maxy, minx:maxx]
        else:
            output[i, :, :, tgtslice:] = img_src[miny:maxy, minx:maxx, :]

    if n_lowsig > 0:
        coordi_rand = np.zeros((n_lowsig, 2))
        for j in range(0, n_lowsig):
            randy = np.random.randint(0, img_tgt.shape[0] - 1 - blksz[0])
            randx = np.random.randint(0, img_tgt.shape[1] - 1 - blksz[1])
            # save image patches around these vessels
            if tgtslice == 1:
                output[j + n_highsig, :, :, 0] = img_tgt[randy:randy + blksz, randx:randx + blksz]
            else:
                output[j + n_highsig, :, :, 0:tgtslice] = img_tgt[randy:randy + blksz, randx:randx + blksz, :]
            if srcslice == 1:
                output[j + n_highsig, :, :, tgtslice] = img_src[randy:randy + blksz, randx:randx + blksz]
            else:
                output[j + n_highsig, :, :, tgtslice:] = img_src[randy:randy + blksz, randx:randx + blksz, :]
            coordi_rand[j, :] = [randy + blksz[0] // 2, randx + blksz[1] // 2]

    return output


# performs 3D region growing using an initial set of points
def returnarteriallocations_volume_regiongrowing_v2(volume, tmax=0.3, tmin=0.2, multiplier=3):
    """
    performs 3D region growing using an initial set of points
    :param volume: data volume to region grow within
    :param tmax: starting signal threshold value for initial point selection
    :param tmin: minimum signal threshold value for additional point selection
    :param multiplier: multiplier for knowing when to iteratively increase 'tmin' to restrict region growing
    :return: list containing [0] time history of points to process, and [1] volume of region that was grown (of type 'uint8')
    """

    vol_processedpoints = np.zeros(volume.shape, dtype='uint8')
    vol_processedpoints = np.where(volume >= tmax, 2,
                                   0)  # mark locations as '2', which indicates that they need to be analyzed

    adjusttmin = True  # flag for adjusting 'tmin' up based on number of additional points that will be added
    # adding too many additional points means that our 'tmin' threshhold is too low and needs to be increased
    n_initial = len(np.argwhere(volume >= tmax))  # find number of initial arterial points
    while (adjusttmin):
        n_finalest = len(np.argwhere(volume >= tmin))
        if n_finalest > multiplier * n_initial:  # if we now have more than 'multiplier' times the initial points, increase the 'tmin' threshold so that this value is <3
            tmin = min(tmax, 1.1 * tmin)
        else:
            adjusttmin = False
    print('tmax', tmax, 'tmin', tmin, n_finalest / n_initial)

    pointstoprocess_timehist = []  # time history of the number of points to process

    while len(np.argwhere(vol_processedpoints == 2)) > 0:  # loop while there are locations to be analyzed
        points = np.argwhere(vol_processedpoints == 2)
        pointstoprocess_timehist.append(len(points))  # add to time history
        print('region growing... number of additional points to analyze is', len(points))
        for s in range(len(points)):  # loop over arterial points that require processing
            x, y, z = points[s][0], points[s][1], points[s][2]
            if x < 1 or x >= volume.shape[0] - 1:  # at volume edge so just mark as analyzed
                vol_processedpoints[x, y, z] = 1
            elif y < 1 or y >= volume.shape[1] - 1:  # at volume edge so just mark as analyzed
                vol_processedpoints[x, y, z] = 1
            elif z < 1 or z >= volume.shape[2] - 1:  # at volume edge so just mark as analyzed
                vol_processedpoints[x, y, z] = 1
            else:
                # print('processing point at',x,y,z)
                minivol = volume[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]
                minivol[1, 1, 1] = 0  # don't add central point as this is currently being analyzed
                minivol = np.where(minivol < tmin, 0, minivol)  # find points with signal intensities >= 'tmin'
                arterialpoints = np.argwhere(minivol)
                for s1 in range(len(arterialpoints)):
                    if vol_processedpoints[
                        x + arterialpoints[s1, 0] - 1, y + arterialpoints[s1, 1] - 1, z + arterialpoints[
                            s1, 2] - 1] != 1:
                        # add this as a new point
                        vol_processedpoints[
                            x + arterialpoints[s1, 0] - 1, y + arterialpoints[s1, 1] - 1, z + arterialpoints[
                                s1, 2] - 1] = 2
                        # mark current point as "analyzed"
                vol_processedpoints[x, y, z] = 1

    # return point count history and volume of arterial points
    return pointstoprocess_timehist, vol_processedpoints.astype('uint8')


# helper function to determine if we need to train a network from scratch
def should_we_train_network(filenameprefix, leave_one_out_train, srcfiles):
    """
    helper function to determine if we need to train a network from scratch
    :param filenameprefix: file name prefix
    :param leave_one_out_train: flag if we are training using a 'leave one out' scheme
    :param srcfiles: id codes for data sets to reconstruct (needed if we're in leave one out mode)
    :return: list containing: [0] boolean if we still need to train; [1] set numbers that we need to still train
    """
    needtotrain = False
    setstotrain = []
    if not leave_one_out_train:
        if not os.path.exists(os.path.join(filenameprefix + ".json")):
            print("couldn't find ", os.path.join(filenameprefix + ".json"))
            needtotrain = True
    else:
        for i in range(len(srcfiles)):
            if not os.path.exists(os.path.join(filenameprefix + "_set" + str(i + 1) + ".json")):
                print("couldn't find ", os.path.join(filenameprefix + "_set" + str(i + 1) + ".json"))
                needtotrain = True
                setstotrain.append(i)
    return needtotrain, setstotrain


# SSIM loss function for Keras
# from https://stackoverflow.com/questions/57357146/use-ssim-loss-function-with-keras
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


# function that applies a custom 3D filter to a volume in all 3 directions followed by a MIP of each result
def filter_stack_in_3dirs_3dfilter(volume_in, custom_filter):
    """
    volin      - input 3D volume
    filter     - input nD filter
    """
    volout = np.zeros(volume_in.shape, dtype='float32')
    volout = ndi.convolve(volume_in, custom_filter)  # filter in axial direction
    voltmp = ndi.convolve(np.swapaxes(volume_in, 2, 0), custom_filter)  # filter in coronal direction
    volout = np.maximum(volout, np.swapaxes(voltmp, 2, 0))
    voltmp = ndi.convolve(np.swapaxes(volume_in, 2, 1), custom_filter)  # filter in sagittal direction
    volout = np.maximum(volout, np.swapaxes(voltmp, 2, 1))
    return volout


# function that applies a custom 1D filter to a volume in all 3 directions followed by a MIP of each result
def filter_stack_in_3dirs_1dfilter(volume_in, custom_filter, axs=[-1]):
    """
    volin      - input 3D volume
    filter     - input 1D filter
    axs        - list of axes (indices) to filter over
    """
    volout = np.zeros(volume_in.shape, dtype='float32')
    for i, axs in enumerate(axs):
        voltmp = ndi.convolve1d(volume_in, custom_filter, axis=axs)  # filter along axis 'axs'
        volout = np.maximum(volout, voltmp)
    return volout


# function to calculate local standard deviation in an image using convolution
# from https://stackoverflow.com/questions/25910050/perform-local-standard-deviation-in-python
def compute_local_standard_deviation(img, N, fastestimatemode=True):
    im1 = np.array(img, dtype=float)
    im2 = im1 ** 2
    ones = np.ones(im1.shape)
    kernel = np.ones((2 * N + 1, 2 * N + 1))
    s = convolve2d(im1, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    if not fastestimatemode:
        ns = convolve2d(ones, kernel, mode="same")
        return np.sqrt(np.abs((s2 - s ** 2 / ns) / ns))
    else:
        return np.sqrt(np.abs((s2 - s ** 2 / (2 * N + 1) ** 2) / (2 * N + 1) ** 2))


# function to calculate local standard deviation in an image using convolution
# from https://stackoverflow.com/questions/25910050/perform-local-standard-deviation-in-python
def compute_local_standard_deviation_3d(vol, N, fastestimatemode=True):
    v1 = np.array(vol, dtype=float)
    v2 = v1 ** 2
    ones = np.ones(v1.shape)
    s = ndi.uniform_filter(v1, 2 * N + 1)
    s2 = ndi.uniform_filter(v2, 2 * N + 1)
    if not fastestimatemode:
        ns = ndi.uniform_filter(ones, 2 * N + 1)
        return np.sqrt(np.abs((s2 - s ** 2 / ns) / ns))
    else:
        return np.sqrt(np.abs((s2 - s ** 2 / (2 * N + 1) ** 3) / (2 * N + 1) ** 3))


# helper function that creates patches during the 'test'/prediction phase
def get_patches(slc, blksz, blksz_recon):
    """
    slc         - imaging slice to get pathes from (2d numpy array)
    blksz       - block/patch size during training phase
    blksz_recon - block/patch size during reconstruction/prediction/test phase
    """
    if isinstance(blksz, tuple) and isinstance(blksz_recon, tuple):
        patches = extract_patches_2d_centerpixels(slc, blksz, blksz_recon)
    else:
        patches = extract_patches_2d_centerpixels(slc, (blksz, blksz), (blksz_recon, blksz_recon))
    return patches


def get_patches_2p5d(slc, blksz, blksz_recon):
    """
    slc         - imaging slice to get pathes from (2d numpy array)
    blksz       - block/patch size during training phase (either an integer or a tuple of integers)
    blksz_recon - block/patch size during reconstruction/prediction/test phase (either an integer or a tuple of integers)
    """
    if isinstance(blksz, tuple) and isinstance(blksz_recon, tuple):
        patches = extract_patches_2p5d_centerpixels(slc, blksz, blksz_recon)
    else:
        patches = extract_patches_2p5d_centerpixels(slc, (blksz, blksz), (blksz_recon, blksz_recon))
    return patches


# function that extracts all patches except at image edges (similar to sklearn.feature_extraction.image.extract_patches_2d)
def extract_patches_2d_centerpixels(arr, blksize, stride):
    """
    arr         - 2D image (nrow x ncol) to get pathes from (2d numpy array)
    blksz       - block/patch size during training phase (tuple)
    stride      - block/patch size during reconstruction/prediction/test phase (tuple)

    outputs:    patches (npatches, nrow, ncol)
    """
    # supports even and odd blksize values
    h = int(arr.shape[0] - (blksize[0] - stride[0])) // stride[0]
    w = int(arr.shape[1] - (blksize[1] - stride[1])) // stride[1]
    patches = np.zeros([h * w, blksize[0], blksize[1]], dtype=np.float16)
    for ih in range(0, h):
        hoffset = int(stride[0] * ih)
        for iw in range(0, w):
            woffset = int(stride[1] * iw)
            patches[ih * w + iw, :, :] = arr[hoffset:blksize[0] + hoffset, woffset:blksize[1] + woffset]
    return patches


def extract_patches_2p5d_centerpixels(arr, blksize, stride):
    """
    arr         - 3D image (nrow x ncol x ncha) to get pathes from (3d numpy array)
    blksz       - block/patch size during training phase (tuple or 2 values)
    stride      - block/patch size during reconstruction/prediction/test phase (tuple of 2 values)

    outputs:    patches (npatches, nrow, ncol, ncha)
    """
    numch = arr.shape[2]
    # supports even and odd blksize values
    h = int(arr.shape[0] - (blksize[0] - stride[0])) // stride[0]
    w = int(arr.shape[1] - (blksize[1] - stride[1])) // stride[1]
    patches = np.zeros([h * w, blksize[0], blksize[1], numch], dtype=np.float16)
    for ih in range(0, h):
        hoffset = int(stride[0] * ih)
        for iw in range(0, w):
            woffset = int(stride[1] * iw)
            patches[ih * w + iw, :, :, :] = arr[hoffset:blksize[0] + hoffset, woffset:blksize[1] + woffset, :]
    return patches


# function that reconstructs an image from the central sub-patches of all incoming image patches (similar to sklearn.feature_extraction.image.reconstruct_from_patches_2d)
def reconstruct_from_patches_2d_centerpixels(patch, image_size, stride, edgeblur=True, fastmode=True):
    """
    function that reconstructs an image from the central sub-patches of all incoming image patches (similar to sklearn.feature_extraction.image.reconstruct_from_patches_2d)
    :param patch: image patch matrix to reconstruct; of shape (# patches,blksize,blksize)
    :param image_size: tuple containing image size of image to be reconstructed
    :param stride: either an integer or an integer tuple of length 2; stride size in pixels in x and y direction for recontruction
    :param edgeblur: boolean that enables/disables patch edge blurring to reduce blocky reconstruction
    :param fastmode: flag for running more efficient code
    :return: reconstructed image
    """

    if not isinstance(stride, tuple):  # create tuple if not
        stride = (stride, stride)

    # supports even and odd patch sizes
    h = int(image_size[0] - (patch.shape[1] - stride[0])) // stride[0]
    w = int(image_size[1] - (patch.shape[2] - stride[1])) // stride[1]

    if edgeblur:  # if we're blurring edge voxels to avoid a blocky reconstruction
        reconimg = np.zeros([image_size[0], image_size[1]],
                            dtype=np.float32)  # reconstructed image to patch together that holds the signal intensities
        weightimg = np.zeros([image_size[0], image_size[1]],
                             dtype=np.float32)  # weighting image that holds the number of cumulative weights for each location in reconstructed image
        weightingmatrix = np.zeros([patch.shape[1], patch.shape[2]], dtype=np.float32)

        # linearly decrease weights from patch center
        assert patch.shape[1] == patch.shape[2]  # ensure these are the same to properly compute weightingmatrix
        wmtrx = np.abs(np.mgrid[0.5:patch.shape[1]:1, 0.5:patch.shape[2]:1] - patch.shape[1] / 2)
        weightingmatrix = abs((np.maximum(wmtrx[0], wmtrx[1]) + 0.5) - patch.shape[1] / 2) + 1
        weightingmatrix = weightingmatrix / patch.shape[1]

        if fastmode:
            weightingmatrix_poly1 = np.tile(weightingmatrix, [patch.shape[0], 1])
            weightingmatrix_poly2 = np.reshape(weightingmatrix_poly1, [patch.shape[0], patch.shape[1], patch.shape[2]])
            patches_poly_weighted = np.multiply(np.squeeze(patch), weightingmatrix_poly2)

            for ih in range(0, h):  # loop over patch rows
                hoffset = int(ih * stride[0])
                for iw in range(0, w):  # loop over patch columns
                    woffset = int(iw * stride[1])
                    reconimg[hoffset:hoffset + patch.shape[1],
                    woffset:woffset + patch.shape[2]] += patches_poly_weighted[ih * w + iw, :, :]
                    weightimg[hoffset:hoffset + patch.shape[1], woffset:woffset + patch.shape[2]] += weightingmatrix
        else:
            for ih in range(0, h):  # loop over patch rows
                hoffset = int(ih * stride[0])
                for iw in range(0, w):  # loop over patch columns
                    woffset = int(iw * stride[1])
                    reconimg[hoffset:hoffset + patch.shape[1], woffset:woffset + patch.shape[2]] += np.multiply(
                        np.squeeze(patch[ih * w + iw, :, :]), weightingmatrix)
                    weightimg[hoffset:hoffset + patch.shape[1], woffset:woffset + patch.shape[2]] += weightingmatrix

        weightimg[np.where(weightimg <= 0)] = 1
        reconimg = np.divide(reconimg,
                             weightimg)  # divide the reconstructed image by the weighting image to equalize signal across the image prior to returning it
        reconimg = np.float16(reconimg)  # convert back to float16 before returning
    else:  # standard block by block recon without blurring of edge pixels
        hoffsettoedgeofpatchcenter = patch.shape[1] // 2 - stride[
            0] // 2  # vertical   offset from edge of incoming patch to edge of center region being reconstructed
        woffsettoedgeofpatchcenter = patch.shape[2] // 2 - stride[
            1] // 2  # horizontal offset from edge of incoming patch to edge of center region being reconstructed
        reconimg = np.zeros([image_size[0], image_size[1]], dtype=np.float16)
        for ih in range(0, h):
            hoffset = int(
                ih * stride[0]) + hoffsettoedgeofpatchcenter  # vertical   offset into final reconstructed image
            for iw in range(0, w):
                woffset = int(
                    iw * stride[1]) + woffsettoedgeofpatchcenter  # horizontal offset into final reconstructed image
                reconimg[hoffset:hoffset + stride[0], woffset:woffset + stride[1]] = np.squeeze(
                    patch[ih * w + iw, patch.shape[1] // 2 - stride[0] // 2:patch.shape[1] // 2 + stride[0] // 2,
                    patch.shape[2] // 2 - stride[1] // 2:patch.shape[2] // 2 + stride[1] // 2])

    return reconimg


# function that returns the neural network .h5 and .json files
def get_model_and_json_files(dirmodel, model_to_apply, blks_rand_shift_mode, loo_training_mode, loo_dataset_to_recon,
                             stride_tuple, additional_string_in_filename='.', debug=False):
    """
    dirmodel             - directory holding neural network model
    model_to_apply       - text of model to look for
    blks_rand_shift_mode - random blocks shift mode during training for residual network
    loo_training_mode    - leave one out training mode flag (needed to choose appropriate model)
    loo_dataset_to_recon - leave one out data set number to reconstruct (used to choose appropriate model)
    stride_tuple         - block stride using during training
    additional_substring_in_filename - additional descriptive substring to find in filename
    debug                - printouts for debugging

    outputs:             modelFileName, jsonFileName
    """
    print('get_model_and_json_files, dirmodel     =>', dirmodel)
    print('get_model_and_json_files, stride_tuple =>', stride_tuple)
    NxtHdf = 1
    files = []
    for root, _, files1 in os.walk(dirmodel):
        if debug: print('files1: ', files1)
        files.append(files1)
        break  # only

    found_model = False
    found_json = False

    modelFileName = []
    jsonFileName = []

    if len(stride_tuple) == 2:
        stridesting = '[' + str(stride_tuple[0]) + 'x' + str(stride_tuple[1]) + ']'
    elif len(stride_tuple) == 3:
        stridesting = '[' + str(stride_tuple[0]) + 'x' + str(stride_tuple[1]) + 'x' + str(stride_tuple[2]) + ']'
    print('get_model_and_json_files, stridesting    =>', stridesting)
    print('get_model_and_json_files, model_to_apply =>', model_to_apply)
    if loo_training_mode:
        loo_string = 'set' + str(loo_dataset_to_recon)
        dashoffset = -4
        for file in files[0]:
            if debug: print(file)
            if model_to_apply in file and loo_string in file and stridesting in file and additional_string_in_filename in file:
                if debug: print('match', file)
                if blks_rand_shift_mode:  # if we are getting models using randomly shifted blocks
                    if '_rsb' in file:
                        if file.endswith('.json'):
                            jsonFileName = file
                            found_json = True
                        elif file.endswith('.hdf5'):
                            hdfname = file.split('-')[dashoffset]
                            if NxtHdf <= int(hdfname):
                                modelFileName = file
                                NxtHdf = int(hdfname)
                                found_model = True
                else:  # standard models trained without randomly shifted blocks
                    if '_rsb' not in file:
                        if file.endswith('.json'):
                            jsonFileName = file
                            found_json = True
                        elif file.endswith('.hdf5'):
                            hdfname = file.split('-')[dashoffset]
                            if NxtHdf <= int(hdfname):
                                modelFileName = file
                                NxtHdf = int(hdfname)
                                found_model = True
    else:
        loo_string = '-set'
        dashoffset = -3
        for file in files[0]:
            if model_to_apply in file and loo_string not in file and stridesting in file and additional_string_in_filename in file:
                if blks_rand_shift_mode:  # if we are getting models using randomly shifted blocks
                    if '_rsb' in file:
                        if file.endswith('.json'):
                            jsonFileName = file
                            found_json = True
                        elif file.endswith('.hdf5'):
                            hdfname = file.split('-')[dashoffset]
                            if NxtHdf <= int(hdfname):
                                modelFileName = file
                                NxtHdf = int(hdfname)
                                found_model = True
                else:  # standard models trained without randomly shifted blocks
                    if '_rsb' not in file:
                        if file.endswith('.json'):
                            jsonFileName = file
                            found_json = True
                        elif file.endswith('.hdf5'):
                            hdfname = file.split('-')[dashoffset]
                            if NxtHdf <= int(hdfname):
                                modelFileName = file
                                NxtHdf = int(hdfname)
                                found_model = True

    print('modelFileName =>', modelFileName)
    print('jsonFileName  =>', jsonFileName)
    return modelFileName, jsonFileName


def get_blocks_within_volume(vol_ref, Vimgs, blksz_3d, stride_3d, n_larger, n_lower=0, seed=0,
                             shuffleP=False, metric_operator='sum', nregions=1, return_bvm=False):
    """
    vol_ref         - metric volume from which 'n_larger' high signal intensity and 'n_lower' random blocks should be selected
    Vimgs           - array of volumes to return blocks for
    blksz_3d        - block dimensions in pixel units (nx,ny,nz)
    stride_3d       - stride size in x, y, and z directions
    n_larger        - number of blocks that correspond to the highest values in 'vol_ref'
    n_lower         - number of blocks that are to be randomly selected from 'vol_ref'
    seed            - seed for np.random
    shuffleP        - flag for enabling minor random shifts (up to blksz_3d[x]//4) during block selection
    nregions        - number of regions in slice direction to break volume into (get n_large/regions blocks per region)
    return_bvm      - flag for returning a high signal block volume useful for seeing where blocks were selected

    outputs:        - xtrain, ytrain, blockvolmap  (if return_bvm == True)
                    - xtrain, ytrain                  (if return_bvm == False)
    """
    # !! attention !!: len(Vimgs) must be the number of datasets. ie Vimgs = [dataset] if dataset is name of one dataset
    t = time.time()
    indx_ravel, n_volx, n_voly, n_volz, xmini, ymini, zmini = get_top_block_locations(vol_ref, blksz_3d, stride_3d,
                                                                                      n_larger, seed, n_lower=n_lower,
                                                                                      metric_operator=metric_operator,
                                                                                      regions=nregions)
    elapsed = time.time() - t
    print('total time obtaining high blocks is using old approach was', elapsed, 'seconds')
    if return_bvm: blockvolmap = np.zeros(vol_ref.shape, dtype='float32')
    xidx, yidx, zidx = np.unravel_index(indx_ravel, (n_volx, n_voly, n_volz))
    xtrain = []
    ytrain = []
    for iv in range(len(Vimgs)):  # loop over incoming volumes
        V1 = Vimgs[iv]
        Vblocks = np.ones((len(indx_ravel), blksz_3d[0], blksz_3d[1], blksz_3d[2]))
        np.random.seed(seed)  # set seed to common value to ensure random integer generation is same for all Vimgs
        psftxarr = np.random.randint(-blksz_3d[0] // 4, blksz_3d[0] // 4 + 1, len(indx_ravel))
        psftyarr = np.random.randint(-blksz_3d[1] // 4, blksz_3d[1] // 4 + 1, len(indx_ravel))
        psftzarr = np.random.randint(-blksz_3d[2] // 4, blksz_3d[2] // 4 + 1, len(indx_ravel))
        for i in range(len(indx_ravel)):
            if shuffleP:
                psftx = psftxarr[i]
                psfty = psftyarr[i]
                psftz = psftzarr[i]
                if xidx[i] == 0:
                    psftx = np.max([0, psftx])
                if (xidx[i] + 1) == n_volx:
                    psftx = np.min([0, psftx])
                if yidx[i] == 0:
                    psfty = np.max([0, psfty])
                if (yidx[i] + 1) == n_voly:
                    psfty = np.min([0, psfty])
                if zidx[i] == 0:
                    psftz = np.max([0, psftz])
                if (zidx[i] + 1) == n_volz:
                    psftz = np.min([0, psftz])
            else:
                psftx = 0
                psfty = 0
                psftz = 0
            x0 = xmini + xidx[i] * stride_3d[0] + psftx
            y0 = ymini + yidx[i] * stride_3d[1] + psfty
            z0 = zmini + zidx[i] * stride_3d[2] + psftz
            Vblocks[i, :, :, :] = V1[x0:x0 + blksz_3d[0], y0:y0 + blksz_3d[1], z0:z0 + blksz_3d[2]]
            if iv == 1 and return_bvm:
                if i < n_larger:
                    blockvolmap[x0:x0 + blksz_3d[0], y0:y0 + blksz_3d[1], z0:z0 + blksz_3d[2]] = 1.0
            # verify the selected images
        print('mean of whole volume, mean of selected volumes: ', np.mean(Vimgs[iv]), ', ', np.mean(Vblocks))
        if iv == 0:
            xtrain = Vblocks
        else:
            ytrain = Vblocks
        if return_bvm: blockvolmap = np.multiply(blockvolmap, Vimgs[iv])
    if return_bvm:
        return xtrain, ytrain, blockvolmap
    else:
        return xtrain, ytrain


def get_top_block_locations(volume, blksz_tuple, stride_tuple, n_larger, seed, n_lower=0, metric_operator='sum',
                            regions=1):
    """
    volume          - metric volume from which 'n_larger' high signal intensity and 'n_lower' random blocks should be selected
    blksz_tuple     - block dimensions in pixel units (nx,ny,nz)
    stride_tuple    - stride size in x, y, and z directions
    n_larger        - number of blocks that correspond to the highest values in 'volume_dif'
    seed            - seed for np.random
    n_lower         - number of blocks that are to be randomly selected from 'volume_dif'
    metric_operator - operator for choosing blocks (either 'sum' or 'max')
    regions         - number of regions in slice direction to break volume into (get n_large/regions blocks per region)

    outputs:        indx_ravel,     (continuous flattened array)
                    n_volx,         (# of x positions)
                    n_voly,         (# of y positions)
                    n_volz,         (# of z positions)
                    xmini,          (x offset index)
                    ymini,          (y offset index)
                    zmini,          (z offset index)
    """
    volshape = volume.shape

    # supports even and odd blksz_tuple sizes
    n_volx = (volshape[0] - (blksz_tuple[0] - stride_tuple[0])) // stride_tuple[0]
    n_voly = (volshape[1] - (blksz_tuple[1] - stride_tuple[1])) // stride_tuple[1]
    n_volz = (volshape[2] - (blksz_tuple[2] - stride_tuple[2])) // stride_tuple[2]
    xspan = n_volx * stride_tuple[0] + (blksz_tuple[0] - stride_tuple[0])
    yspan = n_voly * stride_tuple[1] + (blksz_tuple[1] - stride_tuple[1])
    zspan = n_volz * stride_tuple[2] + (blksz_tuple[2] - stride_tuple[2])
    xmini = int((volshape[0] - xspan) / 2)
    ymini = int((volshape[1] - yspan) / 2)
    zmini = int((volshape[2] - zspan) / 2)

    blk_sum_tmp = np.zeros((n_volx, n_voly, n_volz), dtype=np.single)

    if metric_operator == 'sum':
        for ix in range(n_volx):
            int_sumx = np.squeeze(
                np.sum(volume[xmini + ix * stride_tuple[0]:xmini + ix * stride_tuple[0] + blksz_tuple[0], :, :],
                       axis=0))
            for iy in range(n_voly):
                int_sumy = np.squeeze(np.sum(
                    int_sumx[ymini + iy * stride_tuple[1]:ymini + (iy + 1) * stride_tuple[1] + blksz_tuple[1], :],
                    axis=0))
                for iz in range(n_volz):
                    blk_sum_tmp[ix, iy, iz] = np.sum(
                        int_sumy[zmini + iz * stride_tuple[2]:zmini + iz * stride_tuple[2] + blksz_tuple[2]])
    elif metric_operator == 'max':
        for ix in range(n_volx):
            int_sumx = np.squeeze(
                np.max(volume[xmini + ix * stride_tuple[0]:xmini + ix * stride_tuple[0] + blksz_tuple[0], :, :],
                       axis=0))
            for iy in range(n_voly):
                int_sumy = np.squeeze(np.max(
                    int_sumx[ymini + iy * stride_tuple[1]:ymini + (iy + 1) * stride_tuple[1] + blksz_tuple[1], :],
                    axis=0))
                for iz in range(n_volz):
                    blk_sum_tmp[ix, iy, iz] = np.max(
                        int_sumy[zmini + iz * stride_tuple[2]:zmini + iz * stride_tuple[2] + blksz_tuple[2]])

    print('total blocks in the dataset before selection: ', len(blk_sum_tmp.ravel()))
    # cordi_x, cordi_y, cordi_z = np.unravel_index(np.sort(vol_tmp.ravel()), (n_patchx, n_patchy, volshape[2]))
    if regions == 1:
        indx_ravel = np.argsort(blk_sum_tmp.ravel())[-n_larger:]  # raveled index of largests, with number of n_larger
    elif regions > 1:
        indx_ravel = np.argsort(blk_sum_tmp.ravel())
        good_indx = []
        for irgn in range(0, regions):  # loop over regions in slice direction and add n_large/regions blocks per region
            ncount = 0
            index = 1
            while ncount < (n_larger / regions):
                # print(irgn, index," ",abs(len(indx_ravel)-indx_ravel[-index])//(n_volz/regions))
                x, y, z = np.unravel_index(indx_ravel[-index], (n_volx, n_voly, n_volz))
                print(irgn, index, ncount, z, (n_volz - z) // (n_volz / regions))
                if ((n_volz - z) // (n_volz / regions)) == irgn:
                    good_indx.append(indx_ravel[-index])
                    ncount = ncount + 1
                index = index + 1
        indx_ravel = np.asarray(good_indx)
    else:
        sys.exit("number of regions is 0, quit!")
    if n_lower > 0:  # select at random
        np.random.seed(seed)
        indx_low = np.random.randint(low=1, high=n_volx * n_voly * n_volz, size=n_lower)
        indx_ravel = np.concatenate((indx_ravel, indx_low))
    print('in get_top_block_locations, n_lower  & indx_ravel.shape: ', n_lower, indx_ravel.shape)
    return indx_ravel, n_volx, n_voly, n_volz, xmini, ymini, zmini


def get_patches_within_volume(vol_ref, Vimgs, blksz_2d, stride_2d, n_larger, n_lower=0,
                              seed=0, shuffleP=False, metric_operator='sum'):
    """
    vol_ref         - metric volume from which 'n_larger' high signal intensity and 'n_lower' random patches should be selected
    Vimgs           - array of volumes to return patches for
    blksz_2d        - patch dimension in pixel units (nx, ny)
    stride_2d       - stride sizes for patch selection in x and y directions
    n_larger        - number of patches that correspond to the highest values in 'vol_ref'
    n_lower         - number of patches that are to be randomly selected from 'vol_ref'
    seed            - seed for np.random    
    shuffleP        - flag for enabling minor random shifts (up to blksz//4) during block selection
    metric_operator - metric operator (either 'sum' or 'max')

    outputs:        xtrain (input/low quality), ytrain (output/high quality) training patches
    """
    indx_ravel, n_patchx, n_patchy, xmini, ymini = get_top_patch_locations(vol_ref, blksz_2d,
                                                                           stride_2d, n_larger, seed,
                                                                           n_lower=n_lower,
                                                                           metric_operator=metric_operator)
    xidx, yidx, sidx = np.unravel_index(indx_ravel, (n_patchx, n_patchy, vol_ref.shape[2]))
    Vpatches = np.zeros((len(indx_ravel), blksz_2d[0], blksz_2d[1]))
    xtrain = []
    ytrain = []
    for iv in range(len(Vimgs)):  # loop over income volumes
        np.random.seed(seed)
        V1 = Vimgs[iv]
        # print('Vimgs.shape: ', Vimgs.shape)
        for i in range(len(indx_ravel)):
            if shuffleP:
                # psft = np.random.randint(0, blksz // 4, size=2)
                psft = np.random.randint(0, blksz_2d[0] // 4, size=2)
                if xidx[i] == 0:
                    psft[0] = np.max([0, psft[0]])
                if (xidx[i] + 1) == n_patchx:
                    # print('psft[0]: ', psft[0])
                    psft[0] = np.min([0, psft[0]])
                    # print('after replacement, psft[0]: ', psft[0])
                if yidx[i] == 0:
                    psft[1] = np.max([0, psft[1]])
                if (yidx[i] + 1) == n_patchy:
                    psft[1] = np.min([0, psft[1]])
                # print('xidx[i], xidx[i]*n_patchx, xmini+xidx[i]*blksz+psft[0], xmini+(xidx[i]+1)*blksz+psft[0]: ', xidx[i], xidx[i]*n_patchx, xmini+xidx[i]*blksz+psft[0], xmini+(xidx[i]+1)*blksz+psft[0])
            else:
                psft = [0, 0]
            x0 = xmini + xidx[i] * stride_2d[0] + psft[0]
            y0 = ymini + yidx[i] * stride_2d[1] + psft[1]
            Vpatches[i, :, :] = V1[x0:x0 + blksz_2d[0], y0:y0 + blksz_2d[1], sidx[i]]
            # verify the selected images
        print('mean of whole volume, mean of selected patches: ', np.mean(V1), ', ', np.mean(Vpatches))
        if iv == 0:
            xtrain = np.copy(Vpatches)
        else:
            ytrain = np.copy(Vpatches)
    return xtrain, ytrain


def get_top_patch_locations(volume, blksz_2d, stride_2d, n_larger, seed, n_lower=0, metric_operator='sum'):
    """
    volume          - metric volume from which 'n_larger' high signal intensity and 'n_lower' random patches should be selected
    blksz_2d        - patch dimensions in pixel units
    stride_2d       - stride sizes for patch selection in x and y directions
    n_larger        - number of patches that correspond to the highest values in 'volume_dif'
    seed            - seed for np.random
    n_lower         - number of patches that are to be randomly selected from 'volume_dif'
    metric_operator - compute either 'sum' or 'max' for each candidate patch

    outputs:        indx_ravel,     (continuous flattened array)
                    n_patchx,       (# of x positions)
                    n_patchy,       (# of y positions)
                    xmini,          (x offset index)
                    ymini           (y offset index)
    """
    # supports even and odd blksz_2d sizes
    volshape = volume.shape
    n_patchx = (volshape[0] - (blksz_2d[0] - stride_2d[0])) // stride_2d[0]
    n_patchy = (volshape[1] - (blksz_2d[1] - stride_2d[1])) // stride_2d[1]
    xspan = n_patchx * stride_2d[0] + (blksz_2d[0] - stride_2d[0])
    yspan = n_patchy * stride_2d[1] + (blksz_2d[1] - stride_2d[1])
    xmini = int((volshape[0] - xspan) / 2)
    ymini = int((volshape[1] - yspan) / 2)
    vol_tmp = np.zeros((n_patchx, n_patchy, volshape[2]), dtype=np.single)

    if metric_operator == 'sum':
        for ix in range(n_patchx):
            int_sumx = np.squeeze(
                np.sum(volume[xmini + ix * stride_2d[0]:xmini + (ix + 1) * stride_2d[0], :, :], axis=0))
            for iy in range(n_patchy):
                vol_tmp[ix, iy, :] = np.squeeze(
                    np.sum(int_sumx[ymini + iy * stride_2d[1]:ymini + (iy + 1) * stride_2d[1], :], axis=0))
    elif metric_operator == 'max':
        for ix in range(n_patchx):
            int_sumx = np.squeeze(
                np.max(volume[xmini + ix * stride_2d[0]:xmini + (ix + 1) * stride_2d[0], :, :], axis=0))
            for iy in range(n_patchy):
                vol_tmp[ix, iy, :] = np.squeeze(
                    np.max(int_sumx[ymini + iy * stride_2d[1]:ymini + (iy + 1) * stride_2d[1], :], axis=0))

    print('total patches in the volume before selection: ', len(vol_tmp.ravel()))
    # cordi_x, cordi_y, cordi_z = np.unravel_index(np.sort(vol_tmp.ravel()), (n_patchx, n_patchy, volshape[2]))
    indx_ravel = np.argsort(vol_tmp.ravel())[-n_larger:]  # raveled index of largests, with number of n_larger
    if n_lower > 0:
        np.random.seed(seed)
        indx_low = np.random.randint(low=1, high=n_patchx * n_patchy * volshape[2], size=n_lower)
        indx_ravel = np.concatenate((indx_ravel, indx_low))
    return indx_ravel, n_patchx, n_patchy, xmini, ymini


# from https://stackoverflow.com/questions/15361595/calculating-variance-image-python
def window_variance(arr, win_size):
    """
    function that computes the variance within a window size
    :param arr: input volume to process
    :param win_size: window size
    :return: variance of input volume
    """
    win_mean = ndi.uniform_filter(arr, win_size)
    win_sqr_mean = ndi.uniform_filter(arr ** 2, win_size)
    win_var = win_sqr_mean - win_mean ** 2
    return win_var ** .5


def compute_metric_volume_2d(volume1, volume3, patch_select_mode, stride_2d, n_slices_exclude):
    """
    # patch selection mode options
    # 0: sum of signal                          within patch
    # 1: sum of (target-source) * sqrt(target)  within patch
    # 2: sum of (target-source) * sqrt(source)  within patch
    # 3: sum of (target-source) * target        within patch
    # 4: sum of local standard deviation (of (2*stridex//2+1 x 2*stridex//2+1)) within patch
    # 5: sum of local standard deviation (of 5 x 5 region)                      within patch
    # 6: max of local standard deviation (of 5 x 5 region)                      within patch
    # 7: max of local standard deviation (of 3 x 3 region)                      within patch
    # 8: independent convolution filtering in all three directions using a custom 3D filter and MIP processing
    # 9: independent convolution filtering in all three directions using a custom 1D filter and MIP processing
    # 10: convolution filtering slice direction using a custom 1D filter
    # 11: convolution filtering slice direction using a custom 1D filter
    :param volume1: lower quality volume
    :param volume3: higher quality target volume
    :param patch_select_mode: patch selection algorithm
    :param stride_2d: stride of training patches
    :param n_slices_exclude: # number of slices to exclude
    :return: metric volume ("vol_metric"), metric operator ("metric_operator")
    """
    slc_train_end = volume1.shape[2] - n_slices_exclude
    metric_operator = 'sum'
    num_cores = multiprocessing.cpu_count() - 2
    if patch_select_mode == 0:  # 0: sum of signal                          within patch
        vol_metric = np.copy(volume3[:, :, n_slices_exclude:slc_train_end])
    elif patch_select_mode == 1:  # 1: sum of (target-source) * sqrt(target)  within patch, sum operator
        vol_metric = abs(
            volume3[:, :, n_slices_exclude:slc_train_end] - volume1[:, :, n_slices_exclude:slc_train_end]) * np.sqrt(
            volume3[:, :, n_slices_exclude:slc_train_end])
    elif patch_select_mode == 2:  # 2: sum of (target-source) * sqrt(source)  within patch, sum operator
        vol_metric = abs(
            volume3[:, :, n_slices_exclude:slc_train_end] - volume1[:, :, n_slices_exclude:slc_train_end]) * np.sqrt(
            volume1[:, :, n_slices_exclude:slc_train_end])
    elif patch_select_mode == 3:  # 3: sum of (target-source) * target        within patch, sum operator
        vol_metric = abs(
            volume3[:, :, n_slices_exclude:slc_train_end] - volume1[:, :, n_slices_exclude:slc_train_end]) * volume3[:,
                                                                                                             :,
                                                                                                             n_slices_exclude:slc_train_end]
    elif patch_select_mode == 4:  # 4: standard deviation                     within patch, sum operator
        vol_metric = np.zeros(volume3[:, :, n_slices_exclude:slc_train_end].shape, dtype='float32')
        print('computing local sds over all slices...')
        slice_positions = range(vol_metric.shape[2])
        results = Parallel(n_jobs=num_cores, max_nbytes=None)(
            delayed(compute_local_standard_deviation)(volume3[:, :, iSlc], stride_2d[0] // 2, True) for iSlc in
            slice_positions)
        for i, value in enumerate(results):
            vol_metric[:, :, i] = np.copy(value)
    elif patch_select_mode == 5:  # 5: standard deviation                     within patch of size 5x5, sum operator
        vol_metric = np.zeros(volume3[:, :, n_slices_exclude:slc_train_end].shape, dtype='float32')
        print('computing local sds over all slices...')
        slice_positions = range(vol_metric.shape[2])
        results = Parallel(n_jobs=num_cores, max_nbytes=None)(
            delayed(compute_local_standard_deviation)(volume3[:, :, iSlc], 2, True) for iSlc in slice_positions)
        for i, value in enumerate(results):
            vol_metric[:, :, i] = np.copy(value)
    elif patch_select_mode == 6:  # 6: standard deviation                     within patch of size 5x5, max operator
        metric_operator = 'max'
        vol_metric = np.zeros(volume3[:, :, n_slices_exclude:slc_train_end].shape, dtype='float32')
        print('computing local sds over all slices...')
        slice_positions = range(vol_metric.shape[2])
        results = Parallel(n_jobs=num_cores, max_nbytes=None)(
            delayed(compute_local_standard_deviation)(volume3[:, :, iSlc], 2, True) for iSlc in slice_positions)
        for i, value in enumerate(results):
            vol_metric[:, :, i] = np.copy(value)
    elif patch_select_mode == 7:  # 7: standard deviation                     within patch of size 3x3, max operator
        metric_operator = 'max'
        vol_metric = np.zeros(volume3[:, :, n_slices_exclude:slc_train_end].shape, dtype='float32')
        print('computing local sds over all slices...')
        slice_positions = range(vol_metric.shape[2])
        results = Parallel(n_jobs=num_cores, max_nbytes=None)(
            delayed(compute_local_standard_deviation)(volume3[:, :, iSlc], 1, True) for iSlc in slice_positions)
        for i, value in enumerate(results):
            vol_metric[:, :, i] = np.copy(value)
    elif patch_select_mode == 8:  # 8: independent convolution filtering in all three directions using a custom 3D filter and MIP processing
        custom_filter = np.array(
            [[-1, -1, -1, -1, -1], [-1, 1.75, 1.75, 1.75, -1], [-1, 1.75, 2, 1.75, -1], [-1, 1.75, 1.75, 1.75, -1],
             [-1, -1, -1, -1, -1]])
        custom_filter = filter[..., np.newaxis]  # convert 2D to 3D filter by adding a dimension
        print('computing local convolution filter...')
        vol_metric = filter_stack_in_3dirs_3dfilter(volume3, custom_filter)
        metric_operator = 'max'
    elif patch_select_mode >= 9:
        if patch_select_mode == 9:  # 9: independent convolution filtering in all three directions using a custom 1D filter and MIP processing
            custom_filter = np.array([-1, 0.5, 1, 0.5, -1])
            axestofilter = [0, 1, 2]
        if patch_select_mode == 10:  # 10: convolution filtering slice direction using a custom 1D filter
            custom_filter = np.array([-1, 0.5, 1, 0.5, -1])
            axestofilter = [-1]
        if patch_select_mode == 11:  # 11: convolution filtering slice direction using a custom 1D filter
            custom_filter = np.array([-1, 0, 2, 0, -1])
            axestofilter = [-1]

        # print('computing local convolution filter...')
        vol_metric = filter_stack_in_3dirs_1dfilter(volume3, custom_filter, axestofilter)
        metric_operator = 'max'

    return vol_metric, metric_operator


def compute_metric_volume_3d(volume1, volume3, patch_select_mode, stride_3d):
    """
    # patch selection mode options
    # 0: sum of signal                          within patch
    # 1: sum of (target-source) * sqrt(target)  within patch
    # 2: sum of (target-source) * sqrt(source)  within patch
    # 3: sum of (target-source) * target        within patch
    # 4: sum of local standard deviation (of (2*stridex//2+1 x 2*stridex//2+1 x 2*stridex//2+1)) within patch
    # 5: sum of local standard deviation (of 5 x 5 x 5 region) within patch
    # 6: max of local standard deviation (of 5 x 5 x 5 region) within patch
    # 7: max of local standard deviation (of 3 x 3 x 3 region) within patch
    # 8: independent convolution filtering in all three directions using a custom 3D filter and MIP processing
    # 9: independent convolution filtering in all three directions using a custom 1D filter and MIP processing
    # 10: convolution filtering slice direction using a custom 1D filter
    # 11: convolution filtering slice direction using a custom 1D filter
    :param volume1: lower quality volume
    :param volume3: higher quality target volume
    :param patch_select_mode: patch selection algorithm
    :param stride_3d: stride of training patches
    :return: metric volume ("vol_metric"), metric operator ("metric_operator"), nregions
    """
    nregions = 1
    t = time.time()
    metric_operator = 'sum'
    num_cores = multiprocessing.cpu_count() - 2
    if patch_select_mode == 0:  # 0: sum of signal                          within 3d block
        vol_metric = np.copy(volume3)
    elif patch_select_mode == 1:  # 1: sum of (target-source) * sqrt(target)  within 3d block, sum operator
        vol_metric = abs(volume1 - volume3) * np.sqrt(volume3)  # metric volume used for training block selection
    elif patch_select_mode == 2:  # 2: sum of (target-source) * sqrt(source)  within 3d block, sum operator
        vol_metric = abs(volume1 - volume3) * np.sqrt(volume1)  # metric volume used for training block selection
    elif patch_select_mode == 3:  # 3: sum of (target-source) * target        within 3d block, sum operator
        vol_metric = abs(volume1 - volume3) * volume3  # metric volume used for training block selection
    elif patch_select_mode == 4:  # 4: standard deviation                     within 3d block of size stride_3d[0] x stride_3d[0] x stride_3d[0], sum operator
        vol_metric = np.zeros(volume3.shape, dtype='float32')
        print('computing local sds over all blocks...')
        vol_metric = compute_local_standard_deviation_3d(volume3, stride_3d[0] // 2, True)
        print('computing local sds over all blocks...done')
    elif patch_select_mode == 5:  # 5: standard deviation                     within 3d block of size 5 x 5 x 5, sum operator
        vol_metric = np.zeros(volume3.shape, dtype='float32')
        print('computing local sds over all blocks...')
        vol_metric = compute_local_standard_deviation_3d(volume3, 5 // 2, True)
        print('computing local sds over all blocks...done')
    elif patch_select_mode == 6:  # 6: standard deviation                     within 3d block of size 5 x 5 x 5, max operator
        metric_operator = 'max'
        vol_metric = np.zeros(volume3.shape, dtype='float32')
        print('computing local sds over all blocks...')
        vol_metric = compute_local_standard_deviation_3d(volume3, 5 // 2, True)
        print('computing local sds over all blocks...done')
    elif patch_select_mode == 7:  # 7: standard deviation                     within 3d block of size 3 x 3 x 3, max operator
        metric_operator = 'max'
        vol_metric = np.zeros(volume3.shape, dtype='float32')
        print('computing local sds over all blocks...')
        vol_metric = compute_local_standard_deviation_3d(volume3, 1, True)
        print('computing local sds over all blocks...done')
    elif patch_select_mode == 8:  # 8: independent convolution filtering in all three directions using a custom 3D filter and MIP processing
        custom_filter = np.array(
            [[-1, -1, -1, -1, -1], [-1, 1.75, 1.75, 1.75, -1], [-1, 1.75, 2, 1.75, -1], [-1, 1.75, 1.75, 1.75, -1],
             [-1, -1, -1, -1, -1]])
        custom_filter = custom_filter[..., np.newaxis]  # make 2D filter a 3D filter by adding a dimension
        print('computing local convolution filter...')
        vol_metric = filter_stack_in_3dirs_3dfilter(volume3, custom_filter)
        metric_operator = 'max'
        nregions = 1
    elif patch_select_mode >= 9:
        if patch_select_mode == 9:  # 9: independent convolution filtering in all three directions using a custom 1D filter and MIP processing
            custom_filter = np.array([-1, 0.5, 1, 0.5, -1])
            axestofilter = [0, 1, 2]
        if patch_select_mode == 10:  # 10: convolution filtering slice direction using a custom 1D filter
            custom_filter = np.array([-1, 0.5, 1, 0.5, -1])
            axestofilter = [-1]
        if patch_select_mode == 11:  # 11: convolution filtering slice direction using a custom 1D filter
            custom_filter = np.array([-1, 0, 2, 0, -1])
            axestofilter = [-1]

        # print('computing local convolution filter...')
        vol_metric = filter_stack_in_3dirs_1dfilter(volume3, custom_filter, axestofilter)
        metric_operator = 'max'
        nregions = 1
    elapsed = time.time() - t

    return vol_metric, metric_operator, nregions


def set_augmentation_factor_for_3D_net(data_augm_factor):
    """
    input:
    data_augm_factor - prescribed data augmentation factor for 3D network

    output:          - data_augm_factor of either 1,2,4 or 8
    """
    if data_augm_factor >= 8:
        data_augm_factor = 8
    elif data_augm_factor >= 4:
        data_augm_factor = 4
    elif data_augm_factor >= 2:
        data_augm_factor = 2
    else:
        data_augm_factor = 1
    return data_augm_factor
