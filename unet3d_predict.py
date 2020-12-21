import numpy as np
# import matplotlib.pyplot as plt
import os, os.path
from sklearn.feature_extraction import image
from keras.models import model_from_json
from PIL import Image
import tensorflow as tf
import sys
import tifffile
import time
from joblib import Parallel, delayed
import multiprocessing
from utils import *

##############################################################################
# IMPORTANT RECONSTRUCTION OPTIONS (begin)
###############################################################################
combomatrix = [[8, 8, 8, 4, 4, 4, 9, 10000, 10000, 8,
                False]]  # blkszx, blkszy, blkszz, stridex, stridey, stridez, patchselectionmode, ntrainblockshigh,
                         # ntrainblockslow, unetstartchannels, resnet mode

testmode = False  # test by training/predicting the first case only for one reduction factor
reduction_list = [3] if testmode else [2, 3, 4, 5, 6]  # resolution reduction factors to train/predict
loss_function = 'mean_squared_error'  # 'mean_squared_error' 'ssim_loss'
sleep_when_done = False

leave_one_out_train = True  # performs training using a leave one out scheme
optimizers = ['adam']  # ['adam', 'sgd']

patches_from_volume = True  # False: patches selected from each slice; True: patches selected from whole volume

unet_dim = 3  # if unet_dim == 3, use Conv3D, etc
unet_start_ch = 8
unet_inc_rate = 2
unet_dropout = 0.5
unet_batchnorm = False
unet_residual = False
unet_batchnorm_str = 'F' if not unet_batchnorm else 'T'
unet_residual_str = 'F' if not unet_residual else 'T'

###############################################################################
# IMPORTANT RECONSTRUCTION OPTIONS (end)
###############################################################################
for iRed in reduction_list:  # loop over resolution reduction factors
    reduction = str(iRed) + 'fold'

    for b_index in combomatrix:  # loop over reconstructions to perform

        batch_size_train = 20 if b_index[0] == 32 else 32 // b_index[0] * 20
        unet_depth = 4 if b_index[0] ** 0.25 >= 2 else 3
        print(b_index)
        try:
            # Unet residual mode; Unet tries to predict the difference image/volume that will be added back to the
            # low res data to generate the high res data
            unet_resnet_mode = b_index[10]
        except:
            unet_resnet_mode = False
        try:
            blksz_3d = b_index[0], b_index[1], b_index[2]  # reconstruction block size in pixels
        except:
            print('could not read the block size, so quit out!')
            quit()
        try:
            stride_3d = b_index[3], b_index[4], b_index[5]  # reconstruction block stride in pixels
        except:
            stride_3d = b_index[0] // 2, b_index[1] // 2, b_index[2] // 2

        patches_per_slc_h = patches_per_set_h = b_index[7]
        patches_per_slc_l = patches_per_set_l = b_index[8]
        patches_per_slc = patches_per_set = patches_per_slc_h + patches_per_slc_l

        data_augm_factor = 1
        try:
            patch_select_mode = b_index[6]
        except:
            patch_select_mode = 1

        print(patch_select_mode)

        for optim in optimizers:  # loop through optimizers
            parallel_recon = True

            subset_recon_mode = False  # flag the allows to recon only a subset of slices
            subset_recon_minslc = 1  # min slice to recon
            subset_recon_maxslc = 100  # max slice to recon

            # factors for additional in-plane (i.e. x and y) cropping of volumes during recon/test/prediction phase
            crop_recon_x = 1.0 if not testmode else 0.5
            crop_recon_y = 1.0 if not testmode else 0.5

            batch_size_recon = 25 if b_index[0] == 32 else 32 // b_index[0] * 25

            # input channels for UNets
            if unet_dim == 2:
                input_ch = 1  # 1 slice for 2D UNet
            else:  # 3 slices for 2.5D UNet and b_index[2] slices for 3D UNet
                input_ch = b_index[
                    2] if not unet_dim == 2.5 else 3  # if input_ch==3, it's for 2.5D Unet with 3 input channels
                                                      # input_ch == 1 for normal 2D and 3D Unet

            raw_projection = 0  # 0: no projection, 1, along 1st dimnetion, 2 along 2nd dimension
            blks_rand_shift_mode = False  # 3D random block shift mode

            ###############################################################################
            # check if we've already trained the model; if no, train it, if yes, load it from disk
            ###############################################################################
            # construct folder name where models are
            if loss_function == "mean_squared_error":
                foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(raw_projection) + 'psm' + str(
                    patch_select_mode) + "_" + "mse"
            else:
                foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(raw_projection) + 'psm' + str(
                    patch_select_mode) + "_" + loss_function

            print(foldersuffix)

            rstr = "res" if unet_resnet_mode else ""
            p = patches_per_set, patches_per_set_h, patches_per_set_l if patches_from_volume else patches_per_slc, patches_per_slc_h, patches_per_slc_l
            if unet_dim == 3:
                modelsuffix = "_unet3d" + rstr + "-" + "[" + str(stride_3d[0]) + 'x' + str(stride_3d[1]) + 'x' + str(
                    stride_3d[2]) + "]-psm" + str(patch_select_mode) + "-" + str(unet_start_ch) + "-" + str(
                    unet_depth) + "-" + str(unet_inc_rate) + "-" + str(
                    unet_dropout) + "-" + unet_batchnorm_str + "-" + unet_residual_str + '-batch' + str(
                    batch_size_train)
                modelprefix = "model_" + str(blksz_3d[0]) + "x" + str(blksz_3d[1]) + "x" + str(blksz_3d[2]) + "x" + str(
                    p[0]) + "(" + str(p[1]) + ")(" + str(p[2]) + ")x" + str(data_augm_factor)
            else:
                quit
            if blks_rand_shift_mode:  # if randomly shifting blocks during training process
                modelprefix += '_rsb'

            ###############################################################################
            # load model architecture and weights from .json and .hdf5 files on disk
            ###############################################################################
            script_path = os.path.split(os.path.abspath(__file__))[0]
            # construct folder name where models are
            if unet_dim == 2:
                outpath = 'train_' + 'unet2d' + rstr + '_' + optim + '_' + reduction + '_batch' + str(
                    batch_size_train) + foldersuffix
            elif unet_dim == 2.5:
                outpath = 'train_' + 'unet2p5d' + rstr + '_' + optim + '_' + reduction + '_batch' + str(
                    batch_size_train) + foldersuffix
            elif unet_dim == 3:
                outpath = 'train_' + 'unet3d' + rstr + '_' + optim + '_' + reduction + '_batch' + str(
                    batch_size_train) + foldersuffix

            script_path = os.path.split(os.path.abspath(__file__))[0]
            dirmodel = os.path.join(script_path, outpath)
            if not os.path.exists(dirmodel):
                sys.exit("error - ", dirmodel, "doesn't exist, so can't predict")
            dirinput = os.path.join(script_path, 'input_low_res_' + reduction)

            ####################################
            # load tif files as source to fit
            ####################################
            inputfiles = []
            for root, _, files in os.walk(dirinput):
                for f in files:
                    if f.endswith('.tiff') or f.endswith('.tif'):
                        inputfiles.append(os.path.join(dirinput, f))
                break  # only check to level, no subdirs
            print('################')
            print('input files are')
            for ifile in inputfiles:
                print(ifile)
            print('################')

            ###########################################
            # perform deep learning reconstruction
            ###########################################
            for inputTifs in inputfiles:

                ########################################
                # load model from disk
                ########################################
                datasetnumber = inputfiles.index(inputTifs) + 1
                if testmode and datasetnumber > 1:  # only recon first set when in testmode
                    continue
                modelFileName, jsonFileName = get_model_and_json_files(dirmodel, modelprefix, blks_rand_shift_mode,
                                                                       leave_one_out_train, datasetnumber, stride_3d)
                if len(modelFileName) == 0:
                    print('could not find model for', inputfiles, 'so skip')
                    continue

                print('json file  =>', jsonFileName)
                print('model file =>', modelFileName)
                if int(modelFileName.split('-')[
                           -4]) == 1:  # check that we've adequately trained (i.e. look at the epoch number, if equal to 1, then don't reconstruct this set)
                    for i in range(0, 2): print('###########################################################')
                    print('only 1 epoch so skip over as we failed to train the network')
                    for i in range(0, 2): print('###########################################################')
                    # continue
                print('load json: ', os.path.join(dirmodel, jsonFileName))
                print('load weights: ', os.path.join(dirmodel, modelFileName))
                # add descriptive suffixes to .tif file name
                reconFileNameSuffix = '-'.join(modelFileName.split('-')[:-3]) + '.tif'
                reconFileNameSuffix = reconFileNameSuffix.split('/')[-1]
                if parallel_recon:               reconFileNameSuffix = '.'.join(
                    reconFileNameSuffix.split('.')[:-1]) + '_parallel.tif'
                if blks_rand_shift_mode:        reconFileNameSuffix = '.'.join(
                    reconFileNameSuffix.split('.')[:-1]) + '_rsb.tif'
                if leave_one_out_train: reconFileNameSuffix = '.'.join(reconFileNameSuffix.split('.')[:-1]) + '_loo.tif'
                print('reconFileNameSuffix: ', reconFileNameSuffix)

                fname = inputTifs.split('\\')[-1].split('_')[0] + '_' + str(iRed) + 'fold' + '_' + reconFileNameSuffix
                reconFileName = os.path.join(dirmodel, fname)
                print('reconFileName =>', reconFileName)
                if os.path.isfile(reconFileName):  # don't overwrite existing data
                    print('skipping recon of', reconFileName)
                    continue

                try:  # load model architecture from .json file
                    json_file = open(os.path.join(dirmodel, jsonFileName), 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    model = model_from_json(loaded_model_json)
                except:
                    print("error loading json file from disk")
                    sys.exit()

                try:  # load model weights from .hdf5 file
                    model.load_weights(os.path.join(dirmodel, modelFileName))
                except:
                    print("error loading model weights from disk")
                    sys.exit()

                ########################################
                # load tiff from disk for reconstruction
                ########################################
                reconFileName_final = reconFileName

                volume1 = load_tiff_to_numpy_vol(inputTifs, subset_recon_mode, subset_recon_minslc, subset_recon_maxslc)
                # adjust size to reduce computation load
                # adjust x and y dimensions of volume to divide evenly into blksz_3d
                volume1 = crop_volume_in_xy_and_reproject(volume1, crop_recon_x, crop_recon_y, blksz_3d[:2],
                                                          raw_projection)
                if len(np.argwhere(np.isinf(volume1))) > 0:
                    for xyz in np.argwhere(np.isinf(volume1)):
                        volume1[xyz[0], xyz[1], xyz[2]] = 0
                inputMax = np.amax(volume1)
                # normalize volumes to have range of 0 to 1
                volume1 = np.float16(volume1 / np.amax(volume1))

                # reconstruct the odd numbered slices using deep learning
                print('perform deep learning reconstruction...')
                if subset_recon_mode:
                    minslc = subset_recon_minslc - 1
                    maxslc = min(subset_recon_maxslc, volume1.shape[2])
                else:
                    minslc = 0
                    maxslc = volume1.shape[2]
                incslc = 1  # reconstruct every slice

                # loop over slices to reconstruct
                if unet_dim == 3:
                    input_ch = blksz_3d[2]

                # 3D neural network
                # 3D reconstruction
                debuglogic = False  # debugging option
                if unet_dim == 3:
                    input_ch = blksz_3d[2]
                zstride = input_ch // 2
                # create output stack (with augmented z dimension to align with stride size)

                volume_recon_ai = np.zeros([int(stride_3d[0] * np.ceil(volume1.shape[0] / stride_3d[0])),
                                            int(stride_3d[1] * np.ceil(volume1.shape[1] / stride_3d[1])),
                                            np.int(zstride * np.ceil(volume1.shape[2] / (zstride)))], dtype=np.float16)
                volume1_shape_orig = volume1.shape
                npadvoxs = tuple(np.subtract(volume_recon_ai.shape, volume1_shape_orig))
                volume1p = np.pad(volume1, ((0, npadvoxs[0]), (0, npadvoxs[1]), (0, npadvoxs[2])), 'edge')
                volume_s = np.zeros([volume1p.shape[0], volume1p.shape[1], input_ch], dtype=np.float16)
                print(volume_recon_ai.shape)
                print(volume1.shape)
                print(volume_s.shape)
                print(npadvoxs)
                blk_layers = np.int(volume_recon_ai.shape[2] / zstride)
                # if we've expanded the matrix volume_recon_ai w.r.t. volume1 subtract 1 from the number of recon blocks
                if volume_recon_ai.shape[2] != volume1p.shape[2]:
                    blk_layers -= 1

                slc_folding_total = volume_recon_ai.shape[2] - volume1p.shape[2]  # total additional slices we need to add to the stack to reconstruct on a block by block basis in the z direction
                slc_folding_top = np.int(slc_folding_total // 2)  # additional slices to add to the top    of the volume
                slc_folding_bottom = np.int(slc_folding_total - slc_folding_top)  # additional slices to add to the bottom of the volume
                maxzposprocessed_volume1 = 0

                for iBlks in range(blk_layers):  # loop through reconstruction layers in the z direction
                    print('#############################################')
                    print('recon layer', iBlks + 1, 'of', blk_layers)
                    blk_start = iBlks * zstride  # z (i.e. slice index) of current block being reconstructed
                    if iBlks == 0:  # for uppermost block
                        if slc_folding_top > 0:
                            volume_s[:, :, 0:slc_folding_top] = np.flip(volume1p[:, :, 0:slc_folding_top],2)  # flip upper most slices and place at top of the initial block
                        volume_s[:, :, slc_folding_top:] = volume1p[:, :, 0:blksz_3d[2] - slc_folding_top]  # place remaining true slices into the block
                        maxzposprocessed_volume1 = zstride - slc_folding_top
                        if debuglogic: print(' first blk', iBlks, 'maxzposprocessed_volume1', maxzposprocessed_volume1)
                    elif iBlks == blk_layers - 1:  # for lowermost block
                        volume_s[:, :, 0:input_ch - slc_folding_bottom] = volume1p[:, :,
                                                                          -(input_ch - slc_folding_bottom):]
                        if slc_folding_bottom > 0:
                            volume_s[:, :, input_ch - slc_folding_bottom:] = np.flip(
                                volume1p[:, :, -slc_folding_bottom:], 2)
                        if debuglogic: print(' last blk', iBlks, 'maxzposprocessed_volume1', volume1p.shape[2])
                    else:  # central/middle blocks
                        volume_s[:, :, :] = volume1p[:, :, maxzposprocessed_volume1:maxzposprocessed_volume1 + input_ch]
                        maxzposprocessed_volume1 = maxzposprocessed_volume1 + zstride
                        if debuglogic: print('central blk', iBlks, 'maxzposprocessed_volume1', maxzposprocessed_volume1)

                    # print('max, min before getBlock: ', np.amax(volume_s), np.amin(volume_s))
                    xtest = get_patches_2p5d(volume_s, blksz_3d[:2], stride_3d[:2])
                    # print('max, min after getBlock: ', np.amax(xtest), np.min(xtest))
                    xtest = np.expand_dims(xtest, axis=4)
                    if debuglogic:
                        newimage = np.copy(xtest)  # fast non gpu recon option for debugging code logic
                    else:
                        newimage = model.predict(xtest, batch_size=batch_size_recon, verbose=1)  # verbose=1
                    del xtest
                    # print('iBlks, amax(newimage), amin(newimage), newimage.shape: ', iBlks, np.amax(newimage), np.amin(newimage), newimage.shape)
                    imgshape = newimage.shape
                    if debuglogic:
                        dlimage = np.ones((volume1p.shape[0], volume1p.shape[1], input_ch)) * iBlks
                    else:
                        dlimage = np.zeros((volume1p.shape[0], volume1p.shape[1], input_ch))

                    t = time.time()
                    if debuglogic:
                        pass
                    else:
                        if parallel_recon:
                            # parallel (fast) computation
                            num_cores = multiprocessing.cpu_count() - 2
                            slice_positions = range(input_ch)
                            print('parallel recon slices', slice_positions, '...')
                            results = Parallel(n_jobs=num_cores)(
                                delayed(reconstruct_from_patches_2d_centerpixels)(newimage[:, :, :, iSlc, 0],
                                                                                  [volume1p.shape[0],
                                                                                   volume1p.shape[1]],
                                                                                  (stride_3d[0], stride_3d[1])) for iSlc
                                in slice_positions)
                            for i, value in enumerate(results):
                                dlimage[:, :, i] = np.copy(value)
                        else:
                            # sequential (slow) computation
                            for i in range(input_ch):
                                print('serial recon slice', i + 1, 'of', input_ch)
                                dlimage[:, :, i] = reconstruct_from_patches_2d_centerpixels(newimage[:, :, :, i, 0],
                                                                                            [volume1p.shape[0],
                                                                                             volume1p.shape[1]], (
                                                                                            stride_3d[0], stride_3d[1]))
                        del newimage
                    elapsed = time.time() - t
                    print('total time for tiling', input_ch, 'slices was', elapsed, 'seconds')

                    if input_ch % 2 == 0:
                        scale_per_slice = 1.0 / (zstride - 1)
                        odd_input_ch = False
                    else:
                        scale_per_slice = 1.0 / zstride
                        odd_input_ch = True

                    zoffset = iBlks * zstride
                    if iBlks == 0:  # uppermost block
                        for i in range(zstride):
                            volume_recon_ai[:, :, i] = dlimage[:, :,
                                                       i]  # place uppermost half of block into the final volume without weighting
                            if debuglogic: print('slice', i, 'scale', 1)
                        if odd_input_ch:
                            volume_recon_ai[:, :, zstride] = dlimage[:, :, zstride]
                            if debuglogic: print('slice', zstride, 'scale', 1)
                        for i in range(zstride):
                            volume_recon_ai[:, :, (input_ch - 1 - i)] += dlimage[:, :,
                                                                         input_ch - 1 - i] * i * scale_per_slice  # linearly weight with distance from block center
                            if debuglogic: print('slice', (input_ch - 1 - i), 'scale', i * scale_per_slice)
                    elif iBlks == blk_layers - 1:  # lowermost block
                        for i in range(zstride):
                            volume_recon_ai[:, :, zoffset + i] += dlimage[:, :, i] * i * scale_per_slice
                            if debuglogic: print('slice', zoffset + i, 'scale', i * scale_per_slice)
                        try:
                            if odd_input_ch:
                                volume_recon_ai[:, :, zoffset + zstride] += dlimage[:, :, zstride]
                                if debuglogic: print('slice', zoffset + zstride, 'scale', 1)
                        except:
                            pass
                        for i in range(zstride):
                            try:
                                volume_recon_ai[:, :, iBlks * zstride + (input_ch - 1 - i)] = dlimage[:, :,
                                                                                              input_ch - 1 - i]
                                if debuglogic: print('slice', zoffset + (input_ch - 1 - i), 'scale', 1)
                            except:
                                pass
                    else:  # central blocks
                        for i in range(zstride):
                            if (zoffset + i) < maxslc:
                                volume_recon_ai[:, :, zoffset + i] += dlimage[:, :, i] * i * scale_per_slice
                                if debuglogic: print('slice', zoffset + i, 'scale', i * scale_per_slice)
                        if odd_input_ch:
                            volume_recon_ai[:, :, zoffset + zstride] += dlimage[:, :, zstride]
                            if debuglogic: print('slice', zoffset + zstride, 'scale', 1)
                        for i in range(zstride):
                            if (zoffset + (input_ch - 1 - i)) < maxslc:
                                volume_recon_ai[:, :, zoffset + (input_ch - 1 - i)] += dlimage[:, :,
                                                                                       input_ch - 1 - i] * i * scale_per_slice
                                if debuglogic: print('slice', zoffset + (input_ch - 1 - i), i * scale_per_slice)

                if np.sum(npadvoxs) > 0:  # crop to original input volume size
                    if volume_recon_ai.shape[2] != volume1_shape_orig[2]:
                        volume_recon_ai = volume_recon_ai[:volume_recon_ai.shape[0] - npadvoxs[0],
                                          :volume_recon_ai.shape[1] - npadvoxs[1],
                                          :volume_recon_ai.shape[2] - npadvoxs[2]]
                    else:
                        volume_recon_ai = volume_recon_ai[:volume_recon_ai.shape[0] - npadvoxs[0],
                                          :volume_recon_ai.shape[1] - npadvoxs[1], :]

                # write reconstruction to multipage tiff stack
                volume_recon_ai = undo_reproject(volume_recon_ai, raw_projection)  # switch back to x,y,z order
                if unet_resnet_mode:
                    volume1 = undo_reproject(volume1, raw_projection)  # switch back to x,y,z order

                if unet_resnet_mode:  # if in unet resnet mode add output of network to lower quality input
                    print('data_airecon.shape, volume1.shape: ', volume_recon_ai.shape, volume1.shape)
                    volume_recon_ai = volume_recon_ai + volume1
                    volume_recon_ai[volume_recon_ai < 0] = 0  # replace negative values with zeros

                # save to disk
                ai_max = np.amax(volume_recon_ai)
                data_airecon = np.uint16(np.round(np.float(inputMax) / ai_max) * np.moveaxis(volume_recon_ai, -1,
                                                                                             0))  # move slices from 3rd dimension to 1st dimension
                tifffile.imsave(reconFileName_final, data_airecon, compress=6)

                del volume_recon_ai, volume1  # delete to save memory

if sleep_when_done:
    # from: https://stackoverflow.com/questions/37009777/how-to-make-a-windows-10-computer-go-to-sleep-with-a-python-script
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
