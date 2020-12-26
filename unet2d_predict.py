import tifffile
from keras.models import model_from_json

from utils import *

###############################################################################
# key parameters (begin)
###############################################################################
combomatrix = [[32, 32, 16, 16, 9, 10000, 10000, 8, False]]
''' in form [blksz_2d[0],           patch size in row direction (pixel units)
             blksz_2d[1],           patch size in column direction (pixel units)
             stride_2d[0],          stride of patch reconstruction in row direction (pixel units)
             stride_2d[1],          stride of patch reconstruction in column direction (pixel units)
             patch_select_mode,     patch selection mode (see compute_metric_volume_2d() function in utils.py)
             patches_per_set_h,     number of high signal/edge training patches per volume
             patches_per_set_l,     number of low signal training patches per volume
             unet_start_ch,         number of starting channels for the U-Net
             unet_resnet_mode]      residual learning mode for the U-Net to predict the difference; (default == False)
'''

# test mode options
testmode = False  # test by training/predicting the first case only for one reduction factor

reduction_list = [3] if testmode else [2, 3, 4, 5, 6]  # resolution reduction factors to train/predict
loss_function = 'mean_squared_error'  # 'mean_squared_error' 'ssim_loss'
sleep_when_done = False  # sleep computer when finished
proj_direction = 2  # projection direction for training; 0: no projection, 1: lateral projection, 2: frontal projection
data_augm_factor = 1 # data augmentation factor
patches_from_volume = True  # 0: patches selected from each slice; 1: patches selected from each whole dataset
optimizers = ['adam']  # ['adam', 'sgd']
leave_one_out_train = True  # performs training using a leave one out scheme
###############################################################################
# key parameters (end)
###############################################################################

for iRed in reduction_list:  # loop over resolution reduction factors
    reduction = str(iRed) + 'fold'
    for b_index in combomatrix:  # loop over reconstructions to perform
        print("#################")
        print(b_index)
        print("#################")
        try:
            unet_resnet_mode = b_index[8]  # Unet residual mode; Unet tries to predict the difference image/volume
                                           # that will be added back to the low res data to generate the high res data
        except:
            unet_resnet_mode = False
        try:
            patches_per_set_h = b_index[5]  # set number of high intensity patches selected from each dataset
            patches_per_set_l = b_index[6]  # set number of random intensity patches selected from each dataset
        except:
            patches_per_set_h = 5000  # set number of high intensity patches selected from each dataset
            patches_per_set_l = 5000  # set number of random intensity patches selected from each dataset
        patches_per_set = patches_per_set_h + patches_per_set_l
        # number of total training blocks, high sig blocks, random blocks used for training
        npatches = patches_per_set, patches_per_set_h, patches_per_set_l
        patch_select_mode = b_index[4]  # patch selection mode

        for optim in optimizers:  # loop through optimizers
            batch_size_train = 400  # training batch size
            batch_size_recon = 1000  # recon batch size

            blksz_2d = b_index[0], b_index[1]  # reconstruction block/patch size in pixels
            stride_2d = b_index[2], b_index[3]  # reconstruction stride in pixels

            parallel_recon = True

            patches_per_slc_h = 60  # high signal intensity/arterial blocks/patches per slice used for training
            patches_per_slc_l = 80  # low signal intensity/background blocks/patches per slice used for training
            patches_per_slc = patches_per_slc_h + patches_per_slc_l

            # U-net related inputs/parameters
            try:
                unet_start_ch = b_index[7]
            except:
                unet_start_ch = 16  # number of starting channels
            unet_depth = 4 if b_index[0] ** 0.25 >= 2 else 3  # depth (i.e. # of max pooling steps)
            unet_inc_rate = 2  # channel increase rate
            unet_dropout = 0.5  # dropout rate
            unet_batchnorm = False  # batch normalization mode
            unet_residual = False  # residual connections mode
            unet_batchnorm_str = 'F' if not unet_batchnorm else 'T'
            unet_residual_str = 'F' if not unet_residual else 'T'

            subset_recon_mode = False  # flag the allows to recon only a subset of slices
            subset_recon_minslc = 1  # min slice to recon
            subset_recon_maxslc = 100  # max slice to recon

            # factors for additional in-plane (i.e. x and y) cropping of volumes during recon/test/prediction phase
            crop_recon_x = 1.0 if not testmode else 0.5
            crop_recon_y = 1.0 if not testmode else 0.5

            batch_size = 1000

            blks_rand_shift_mode = False  # random block shift mode

            ###############################################################################
            # check if we've already trained the model; if no, train it, if yes, load it from disk
            ###############################################################################
            rstr = "res" if unet_resnet_mode else ""
            p = patches_per_set, patches_per_set_h, patches_per_set_l if patches_from_volume else patches_per_slc, patches_per_slc_h, patches_per_slc_l
            modelsuffix = "_unet2d" + rstr + "-" + "[" + str(stride_2d[0]) + 'x' + str(
                stride_2d[1]) + "]-psm" + str(patch_select_mode) + "-" + str(unet_start_ch) + "-" + str(
                unet_depth) + "-" + str(unet_inc_rate) + "-" + str(
                unet_dropout) + "-" + unet_batchnorm_str + "-" + unet_residual_str + '-batch' + str(
                batch_size_train)
            modelprefix = "model_" + str(blksz_2d[0]) + "x" + str(blksz_2d[1]) + "x" + str(p[0]) + "(" + str(
                p[1]) + ")(" + str(p[2]) + ")x" + str(data_augm_factor)

            # construct folder name where models are
            if loss_function == "mean_squared_error":
                foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(proj_direction) + 'psm' + str(
                    patch_select_mode) + "_" + "mse"
            else:
                foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(proj_direction) + 'psm' + str(
                    patch_select_mode) + "_" + loss_function

            outpath = 'train_' + 'unet2d' + rstr + '_' + optim + '_' + reduction + '_batch' + str(
                batch_size_train) + foldersuffix

            suffix_npy = "_unet2d" + rstr + "_" + str(blksz_2d[0]) + 'x' + str(blksz_2d[1]) + 'x' + str(
                p[0]) + "(" + str(p[1]) + ")(" + str(p[2]) + ")" + "_[" + str(stride_2d[0]) + 'x' + str(
                stride_2d[1]) + ']' + "_proj" + str(proj_direction) + "_" + reduction + "_psm" + str(
                patch_select_mode)

            ###############################################################################
            # load model architecture and weights from .json and .hdf5 files on disk
            ###############################################################################
            script_path = os.path.split(os.path.abspath(__file__))[0]
            dirmodel = os.path.join(script_path, outpath)
            if not os.path.exists(dirmodel):
                sys.exit("error - ", dirmodel, "doesn't exist, so can't predict")
            dirinput = os.path.join(script_path, 'input_low_res_' + reduction)

            model_to_apply = modelprefix

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
                extraconfigstring = str(unet_start_ch) + "-" + str(unet_depth) + "-" + str(unet_inc_rate) + "-" + str(
                    unet_dropout) + "-" + unet_batchnorm_str + "-" + unet_residual_str
                modelFileName, jsonFileName = get_model_and_json_files(dirmodel, model_to_apply, blks_rand_shift_mode,
                                                                       leave_one_out_train, datasetnumber, stride_2d,
                                                                       extraconfigstring)
                if len(modelFileName) == 0:
                    print('could not find model for', inputfiles, 'so skip')
                    continue

                print('json file  =>', jsonFileName)
                print('model file =>', modelFileName)
                print('load json:    ', os.path.join(dirmodel, jsonFileName))
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

                fname = inputTifs.split('.')[0].split('\\')[-1] + '_' + reconFileNameSuffix
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
                volume1 = load_tiff_to_numpy_vol(inputTifs, subset_recon_mode, subset_recon_minslc, subset_recon_maxslc)
                # adjust size to reduce computation load
                # adjust x and y dimensions of volume to divide evenly into blksz_2d
                volume1 = crop_volume_in_xy_and_reproject(volume1, crop_recon_x, crop_recon_y, blksz_2d, proj_direction)
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

                # loop over slices to reconstruct
                # create ai volume
                volume_recon_ai = np.zeros([int(stride_2d[0] * np.ceil(volume1.shape[0] / stride_2d[0])),
                                            int(stride_2d[1] * np.ceil(volume1.shape[1] / stride_2d[1])),
                                            volume1.shape[2]], dtype=np.float16)
                volume1_shape_orig = volume1.shape
                npadvoxs = tuple(np.subtract(volume_recon_ai.shape, volume1_shape_orig))
                # print(npadvoxs,npadvoxs[0],npadvoxs[1],npadvoxs[2])
                volume1p = np.pad(volume1, ((0, npadvoxs[0]), (0, npadvoxs[1]), (0, npadvoxs[2])), 'edge')
                volume_s = np.zeros([volume1p.shape[0], volume1p.shape[1], 1], dtype=np.float16)
                print('volume_s.shape', volume_s.shape)
                for iSlc in range(minslc, maxslc):

                    print('reconstructing slice', iSlc + 1, 'of', maxslc)

                    # get patches for target slice that will be predicted
                    patches1 = get_patches(np.squeeze(volume1p[:, :, iSlc]), blksz_2d, stride_2d)
                    # create arrays to store validation set
                    xtest = np.zeros([patches1.shape[0], blksz_2d[0], blksz_2d[1], 1], dtype=np.float16)
                    # save patches to test data set
                    xtest[:, :, :, 0] = patches1[:, :, :]
                    # delete some variables to save memory
                    del patches1

                    ###############################################################################
                    # predict using the model
                    ###############################################################################
                    newimage = model.predict(xtest, batch_size=batch_size_recon, verbose=1)

                    ###############################################################################
                    # reconstruct the deep learned image
                    ###############################################################################
                    dlimage = reconstruct_from_patches_2d_centerpixels(newimage, volume1p.shape[0:2], stride_2d)

                    # ensure we aren't saturating the float data type
                    inds_sigtoolarge = np.where(dlimage > 1)
                    dlimage[inds_sigtoolarge] = 1

                    ###############################################################################
                    # save deep learned slices to separate output volume
                    ###############################################################################
                    volume_recon_ai[:, :, iSlc] = np.float16(dlimage)

                    # delete some variables to save memory
                    del xtest

                # crop generated volume to original size of input volume
                print(volume_recon_ai.shape, 'before cropping')
                volume_recon_ai = volume_recon_ai[:volume_recon_ai.shape[0] - npadvoxs[0],
                                  :volume_recon_ai.shape[1] - npadvoxs[1], :volume_recon_ai.shape[2] - npadvoxs[2]]
                print(volume_recon_ai.shape, 'after cropping')

                # write reconstruction to multipage tiff stack
                volume_recon_ai = undo_reproject(volume_recon_ai, proj_direction)  # switch back to x,y,z order
                if unet_resnet_mode:
                    volume1 = undo_reproject(volume1, proj_direction)  # switch back to x,y,z order

                if unet_resnet_mode:  # if in unet resnet mode add output of network to lower quality input
                    print('data_airecon.shape, volume1.shape: ', volume_recon_ai.shape, volume1.shape)
                    volume_recon_ai = volume_recon_ai + volume1
                    volume_recon_ai[volume_recon_ai < 0] = 0  # replace negative values with zeros

                # save to disk
                ai_max = np.amax(volume_recon_ai)
                data_airecon = np.uint16(np.round(np.float(inputMax) / ai_max) * np.moveaxis(volume_recon_ai, -1,
                                                                                             0))  # move slices from 3rd dimension to 1st dimension
                tifffile.imsave(reconFileName, data_airecon, compress=6)

                del volume_recon_ai, volume1  # delete to save memory

if sleep_when_done:
    # from: https://stackoverflow.com/questions/37009777/how-to-make-a-windows-10-computer-go-to-sleep-with-a-python-script
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
