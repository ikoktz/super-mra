from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam

from utils import *

###############################################################################
# key parameters (begin)
###############################################################################
combomatrix = [[32, 32, 16, 16, 9, 10000, 10000, 8, False]]
''' in form [blksz_2d[0],           patch size in row direction (pixel units)
             blksz_2d[1],           patch size in column direction (pixel units)
             stride_2d[0],          stride of patch selection in row direction (pixel units)
             stride_2d[1],          stride of patch selection in column direction (pixel units)
             patch_select_mode,     patch selection mode (see compute_metric_volume_2d() function in utils.py)
             patches_per_set_h,     number of high signal/edge training patches per volume
             patches_per_set_l,     number of low signal training patches per volume
             unet_start_ch,         number of starting channels for the U-Net
             unet_resnet_mode]      residual learning mode for the U-Net to predict the difference; (default == False)
'''

# test mode options
testmode = False  # test by training/predicting the first case only for one reduction factor
testmode_epochs = False  # limits the number of epochs

# basic inputs/parameters
reduction_list = [3] if testmode else [2, 3, 4, 5, 6]  # resolution reduction factors to train/predict
proj_direction = 2  # projection direction for training; 0: no projection, 1: lateral projection, 2: frontal projection
loss_function = 'mean_squared_error'  # 'ssim_loss'
sleep_when_done = False  # sleep computer when finished
patches_from_volume = True  # False: patches selected from each slice; True: patches selected from whole volume
data_augm_factor = 1  # data augmentation factor; arbitrary values allowed for 2D
optimizers = ['adam']  # ['adam', 'sgd']
leave_one_out_train = True  # performs training using a leave one out scheme
###############################################################################
# key parameters (end)
###############################################################################

for iRed in reduction_list:  # loop over resolution reduction factors
    reduction = str(iRed) + 'fold'
    for b_index in combomatrix:  # loop over reconstructions to perform
        patch_select_mode = b_index[4]  # set patch selection mode
        try:
            patches_per_set_h = b_index[5]  # set number of high intensity patches selected from each dataset
            patches_per_set_l = b_index[6]  # set number of random intensity patches selected from each dataset
        except:
            patches_per_set_h = 5000  # set number of high intensity patches selected from each dataset
            patches_per_set_l = 5000  # set number of random intensity patches selected from each dataset
        try:
            unet_resnet_mode = b_index[8]
        except:
            unet_resnet_mode = False

        # construct folder name where models are                
        if loss_function == "mean_squared_error":
            foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(proj_direction) + 'psm' + str(
                patch_select_mode) + "_" + "mse"
        else:
            foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(proj_direction) + 'psm' + str(
                patch_select_mode) + "_" + loss_function

        for optim in optimizers:  # loop through optimizers
            nepochs = 2 if testmode_epochs else 200  # number of epochs to train for
            batch_size_train = 400  # training batch size

            blksz_2d = b_index[0], b_index[1]   # block/patch size in pixels
            stride_2d = b_index[2], b_index[3]  # stride for obtaining training blocks

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

            n_edge_after_proj = 0  # the edge images not used after proj_direction
            patches_per_set = patches_per_set_h + patches_per_set_l

            blks_rand_shift_mode = False
            training_patience = 3
            n_slices_exclude = 0  # number of edge slices to not train from (default value is zero)

            subset_train_mode = False  # flag that allows to only to train from a subset of slices from the full volume
            subset_train_minslc = 1  # training subset mode - minimum slice
            subset_train_maxslc = 500  # training subset mode - maximum slice

            # factor for additional in-plane (i.e. x and y) cropping of volumes during training phase
            crop_train_x = 0.50 if patches_from_volume else 1
            crop_train_y = 0.50 if patches_from_volume else 0.6

            ###############################################################################
            # find distinct data sets
            ###############################################################################
            script_path = os.path.split(os.path.abspath(__file__))[0]
            dirsource = os.path.join(script_path, 'input_low_res_' + reduction)
            dirtarget = os.path.join(script_path, 'input_high_res')
            try:
                srcfiles, tgtfiles = get_source_and_target_files(dirsource, dirtarget)
            except:
                print('could not find', reduction, 'source or target files so skipping')
                continue

            ###############################################################################
            # load .tiff files from disk and count number of total slices
            ###############################################################################
            totalnumberofslices = 0  # count of slices for all data sets
            slices_in_files = dict()  # to hold slice number for training from each file
            for m, icode in enumerate(srcfiles):
                print('counting slices for ' + icode)
                nslices = count_tiff_slices(os.path.join(dirsource, icode), subset_train_mode, subset_train_minslc,
                                            subset_train_maxslc, proj_direction)
                if proj_direction == 0:
                    slices_in_files[icode] = nslices - 2 * n_slices_exclude
                else:
                    slices_in_files[icode] = nslices - 2 * n_edge_after_proj
                # update total count of slices
                totalnumberofslices += slices_in_files[icode]

            ###############################################################################
            # check if we've already trained the model; if no, train it, if yes, load it from disk
            ###############################################################################
            rstr = "res" if unet_resnet_mode else ""
            p = patches_per_set, patches_per_set_h, patches_per_set_l if patches_from_volume else patches_per_slc, patches_per_slc_h, patches_per_slc_l

            # 2D UNet
            modelsuffix = "_unet2d" + rstr + "-" + "[" + str(stride_2d[0]) + 'x' + str(
                stride_2d[1]) + "]-psm" + str(patch_select_mode) + "-" + str(unet_start_ch) + "-" + str(
                unet_depth) + "-" + str(unet_inc_rate) + "-" + str(
                unet_dropout) + "-" + unet_batchnorm_str + "-" + unet_residual_str + '-batch' + str(
                batch_size_train)
            modelprefix = "model_" + str(blksz_2d[0]) + "x" + str(blksz_2d[1]) + "x" + str(p[0]) + "(" + str(
                p[1]) + ")(" + str(p[2]) + ")x" + str(data_augm_factor)

            # check if we need to train more models and set the training_needed_flag,
            # as well as return the list for leave one out training mode
            outpath = 'train_' + 'unet2d' + rstr + '_' + optim + '_' + reduction + '_batch' + str(
                batch_size_train) + foldersuffix

            script_path = os.path.split(os.path.abspath(__file__))[0]
            dirmodel = os.path.join(script_path, outpath)
            if not os.path.exists(dirmodel):
                os.makedirs(dirmodel)

            training_needed_flag, indices_of_datasets_to_train = should_we_train_network(
                os.path.join(dirmodel, modelprefix + modelsuffix), leave_one_out_train, srcfiles)

            if training_needed_flag:
                print("model not found for sets", " ".join(str(indices_of_datasets_to_train)), ", so must train it")
                ###############################################################################
                # create training data if not already created
                ###############################################################################
                suffix_npy = "_unet2d" + rstr + "_" + str(blksz_2d[0]) + 'x' + str(blksz_2d[1]) + 'x' + str(
                    p[0]) + "(" + str(p[1]) + ")(" + str(p[2]) + ")" + "_[" + str(stride_2d[0]) + 'x' + str(
                    stride_2d[1]) + ']' + "_proj" + str(proj_direction) + "_" + reduction + "_psm" + str(
                    patch_select_mode)

                if not os.path.exists(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy')):
                    print("training data not found, so must create it")
                    ###############################################################################
                    # create training arrays
                    ###############################################################################
                    if patches_from_volume:  # select patches from a complete set of data, not slice by slice
                        totalpatches = len(srcfiles) * patches_per_set
                    else: # select patches from each slice
                        totalpatches = totalnumberofslices * patches_per_slc

                    xtrain_master_noaug = np.zeros([totalpatches, blksz_2d[0], blksz_2d[1], 1], dtype=np.float16)
                    ytrain_master_noaug = np.zeros([totalpatches, blksz_2d[0], blksz_2d[1], 1], dtype=np.float16)

                    # count the number of total slices to learn from during training phase
                    # (i.e. loop through all data sets except the one that is being reconstructed)
                    slice_count = 0  # counter holding number of slices incorporated into training
                    ###############################################################################
                    # load .npy files from disk, display central coronal slice, and fill xtrain and ytrain matrices
                    ###############################################################################
                    slices_per_file = []  # recording the number of slices used for training data extractions
                    if proj_direction > 0:
                        n_slices_exclude = n_edge_after_proj

                    for m, icode in enumerate(srcfiles):  # loop over volumes to train from
                        print('##############################################')
                        print('low resolution volume is =>', icode)
                        ######################################
                        # load numpy arrays, reproject dataset (if needed) and trim and normalize dataset,
                        ###################################
                        volume1, volume1max = load_tiff_volume_and_scale_si(dirsource, icode, crop_train_x,
                                                                            crop_train_y, blksz_2d, proj_direction,
                                                                            subset_train_mode, subset_train_minslc,
                                                                            subset_train_maxslc)

                        print('high resolution volume is =>', tgtfiles[m])
                        volume3, volume3max = load_tiff_volume_and_scale_si(dirtarget, tgtfiles[m], crop_train_x,
                                                                            crop_train_y, blksz_2d, proj_direction,
                                                                            subset_train_mode, subset_train_minslc,
                                                                            subset_train_maxslc)

                        print('creating training data set...')
                        if patches_from_volume:  # select patches from each whole dataset
                            # xtrain, patches1, volume1 are source # ytrain, patches3, volume3 are target
                            # create metric volume used to select blocks
                            vol_metric, metric_operator = compute_metric_volume_2d(volume1, volume3,
                                                                                   patch_select_mode, stride_2d,
                                                                                   n_slices_exclude)
                            slc_train_end = volume1.shape[2] - n_slices_exclude
                            if not unet_resnet_mode:
                                xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                                0], ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                                    0] = get_patches_within_volume(vol_metric,
                                                                   [volume1[:, :, n_slices_exclude:slc_train_end],
                                                                    volume3[:, :, n_slices_exclude:slc_train_end]],
                                                                   blksz_2d, stride_2d, patches_per_set_h,
                                                                   patches_per_set_l, seed=m, shuffleP=False,
                                                                   metric_operator=metric_operator)
                            else:
                                xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                                0], ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :,
                                    0] = get_patches_within_volume(vol_metric, [
                                    volume1[:, :, n_slices_exclude:slc_train_end],
                                    volume3[:, :, n_slices_exclude:slc_train_end] - volume1[:, :,
                                                                                    n_slices_exclude:slc_train_end]],
                                                                   blksz_2d,
                                                                   stride_2d,
                                                                   patches_per_set_h,
                                                                   patches_per_set_l, seed=m,
                                                                   shuffleP=False,
                                                                   metric_operator=metric_operator)

                            slice_count = slice_count + volume3.shape[2] - 2 * n_slices_exclude
                            slices_per_file.append(volume3.shape[2] - 2 * n_slices_exclude)
                            print('ytrain_master_noaug.mean: ', np.mean(ytrain_master_noaug))

                        else:  # select patches from each slice
                            for iSlc in range(n_slices_exclude, volume1.shape[2] - n_slices_exclude, 10):
                                if iSlc == n_slices_exclude:
                                    print('training from set', icode, 'slice', iSlc, end="")
                                else:
                                    print(',', iSlc, end="")

                                # 2D Unet
                                tmp = get_local_max_patches_from_image_unaugmented(np.squeeze(volume3[:, :, iSlc]),
                                                                                   np.squeeze(volume1[:, :, iSlc]),
                                                                                   blksz_2d, patches_per_slc_h,
                                                                                   patches_per_slc_l,
                                                                                   patch_select_mode)
                                patches3 = np.squeeze(tmp[:, :, :, 0])
                                patches1 = np.squeeze(tmp[:, :, :, 1])

                                xtrain_master_noaug[
                                slice_count * patches_per_slc:(slice_count + 1) * patches_per_slc, :, :,
                                0] = patches1[:, :, :]  # xtrain, patches1, volume1 are source
                                if not unet_resnet_mode:
                                    ytrain_master_noaug[
                                    slice_count * patches_per_slc:(slice_count + 1) * patches_per_slc, :, :,
                                    0] = patches3[:, :, :]  # ytrain, patches3, volume3 are target
                                else:
                                    ytrain_master_noaug[
                                    slice_count * patches_per_slc:(slice_count + 1) * patches_per_slc, :, :,
                                    0] = patches3[:, :, :] - patches1[:, :,
                                                             :]  # ytrain, patches3-patches1, volume3 are target (residual unet mode)

                                slice_count = slice_count + 1

                            slices_per_file.append(iSlc + 1)

                        if m == (len(srcfiles) - 1):  # if last volume, save the training data to disk
                            np.save(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy), xtrain_master_noaug)
                            np.save(os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy), ytrain_master_noaug)

                    print('total files read for training: ', len(srcfiles))
                    print('total slices read for training: ', slice_count)
                    print('unaugmented patch array size for training: ', xtrain_master_noaug.shape)
                    print('slices (or blks) read from all the files: ', slices_per_file)
                ###############################################################################
                # load training data from disk if not already created
                ###############################################################################
                else:
                    print("training data found, so just load it")
                    # load numpy arrays
                    xtrain_master_noaug = np.load(
                        os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy'))
                    ytrain_master_noaug = np.load(
                        os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy + '.npy'))
                    print('loaded xtrain_master_noaug' + suffix_npy + '.npy' + ' shape: ', xtrain_master_noaug.shape)
                    print('loaded ytrain_master_noaug' + suffix_npy + '.npy' + ' shape: ', ytrain_master_noaug.shape)
                print(" ")

                ###############################################################################
                # augment training data by factor of data_augm_factor
                ###############################################################################
                if data_augm_factor > 1:
                    print("augmenting data by factor of", data_augm_factor, "...")
                    iSlc = 0
                    xtrain_master = augment_patches(xtrain_master_noaug, data_augm_factor, iSlc, 180,
                                                        (blksz_2d[0] // 4, blksz_2d[1] // 4), 0.4)
                    ytrain_master = augment_patches(ytrain_master_noaug, data_augm_factor, iSlc, 180,
                                                        (blksz_2d[0] // 4, blksz_2d[1] // 4), 0.4)
                    print("augmenting data by factor of", data_augm_factor, "... done")
                else:
                    # don't data augment
                    ytrain_master = np.copy(ytrain_master_noaug)
                    xtrain_master = np.copy(xtrain_master_noaug)
                    ytrain = ytrain_master_noaug
                    del ytrain_master_noaug
                    xtrain = xtrain_master_noaug
                    del xtrain_master_noaug

                ###############################################################################
                # define model and print summary
                ###############################################################################
                if leave_one_out_train:
                    n_loo_loops = len(srcfiles)  # a network is trained for each data set
                else:
                    n_loo_loops = 1  # only one network is trained for all data sets

                # loop through "leave one out" iterations (i.e. distinct data sets)
                for set_to_train in indices_of_datasets_to_train:
                    if testmode:  # only do first set if we are prototyping
                        if set_to_train > 0:
                            continue

                    ###################################################
                    # cut out some training data from data sets to be predicted if we are in leave one out mode
                    ###################################################
                    inds_all = np.arange(0, xtrain_master.shape[0])
                    if leave_one_out_train:
                        print('leave one out training mode, leaving out data set', str(set_to_train + 1), '=>',
                              srcfiles[set_to_train])
                        ntrainblockspervolume = int(xtrain_master.shape[0] / len(srcfiles))
                        inds_to_remove = np.arange(set_to_train * ntrainblockspervolume,
                                                   (set_to_train + 1) * ntrainblockspervolume)
                        inds_to_train_from = inds_all[~np.in1d(inds_all, inds_to_remove).reshape(inds_all.shape)]
                    else:
                        print('using all data to train network')
                        inds_to_train_from = np.copy(inds_all)
                    xtrain = xtrain_master[inds_to_train_from, :, :, :]
                    ytrain = ytrain_master[inds_to_train_from, :, :, :]

                    ###############################################################################
                    # define model and print summary
                    ###############################################################################
                    try:
                        del model
                    except:
                        pass
                    model = unet2d((blksz_2d[0], blksz_2d[1], 1), out_ch=1, start_ch=unet_start_ch,
                                   depth=unet_depth, inc_rate=unet_inc_rate, activation="relu", dropout=unet_dropout,
                                   batchnorm=unet_batchnorm, maxpool=True, upconv=False, residual=unet_residual)
                    print(model.summary())

                    ###############################################################################
                    # compile the model
                    ###############################################################################
                    if optim == 'adam':
                        opt = Adam()
                    else:
                        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
                    if loss_function != 'ssim_loss':
                        model.compile(loss=loss_function, optimizer=opt)
                    else:
                        model.compile(loss=ssim_loss, optimizer=opt, metrics=[ssim_loss, 'accuracy'])

                    ###############################################################################
                    # checkpointing the model
                    ###############################################################################
                    if leave_one_out_train:
                        filepath = os.path.join(dirmodel,
                                                modelprefix + modelsuffix + "-{epoch:02d}-{loss:.6f}-{val_loss:.6f}-set" + str(
                                                    int(set_to_train + 1)) + ".hdf5")
                    else:
                        filepath = os.path.join(dirmodel,
                                                modelprefix + modelsuffix + "-{epoch:02d}-{loss:.6f}-{val_loss:.6f}.hdf5")
                    checkpoint1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                  mode='min')
                    checkpoint2 = EarlyStopping(monitor='val_loss', patience=training_patience, verbose=1, mode='min')
                    callbacks_list = [checkpoint1, checkpoint2]

                    ###############################################################################
                    # fit the model
                    ###############################################################################
                    print('xtrain size: ', xtrain.shape)
                    print('ytrain size: ', ytrain.shape)
                    with tf.device('/gpu:0'):
                        history = model.fit(xtrain, ytrain, batch_size=max(data_augm_factor, batch_size_train),
                                            epochs=nepochs, callbacks=callbacks_list, validation_split=0.2,
                                            shuffle=True)
                        print(history.history.keys())
                        print("loss: ", history.history['loss'])
                        print("val_loss ", history.history['val_loss'])

                    ##################################################################
                    # find the the number of the last training epoch written to disk
                    ##################################################################
                    found = False
                    lastepochtofind = len(history.history['loss'])
                    print('looking for a file that starts with',
                          modelprefix + modelsuffix + "-" + "{0:0=2d}".format(lastepochtofind), 'and ends with .hdf5')
                    while not found and lastepochtofind > 0:
                        for root, _, files in os.walk(dirmodel):
                            for f in files:
                                if f.startswith(modelprefix + modelsuffix + "-" + "{0:0=2d}".format(
                                        lastepochtofind)) > 0 and f.endswith('.hdf5'):
                                    found = True
                                    print("last epoch was " + str(lastepochtofind))
                                    break
                        if not found:
                            lastepochtofind -= 1  # reduce epoch number to try to find .hdf5 file if not already found
                    if not found:
                        print("failed to find most recent good training epoch... this shouldn't happen")
                        sys.exit()

                    ###############################################################################
                    # serialize model to json and write to disk
                    ###############################################################################
                    model_json = model.to_json()
                    with open(os.path.join(dirmodel,
                                           modelprefix + modelsuffix + "_set" + str((set_to_train + 1)) + ".json"),
                              "w") as json_file:
                        json_file.write(model_json)
                    ###############################################################################
                    # save weights and write to disk as .h5 file
                    ###############################################################################
                    model.save_weights(
                        os.path.join(dirmodel, modelprefix + modelsuffix + "_set" + str((set_to_train + 1)) + ".h5"))
                    print("saved model to disk")
            else:
                pass  # network already trained so do nothing

if sleep_when_done:
    # from: https://stackoverflow.com/questions/37009777/how-to-make-a-windows-10-computer-go-to-sleep-with-a-python-script
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
