import tifffile
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam

from models import *
from utils import *

###############################################################################
# key parameters (begin)
###############################################################################
combomatrix = [[32, 32, 32, 16, 16, 16, 64, 1, 9, 500, 500, False, 'mean_squared_error']]
''' in form [blksz_3d[0],           block size in row direction (pixel units)
             blksz_3d[1],           block size in column direction (pixel units)
             blksz_3d[2],           block size in slice direction (pixel units)
             stride_3d[0],          stride of block selection in row direction (pixel units)
             stride_3d[1],          stride of block selection in column direction (pixel units)
             stride_3d[2],          stride of block selection in slice direction (pixel units)
             resnet_filters,        number of convolution filters
             data_augm_factor,      data augmentation factor
             patch_select_mode,     block selection mode (see compute_metric_volume_3d() function in utils.py)
             patches_per_set_h,     number of high signal/edge training blocks per volume
             patches_per_set_l,     number of low signal training blocks per volume
             filter_mode_1D,        1d filter mode in x & y directions (default = False)
             loss_function]         loss function used for training 
'''

# test mode options
testmode = False  # test by training/predicting the first case only for one reduction factor
testmode_epochs = False  # limits the number of epochs

# basic inputs/parameters
reduction_list = [3] if testmode else [2, 3, 4, 5, 6]  # resolution reduction factors to train/predict
raw_projection = 0  # projection direction for training; 0: no projection, 1: lateral projection, 2: frontal projection
sleep_when_done = False  # sleep computer when finished
resnet_mode = True  # serial convolutions + residual connection mode
resnet_cnn_depth = [7]  # number of convolutional blocks
optimizers = ['adam']  # ['adam', 'sgd']
leave_one_out_train = True  # performs training using a leave one out scheme
###############################################################################
# key parameters (end)
###############################################################################

for iRed in reduction_list:  # loop over resolution reduction factors
    reduction = str(iRed) + 'fold'
    print(("########## training ########## ", reduction, " ###################################\n") * 10)

    for b_index in combomatrix:  # loop over reconstructions to perform
        try:
            loss_function = b_index[12]
        except:
            loss_function = 'mean_squared_error'
        try:
            patches_per_set_h = b_index[9]
            patches_per_set_l = b_index[10]
        except:
            patches_per_set_h = 500
            patches_per_set_l = 500
        patches_per_set = patches_per_set_h + patches_per_set_l
        try:
            filter_mode_1d = b_index[11]
        except:
            filter_mode_1d = False
        try:
            blksz_3d = b_index[0], b_index[1], b_index[2]  # patch/block sizes that are used to train the model
        except:
            print('could not define blksz_tuple, so quit out!')
            quit()
        try:  # custom training stride configuration
            stride_3d = b_index[3], b_index[4], b_index[5]
        except:
            stride_3d = b_index[0] // 2, b_index[1] // 2, b_index[2] // 2
        try:
            resnet_filters = b_index[6]
        except:
            resnet_filters = 64
        try:
            data_augm_factor = b_index[7]
        except:
            data_augm_factor = 1
        try:
            patch_select_mode = b_index[8]
        except:
            patch_select_mode = 1

        for curr_depth in resnet_cnn_depth:  # loop through depths of network
            print(("###################### depth = ", curr_depth, " #####################\n") * 5)
            for optim in optimizers:  # loop through optimizers
                nepochs = 2 if testmode_epochs else 200  # number of epochs to train for
                batch_size_train = 20  # training batch size

                data_augm_factor = set_augmentation_factor_for_3D_net(
                    data_augm_factor)  # set augmentation factor to 1,2,4 or 8 based on initial value

                resnet_dropout = 0
                n_edge_after_proj = 0  # the edge images not used after raw_projection
                local_max_dif_mod = 1  # 0: local_max on target image, 1: local_max on weighted target-source difference image with gaussian_filter
                patches_from_volume = True  # False: patches selected from each slice; True: patches selected from whole volume

                blks_rand_shift_mode = False
                training_patience = 3
                n_slices_exclude = 0  # number of edge slices to not train from (default value is zero)

                subset_train_mode = False  # flag that allows to only to train from a subset of slices from the full volume
                subset_train_minslc = 1  # training subset mode - minimum slice
                subset_train_maxslc = 500  # training subset mode - maximum slice

                crop_train_x = 0.50 if patches_from_volume else 1  # factor for additional in-plane (i.e. x and y) cropping of volumes during training phase
                crop_train_y = 0.50 if patches_from_volume else 0.6  # was 0.5, factor for additional in-plane (i.e. x and y) cropping of volumes during training phase

                if loss_function == 'mean_squared_error':
                    foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(raw_projection) + 'psm' + str(
                        patch_select_mode) + "_" + "mse"
                else:
                    foldersuffix = '_' + str(data_augm_factor) + 'aug_proj' + str(raw_projection) + 'psm' + str(
                        patch_select_mode) + "_" + loss_function

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
                # check if we've already trained the model; if no, train it, if yes, load it from disk
                ###############################################################################    
                filterstring = "1Dfilter" if filter_mode_1d else ""
                modelsuffix = "_scrc3d" + filterstring + "-" + "[" + str(stride_3d[0]) + 'x' + str(
                    stride_3d[1]) + 'x' + str(stride_3d[2]) + ']-psm' + str(patch_select_mode) + '-batch' + str(
                    batch_size_train)
                modelprefix = "model_" + str(blksz_3d[0]) + "x" + str(blksz_3d[1]) + "x" + str(blksz_3d[2]) + "x" + str(
                    patches_per_set) + "(" + str(patches_per_set_h) + ")(" + str(patches_per_set_l) + ")x" + str(
                    data_augm_factor)
                if blks_rand_shift_mode:  # if randomly shifting blocks during training process
                    modelsuffix += '_rsb'

                # check if we need to train more models and set the training_needed_flag, as well as return the list
                # for leave one out training mode
                filterstring = "1Dfilter" if filter_mode_1d else ""
                outpath = 'train_' + 'scrc3d' + filterstring + '_' + optim + '_' + str(
                    curr_depth) + '_' + reduction + '_' + str(resnet_filters) + 'filters_batch' + str(
                    batch_size_train) + foldersuffix

                script_path = os.path.split(os.path.abspath(__file__))[0]
                dirmodel = os.path.join(script_path, outpath)
                if not os.path.exists(dirmodel):
                    os.makedirs(dirmodel)

                print('dirmodel    =>', dirmodel)
                print('modelprefix =>', modelprefix)
                print('modelsuffix =>', modelsuffix)
                training_needed_flag, indices_of_datasets_to_train = should_we_train_network(
                    os.path.join(dirmodel, modelprefix + modelsuffix), leave_one_out_train, srcfiles)

                if training_needed_flag:
                    print("model not found for sets", " ".join(str(indices_of_datasets_to_train)), ", so must train it")
                    ###############################################################################
                    # create training data if not already created
                    ###############################################################################
                    suffix_npy = "_scrc3d_" + str(blksz_3d[0]) + "x" + str(blksz_3d[1]) + "x" + str(
                        blksz_3d[2]) + "_" + str(patches_per_set) + "(" + str(patches_per_set_h) + ")(" + str(
                        patches_per_set_l) + ")" + "_[" + str(stride_3d[0]) + 'x' + str(stride_3d[1]) + 'x' + str(
                        stride_3d[2]) + ']' + "_" + reduction + "_psm" + str(patch_select_mode)
                    if blks_rand_shift_mode:  # if randomly shifting blocks during training process
                        suffix_npy += '_rsb'

                    if not os.path.exists(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy + '.npy')):
                        print("training data not found, so must create it")
                        ###############################################################################
                        # create training arrays
                        ###############################################################################
                        totalpatches = len(srcfiles) * patches_per_set

                        xtrain_master_noaug = np.zeros([totalpatches, blksz_3d[0], blksz_3d[1], blksz_3d[2], 1],
                                                       dtype=np.float16)
                        ytrain_master_noaug = np.zeros([totalpatches, blksz_3d[0], blksz_3d[1], blksz_3d[2], 1],
                                                       dtype=np.float16)
                        print('totalblocks, and train data size: ', totalpatches, xtrain_master_noaug.shape)

                        # count the number of total slices to learn from during training phase
                        # (i.e. loop through all data sets except the one that is being reconstructed)
                        slice_count = 0  # counter holding number of slices incorporated into training
                        ###############################################################################
                        # load .npy files from disk, display central coronal slice, and fill xtrain and ytrain matrices
                        ###############################################################################
                        slices_per_file = []  # recording the number of slices used for training data extractions
                        if raw_projection > 0:
                            n_slices_exclude = n_edge_after_proj

                        for m, icode in enumerate(srcfiles):  # loop over volumes to train from
                            print('##############################################')
                            print('low resolution volume is =>', icode)
                            ######################################
                            # load numpy arrays, reproject dataset (if needed) and trim and normalize dataset,
                            ###################################
                            volume1, volume1max = load_tiff_volume_and_scale_si(dirsource, icode, crop_train_x,
                                                                                crop_train_y, blksz_3d[:2],
                                                                                raw_projection,
                                                                                subset_train_mode, subset_train_minslc,
                                                                                subset_train_maxslc)

                            print('high resolution volume is =>', tgtfiles[m])
                            volume3, volume3max = load_tiff_volume_and_scale_si(dirtarget, tgtfiles[m],
                                                                                crop_train_x,
                                                                                crop_train_y, blksz_3d[:2],
                                                                                raw_projection,
                                                                                subset_train_mode, subset_train_minslc,
                                                                                subset_train_maxslc)

                            print('creating training data set...')
                            # prepare training data for 3D Unet or 3D Residual net

                            # create metric volume used to select blocks
                            vol_metric, metric_operator, nregions = compute_metric_volume_3d(volume1, volume3,
                                                                                             patch_select_mode,
                                                                                             stride_3d)
                            # create the training input (xtrain_master...) and output (ytrain_master...) 3D blocks
                            xtrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :, :,
                            0], ytrain_master_noaug[m * patches_per_set:(m + 1) * patches_per_set, :, :, :,
                                0], blockvolmap = get_blocks_within_volume(vol_metric, [volume1, volume3], blksz_3d,
                                                                           stride_3d, patches_per_set_h,
                                                                           n_lower=patches_per_set_l, seed=m,
                                                                           shuffleP=blks_rand_shift_mode,
                                                                           metric_operator=metric_operator,
                                                                           nregions=nregions, return_bvm=True)
                            print(np.amax(blockvolmap), np.amin(blockvolmap))
                            blockvolmap = np.uint16(np.round(
                                np.float(volume3max / np.amax(blockvolmap)) * np.moveaxis(blockvolmap, -1,
                                                                                          0)))  # move slices from 3rd dimension to 1st dimension
                            tifffile.imsave(
                                'blockvolmap' + suffix_npy + '_' + str(m) + "_" + str(nregions) + "regions.tiff",
                                blockvolmap, compress=6)
                            slices_per_file.append(patches_per_set)
                            print('train data fitted: m, m*patches_per_set:(m+1)*patches_per_set: ', m,
                                  m * patches_per_set, (m + 1) * patches_per_set)

                            if m == (len(srcfiles) - 1):  # if last volume, save the training data to disk
                                np.save(os.path.join(script_path, 'xtrain_master_noaug' + suffix_npy),
                                        xtrain_master_noaug)
                                np.save(os.path.join(script_path, 'ytrain_master_noaug' + suffix_npy),
                                        ytrain_master_noaug)

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
                        print('loaded xtrain_master_noaug' + suffix_npy + '.npy' + ' shape: ',
                              xtrain_master_noaug.shape)
                        print('loaded ytrain_master_noaug' + suffix_npy + '.npy' + ' shape: ',
                              ytrain_master_noaug.shape)
                    print(" ")

                    ###############################################################################
                    #  flip the data in y, x, an z directions, augmenting by factors of 2, 4, or 8
                    ###############################################################################
                    ytrain_master_noaug = ytrain_master_noaug - xtrain_master_noaug  # COMPUTE RESIDUAL!!!
                    if data_augm_factor >= 2:
                        xtrain_master = np.concatenate((xtrain_master_noaug, np.flip(xtrain_master_noaug, axis=1)))  # flip vertically (y)
                        ytrain_master = np.concatenate((ytrain_master_noaug, np.flip(ytrain_master_noaug, axis=1)))
                    if data_augm_factor >= 4:
                        xtrain_master = np.concatenate((xtrain_master, np.flip(xtrain_master, axis=2)))  # flip horizontally (x)
                        ytrain_master = np.concatenate((ytrain_master, np.flip(ytrain_master, axis=2)))
                    if data_augm_factor >= 8 and len(xtrain_master_noaug.shape) == 4:
                        xtrain_master = np.concatenate((xtrain_master, np.flip(xtrain_master, axis=3)))  # flip over (z)
                        ytrain_master = np.concatenate((ytrain_master, np.flip(ytrain_master, axis=3)))
                    else:  # no augmentation
                        ytrain_master = np.copy(ytrain_master_noaug)
                        xtrain_master = np.copy(xtrain_master_noaug)

                    shape_xtrain_master_noaug = xtrain_master_noaug.shape
                    del ytrain_master_noaug
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
                        inds_all = np.arange(0, xtrain_master.shape[0] * data_augm_factor)
                        if leave_one_out_train:
                            print('leave one out training mode, leaving out data set', str(set_to_train + 1), '=>',
                                  srcfiles[set_to_train])
                            n_blocks_per_set_noaug = int(shape_xtrain_master_noaug[0] / len(srcfiles))
                            for ii in range(0, data_augm_factor):  # compute indices in xtrain
                                if ii == 0:  # original indices in set 'set_to_train' without augmentation
                                    inds_to_remove = np.arange(set_to_train * n_blocks_per_set_noaug,
                                                               (set_to_train + 1) * n_blocks_per_set_noaug)
                                else:  # add additional indices in set 'set_to_train' with augmentation
                                    inds_to_remove = np.concatenate((inds_to_remove,
                                                                     np.arange(set_to_train * n_blocks_per_set_noaug, (
                                                                             set_to_train + 1) * n_blocks_per_set_noaug) +
                                                                     shape_xtrain_master_noaug[0] * ii))
                            # compute indices to train from (i.e. all indices that do not come from the 'set to exclude'
                            inds_to_train_from = inds_all[~np.in1d(inds_all, inds_to_remove).reshape(inds_all.shape)]
                        else:
                            print('using all data to train network')
                            inds_to_train_from = np.copy(inds_all)
                        xtrain = np.copy(xtrain_master[inds_to_train_from, :, :, :])
                        ytrain = np.copy(ytrain_master[inds_to_train_from, :, :, :])

                        ###############################################################################
                        # define model and print summary
                        ###############################################################################   
                        try:
                            del model
                        except:
                            pass
                        if filter_mode_1d:  # 1x1x3 filter size
                            model = scrc3dflexfiltersize((blksz_3d[0], blksz_3d[1], blksz_3d[2], 1),
                                                         filters=resnet_filters, filtersize=(1, 1, 3), filter_out=1,
                                                         depth=curr_depth, activation='relu', dropout=resnet_dropout)
                        else:  # 3x3x3 filter size
                            model = scrc3dflexfiltersize((blksz_3d[0], blksz_3d[1], blksz_3d[2], 1),
                                                         filters=resnet_filters, filtersize=(3, 3, 3), filter_out=1,
                                                         depth=curr_depth, activation='relu', dropout=resnet_dropout)
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
                        checkpoint2 = EarlyStopping(monitor='val_loss', patience=training_patience, verbose=1,
                                                    mode='min')
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
                              modelprefix + modelsuffix + "-" + "{0:0=2d}".format(lastepochtofind),
                              'and ends with .hdf5')
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
                        model.save_weights(os.path.join(dirmodel, modelprefix + modelsuffix + "_set" + str(
                            (set_to_train + 1)) + ".h5"))
                        print("saved model to disk")
                else:
                    pass  # network already trained so do nothing

if sleep_when_done:
    # from: https://stackoverflow.com/questions/37009777/how-to-make-a-windows-10-computer-go-to-sleep-with-a-python-script
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
