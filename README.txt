Deep Learning Super-Resolution MRI / MRA

This repository contains code for performing 2D patch-based and 3D block-based deep learning super-resolution MRI and MRA. Four deep neural networks are included, including a 2D U-Net (see note below regarding the 2D U-net), a 3D U-Net, a 2D serial convolution with residual connection (SCRC) neural network, and a 3D SCRC deep neural network.

To use, place the low resolution volumes in multi-page tiff format in the "input_low_res_Xfold" directory (where "X" is the reduction factor), and the corresponding high resolution volumes (also in tiff format) in the "input_high_res" directory.
 	The 2D U-Net can be trained and then applied by running the "unet2d_train.py" and "unet2d_predict.py" scripts
	The 3D U-Net can be trained and then applied by running the "unet3d_train.py" and "unet3d_predict.py" scripts
	The 2D SCRC network can be trained and applied by running the "scrc2d_train.py" and "scrc2d_predict.py" scripts
	The 3D SCRC network can be trained and applied by running the "scrc3d_train.py" and "scrc3d_predict.py" scripts

Note: With minimal modification, the 2D U-Net network, for instance, can be applied for denoising, to reproduce the results reported in Koktzoglou et al. Magn Reson Med. 2020 Aug;84(2):825-837. doi: 10.1002/mrm.28179. Epub 2020 Jan 23. (https://doi.org/10.1002/mrm.28179). For denoising (instead of super-resolution reconstruction), place the noisy volumes in a "input_low_res_Xfold" directory, and the higher signal-to-noise ratio volumes in the "input_high_res" folder. 

Important configuration parameters are found early in the "train" and "predict" scripts. In particular, the 'combomatrix' lists hold several important training and reconstruction parameters. You can add more parameters to these lists.

Note: 2D U-Net code is not included in the "models.py" file but an excellent version can be found here: https://github.com/pietz/unet-keras

Please see the LICENSE.txt file
