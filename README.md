# Deep Learning Super-Resolution MRI / MRA

Code for performing 2D patch-based and 3D block-based deep learning super-resolution as described in:
	
	Koktzoglou, I, Huang, R, Ankenbrandt, WJ, Walker, MT, Edelman, RR. Super‐resolution head and neck MRA using deep machine learning. Magn Reson Med. 2021; https://doi.org/10.1002/mrm.28738

Tested for MRA and MRI. Four deep neural networks can be trained and applied, including a 2D U-Net (see note below regarding 2D U-net code), a 3D U-Net, a 2D serial convolution with residual connection (SCRC) deep neural network, as well as a 3D SCRC deep neural network.

## Usage

To use, place the low resolution volumes in multi-page _.tif_ or _.tiff_ format in the _"input_low_res_X_fold"_ directory (where "_X_" is the reduction factor), and the corresponding high resolution volumes (also in _.tif_ or _.tiff_ format) in the _"input_high_res"_ directory. You'll need to create these folders in the same directory as the Python scripts.
 + The 2D U-Net can be trained and later used for prediction by running the _"unet2d_train.py"_ and _"unet2d_predict.py"_ scripts, respectively
 + The 3D U-Net can be trained and later used for prediction by running the _"unet3d_train.py"_ and _"unet3d_predict.py"_ scripts, respectively
 + The 2D SCRC network can be trained and later used for prediction by running the _"scrc2d_train.py"_ and _"scrc2d_predict.py"_ scripts, respectively
 + The 3D SCRC network  can be trained and later used for prediction by running the _"scrc3d_train.py"_ and _"scrc3d_predict.py"_ scripts, respectively

## Notes
 + 2D U-Net Keras code is not included in the "models.py" file but an excellent version can be found at: https://github.com/pietz/unet-keras
 + Important configuration parameters are found early in the _"train.py"_ and _"predict.py"_ scripts in the _'combomatrix'_ variable. This list variable contains several important network training and reconstruction parameters. Add more parameters combinations to these lists to train and apply variously configured models.
 + With minimal modification, the 2D U-Net code can also be applied for image denoising, to reproduce the results reported in: Koktzoglou et al. Magn Reson Med. 2020 Aug;84(2):825-837. doi: 10.1002/mrm.28179. Epub 2020 Jan 23. (https://doi.org/10.1002/mrm.28179). For denoising (instead of super-resolution reconstruction), place the noisy volumes in a "input_low_res_Xfold" directory, and the higher signal-to-noise ratio volumes in the "input_high_res" folder. 

Please note the LICENSE.txt file.
