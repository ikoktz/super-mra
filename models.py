import numpy as np
from keras.layers import Conv2D, Concatenate, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers import UpSampling3D, Dropout, BatchNormalization, Activation
from keras.models import Input, Model


# 3D U-Net
def unet3d(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
           dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False, zdim=8, true_unet=True, kzmax=3):
    """
    ##################################
    # 3D U-Net
    ##################################
    """
    i = Input(shape=img_shape)
    print(img_shape)
    o = level_block3d(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual, zdim,
                      true_unet, kzmax)
    o = Conv3D(1, (1, 1, 1), padding='valid', activation='linear')(o)  # 3D input and 3D output have the same size
    return Model(inputs=i, outputs=o)


# convolution block for 3D U-Net
def conv_block3d(m, dim, acti, bn, res, zdim, kzmax=3, do=0):
    # kzmax should be odd integer, the maximum kernel size in z direction
    kernz = max(zdim, kzmax)
    kernz = (kernz // 2) * 2 + 1
    n = Conv3D(dim, (3, 3, kernz), activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv3D(dim, (3, 3, kernz), activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n


# level block for 3D U-Net
def level_block3d(m, dim, depth, inc, acti, do, bn, mp, up, res, zdim, true_unet, kzmax):
    print('m', m)
    print('dim', dim)
    if depth > 0:
        n = conv_block3d(m, dim, acti, bn, res, zdim, kzmax)
        if zdim == 1:
            m = MaxPooling3D(pool_size=(2, 2, 1))(n) if mp else Conv3D(dim, 3, strides=2, padding='same')(n)
            zstride = 1
            zdim_next = 1
        elif zdim == 3:
            m = MaxPooling3D(pool_size=(2, 2, 3))(n) if mp else Conv3D(dim, 3, strides=2, padding='same')(n)
            zstride = 3 if true_unet else 1  # set zstride = 1 for expansion if Unet3Din2Dout is true
            zdim_next = 1
        else:
            zdim_next = zdim // 2
            zstride = np.int(zdim / zdim_next)
            m = MaxPooling3D(pool_size=(2, 2, zstride))(n) if mp else Conv3D(dim, 3, strides=2, padding='same')(n)
            zstride = np.int(
                zdim / zdim_next) if true_unet else 1  # set zstride to 1 for expansion if Unet3Din2Dout is true
        m = level_block3d(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res, zdim_next, true_unet, kzmax)
        if up:
            m = UpSampling3D()(m)
            m = Conv3D(dim, 2, activation=acti, padding='same')(m)
        else:
            print(zstride)
            if zstride == 1:
                m = Conv3DTranspose(dim, 3, strides=(2, 2, 1), activation=acti, padding='same')(m)
            elif zstride == 2:
                m = Conv3DTranspose(dim, 3, strides=(2, 2, 2), activation=acti, padding='same')(m)
            elif zstride == 3:
                m = Conv3DTranspose(dim, 3, strides=(2, 2, 3), activation=acti, padding='same')(m)
            else:
                print("error in Unet3d ....")
                return
        n = Concatenate()([n, m])
        m = conv_block3d(n, dim, acti, bn, res, zdim, kzmax)
    else:
        m = conv_block3d(m, dim, acti, bn, res, zdim, kzmax, do)
    return m


# 3D 'resnet' (i.e. serial convolution + residual connection)
def scrc3d(input_shape, filters=64, filter_out=1, depth=20, activation='relu', dropout=0.5):
    """
    ##################################
    # 3D ResNet (Serial Convolution + Residual Connection)
    ##################################
    """
    in_ = Input(shape=input_shape)
    out_ = in_
    for i in range(depth - 1):
        if ((i != depth // 2) & (dropout > 0)) or dropout == 0:
            out_ = Conv3D(filters, 3, activation=activation, padding='same')(out_)
        else:
            out_ = Conv3D(filters, 3, activation=None, padding='same')(out_)
            out_ = Dropout(dropout)(out_)
            out_ = Activation(activation=activation)(out_)
    # it is said:  As a rule of thumb, place the dropout after the activate function for all activation functions other than relu
    out_ = Conv3D(filter_out, 3, padding='same')(out_)
    return Model(inputs=in_, outputs=out_)


# 3D 'resnet' (i.e. serial convolution + residual connection w/ variable 'filtersize')
def scrc3dflexfiltersize(input_shape, filters=64, filtersize=(3, 3, 3), filter_out=1, depth=20, activation='relu',
                         dropout=0.5):
    """
    ##################################
    # 3D ResNet (Serial Convolution + Residual Connection) w/ variable 'filtersize'
    ##################################
    """
    in_ = Input(shape=input_shape)
    out_ = in_
    for i in range(depth - 1):
        if ((i != depth // 2) & (dropout > 0)) or dropout == 0:
            out_ = Conv3D(filters, filtersize, activation=activation, padding='same')(out_)
        else:
            out_ = Conv3D(filters, filtersize, activation=None, padding='same')(out_)
            out_ = Dropout(dropout)(out_)
            out_ = Activation(activation=activation)(out_)
    # it is said:  As a rule of thumb, place the dropout after the activate function for all activation functions other than relu
    out_ = Conv3D(filter_out, filtersize, padding='same')(out_)
    return Model(inputs=in_, outputs=out_)


# 2D 'resnet' (i.e. serial convolution + residual connection)
def scrc2d(input_shape, filters=64, filter_out=1, depth=20, activation='relu', dropout=0.5):
    """
    ##################################
    # 2D ResNet (Serial Convolution + Residual Connection)
    ##################################
    """
    in_ = Input(shape=input_shape)
    out_ = in_
    for i in range(depth - 1):
        if ((i != depth // 2) & (dropout > 0)) or dropout == 0:
            out_ = Conv2D(filters, 3, activation=activation, padding='same')(out_)
        else:
            out_ = Conv2D(filters, 3, activation=None, padding='same')(out_)
            out_ = Dropout(dropout)(out_)
            out_ = Activation(activation=activation)(out_)
    # it is said:  As a rule of thumb, place the dropout after the activate function for all activation functions other than relu
    out_ = Conv2D(filter_out, 3, padding='same')(out_)
    return Model(inputs=in_, outputs=out_)
