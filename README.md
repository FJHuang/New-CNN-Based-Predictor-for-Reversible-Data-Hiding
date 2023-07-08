# New-CNN-Based-Predictor-for-Reversible-Data-Hiding

DOI: 10.1109/LSP.2022.3231193

## Abstract

In this letter, we propose a convolutional neural network (CNN) based predictor for reversible data hiding (RDH). Firstly, a new image division strategy is presented, which can divide the cover image into four independent parts. Via using it, any pixel in each part can be predicted by all its 8-neighbor pixels to generate the preprocessed images. Then, the preprocessed image is fed into a carefully designed CNN-based prediction model to output the predicted image, which is used to build the prediction-error histogram for RDH. Experimental results demonstrate that a sharply distributed prediction-error histogram (i.e., small prediction errors) can be easily obtained by our proposed CNN-based predictor. Furthermore, combining with the classical prediction-error expansion (PEE) embedding strategy, a series of new RDH algorithms with higher visual quality can be formed in contrast to the state-of-the-art RDH schemes.

## Code

trainging.py: Training code for CNNP

cnnp.py: the CNN model of CNNP
