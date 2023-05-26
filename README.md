# Image Denoising: Traditional Image Processing filters Vs. Deep Learning

## Introduction:
Image Denoising is the the process of enhancing an image by removing any noise from it. There are some image processing filters that are usually used in image denoising. However, these filters usually result in blurring the original image and not removing the noise well. Recently, some Deep Learning methods has been used to denoise the image while preserving the quality of the original image. This prject tries to solve the questions: How to use Deep Learning in image denoising and How can traditional image processing filters and deep learning models be compared in this context.

## Data:
[ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100) dataset is used in this project. It is a subset of ImageNet-1k Dataset from ImageNet Large Scale Visual Recognition Challenge 2012. A noised copy of the data is saved after adding gaussian noise with zero mean and 0.05 standard deviation.

## Comparison Metrics:
Different denoising methods used in this project are compared both qualitatively and quantitatively; qualitatively on the basis of the visual apperances of the resulted images and quantitatively on the basis of PSNR between the original validation set and the results of the methods on noised copy of the validation set.

## Image Denoising filters:
Some common image processing filters are compared and the filter with the highest PSNR score is choosed to be compared with the deep learning model. The compared filters are: Gaussian, Box, Median, Bilateral, and Non-local means. Each of them is tried with different parameters. 

## The deep learning model:
A U-Net denoising autoencoder is trained on the trainig subset of the data and the PSNR is calculated on the validation subset.

## Results:
The highest PSNR result achieved from thesr filters is 32.07 from the Bilateral Filter with a filter size of 5 and a sigma of 55 for both color space and cordinate space, but when compared with the deep learning model, the deep learning model achieved a 33.36 PSNR and resulted in better visual apperances. Some example images can be found [here](results).
