# Histology image correction / enhancement
Name: Erik Matovič  
Methods used: 

## Assignment
### Dataset  
Dataset link: [link](https://drive.google.com/file/d/11_y1TZOKQb7xl4esCCjYcM9o4uv7w5pC/view) (Histology nuclei)  

[H&M](https://drive.google.com/file/d/1o7PdpZfsnh7O4xbbwNt3Y7zyPhsZWW3J/view) and [P63](https://drive.google.com/file/d/1a70V9PDNdzAV4FDyEVaqV6ZbUfFflzA6/view) registered images.  

The stained colors of the tissue components are popularly used as features for image analysis. Dataset contain stained histology images. Variations in the staining condition of the histology slides could however negatively impact the accuracy of the analysis.   

The assignment goal is to experiment with the histology images and try to correct the color distribution of the provided dataset.  

### Experiment with the following tasks for image correction
 - Histogram computation (visualize histogram for each color model used)
 - Histogram equalization - for multiple color models (Grayscale, RGB, YCbCr, HSV, XYZ, Lab). Make assumption how to properly correct the image using following color models and document the steps with your reasoning.
 - Gamma correction (hint: pow)
 - **Optional:** Source -> Target color correction using eCDF (effective Cummulative Distribution Function) and linear interpolation
   - Target is an image you selected from the dataset. We want to change the color distribution of other (sources) images to the target one
   - for RGB and YCbCr only

### Experiment with the following tasks for image enhancement
 - Segmentation utilizing the color information (Lab color model)
   - Make sure your Lab conversion is in correct data range 
   - Select the small part of the image containing only nuclei and compute its Lab color model (target) - average L / a / b values
   - Compute delta Lab - difference image between the input and the target
     - Hint how to compute difference in the slides from lecture
     - Visualize
   - Use the obtained difference image and try to segment nuclei by thresholding
 - Use your knowledge of local descriptors to detect and localize image patch within the larger image from following data - [link](https://drive.google.com/file/d/10oNHED7BGrcYomKd3Cn5J1ZKQuQ9TR5d/view)
   - Experiment with feature detectors: SIFT, FAST, Harris
   - Experiment with feature descriptors: SIFT, SURF, ORB
   - Compute homography matrix and localize your image patch
     - Find the best combination of detection & description for your task


## Usage
To run Jupyter Notebook, you need OpenCV and matplotlib. You can install them using pip:  
```bash
pip install opencv-python matplotlib
```

[OpenCV documentation](https://docs.opencv.org/4.7.0/)

## Solution
### 1. Load and resize images
After loading images, we downsized histological images up to half their size and converted them to grayscale.


Original images:  
<p align="center">
	<img src="./outputs/images.png">
</p>


### 2. Image correction
Histogram computation (visualize histogram for each color model used)


### 3. Image enhancement
Segmentation utilizing the color information (Lab color model)
   - Make sure your Lab conversion is in correct data range 
   - Select the small part of the image containing only nuclei and compute its Lab color model (target) - average L / a / b values
   - Compute delta Lab - difference image between the input and the target
     - Hint how to compute difference in the slides from lecture
     - Visualize
   - Use the obtained difference image and try to segment nuclei by thresholding


## Conclusion  
TO DO

