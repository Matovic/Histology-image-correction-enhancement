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
### 1. Load image and convert to grayscale
After loading images, we downsized histological images up to half their size and converted them to grayscale.

Function for resizing images and for displaying images:
```python
def show_img(img: cv2.Mat, txt: str) -> None:
    """
    Show images
    :param: img - image
    :param: txt - text of a window
    """
    cv2.imshow(txt, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def resize_img(img: cv2.Mat, scale_percent: int) -> cv2.Mat:
    """
    Resizing images.
    :param: img - image
    :param: scale_percent - percent by which the image is resized
    :return: Resized image
    """
    # calculate the scale percent of original dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    img_resize = cv2.resize(img, dsize)
    return img_resize
```

Conversion to grayscale:
```python
img1_grayscale = cv2.cvtColor(img1_resize, cv2.COLOR_BGR2GRAY)
img2_grayscale = cv2.cvtColor(img2_resize, cv2.COLOR_BGR2GRAY)
img3_grayscale = cv2.cvtColor(img3_resize, cv2.COLOR_BGR2GRAY)
```

Original images:  
<p align="center">
	<img src="./outputs/images.png">
</p>

Grayscale images:  
<p align="center">
	<img src="./outputs/images_grayscale.png">
</p>

### 2. Image pre-processing  
Noise removal with blurring for 3 images:  
```python
img_blur = cv2.blur(img_grayscale, (5,5))
img_gauss = cv2.GaussianBlur(img_grayscale, (5,5), 0)
img_median = cv2.medianBlur(img_grayscale, 5)
img_bilateral = cv2.bilateralFilter(img_grayscale, 9, 75, 75)
```

<p align="center">
	<img src="./outputs/1_blur.png">
	<img src="./outputs/2_blur.png">
	<img src="./outputs/3_blur.png">
</p>

Sharpening image with sharpening kernel defined from Laplacian kernel:

```python
# show original image
show_img(img1_grayscale, 'img1 grayscale')

# create a sharpening kernels
sharpening_kernel1 = np.array([
    [-1,-1,-1],
    [-1, 9,-1],
    [-1,-1,-1]
])

sharpening_kernel2 = np.array([
    [0,-1, 0],
    [-1,5,-1],
    [0,-1, 0]
])

# convolution
img1_sharpening1 = cv2.filter2D(img1_grayscale ,-1, sharpening_kernel1)
img1_sharpening2 = cv2.filter2D(img1_grayscale ,-1, sharpening_kernel2)
show_img(img1_sharpening1, 'img1_sharpening1')
show_img(img1_sharpening2, 'img1_sharpening2')
```

<p align="center">
	<img src="./outputs/1_sharpening.png">
</p>

### 3. Binarization
Binarization of image 1 using thresholding & inrage after gaussian blurring:  

```python
retValue, img1_threshold = cv2.threshold(img1_gauss, 127, 255, cv2.THRESH_BINARY)
retValue, img1_otsu = cv2.threshold(img1_gauss, 0, 255, cv2.THRESH_OTSU)
img1_adaptiveThreshold = cv2.adaptiveThreshold(img1_gauss, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
img1_inrange = cv2.inRange(img1_gauss, 127, 255)
```

<p align="center">
	<img src="./outputs/1_binarization.png">
</p>

Binarization of image 2 using thresholding & inrage after bilateral blurring: 
 
```python
retValue, img2_threshold = cv2.threshold(img2_bilateral, 127, 255, cv2.THRESH_BINARY)
retValue, img2_otsu = cv2.threshold(img2_bilateral, 0, 255, cv2.THRESH_OTSU)
img2_adaptiveThreshold = cv2.adaptiveThreshold(img2_bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
img2_inrange = cv2.inRange(img2_bilateral, 127, 255)
```

<p align="center">
	<img src="./outputs/2_binarization.png">
</p>

Binarization of image 3 using thresholding & inrage after median blurring: 
 
```python
retValue, img3_threshold = cv2.threshold(img3_median, 127, 255, cv2.THRESH_BINARY)
retValue, img3_otsu = cv2.threshold(img3_median, 0, 255, cv2.THRESH_OTSU)
img3_adaptiveThreshold = cv2.adaptiveThreshold(img3_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
img3_inrange = cv2.inRange(img3_median, 127, 255)
```

<p align="center">
	<img src="./outputs/3_binarization.png">
</p>

Binarization using Sobel edge detection:  
```python
# img1
grad_x = cv2.Sobel(src=img1_gauss, ddepth=-1, dx=1, dy=0, ksize=3)
grad_y = cv2.Sobel(src=img1_gauss, ddepth=-1, dx=0, dy=1, ksize=3)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

# blending two images
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# img2
img2_grad_x = cv2.Sobel(src=img2_bilateral, ddepth=-1, dx=1, dy=0, ksize=3)
img2_grad_y = cv2.Sobel(src=img2_bilateral, ddepth=-1, dx=0, dy=1, ksize=3)

img2_abs_grad_x = cv2.convertScaleAbs(img2_grad_x)
img2_abs_grad_y = cv2.convertScaleAbs(img2_grad_y)

img2_grad = cv2.addWeighted(img2_abs_grad_x, 0.5, img2_abs_grad_y, 0.5, 0)

# img3
img3_grad_x = cv2.Sobel(src=img3_median, ddepth=-1, dx=1, dy=0, ksize=3)
img3_grad_y = cv2.Sobel(src=img3_median, ddepth=-1, dx=0, dy=1, ksize=3)

img3_abs_grad_x = cv2.convertScaleAbs(img3_grad_x)
img3_abs_grad_y = cv2.convertScaleAbs(img3_grad_y)

img3_grad = cv2.addWeighted(img3_abs_grad_x, 0.5, img3_abs_grad_y, 0.5, 0)
```

Binarization using Laplacian & Canny edge detection:   
```python
img1_laplac = cv2.Laplacian(src=img1_gauss, ddepth=-1, ksize=7)
img1_canny = cv2.Canny(img1_gauss, 30, 100)

img2_canny = cv2.Canny(img2_bilateral, 30, 100)
img2_laplac = cv2.Laplacian(src=img2_bilateral, ddepth=-1, ksize=7)

img3_laplac = cv2.Laplacian(src=img3_median, ddepth=-1, ksize=7)
img3_canny = cv2.Canny(img3_median, 30, 100)
```

Compare edge detection with adaptive thresholding after gaussian blurring on image 1:
<p align="center">
	<img src="./outputs/1_edge_detection.png">
</p>

Compare edge detection with adaptive thresholding after bilateral blurring on image 2:
<p align="center">
	<img src="./outputs/2_edge_detection.png">
</p>

Compare edge detection with adaptive thresholding after median blurring on image 3:
<p align="center">
	<img src="./outputs/3_edge_detection.png">
</p>

### 4. Cell segmentation
Using morphological operations(dilate, erode, structuring element, distance transform, Morphological Gradient, top hat, black hat) and convolution after Canny Edge Detection after Gauss blurring or sharpening:
```python
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

img1_dilate = cv2.dilate(img1_canny,(-1, -1), 3)

img1_dilate_inverted = cv2.bitwise_not(img1_dilate)

img1_dilate_element = cv2.dilate(img1_canny, element, 3)

img1_dilate_inverted_element = cv2.bitwise_not(img1_dilate_element)

img1_sharp1_dilate_element = cv2.dilate(img1_sharp1_canny, element, 3)

img1_sharp2_dilate_element = cv2.dilate(img1_sharp2_canny, element, 3)

img1_sharp1_dilate_element_inverted = cv2.bitwise_not(img1_sharp1_dilate_element)´

img1_sharp2_dilate_element_inverted = cv2.bitwise_not(img1_sharp2_dilate_element)

img1_erode = cv2.erode(img1_canny, kernel=(3,3))

img1_closing = cv2.morphologyEx(img1_canny, cv2.MORPH_CLOSE, element)

img1_closing_inverted = cv2.bitwise_not(img1_closing)

img1_opening = cv2.morphologyEx(img1_canny, cv2.MORPH_OPEN, element)

img1_opening_inverted = cv2.bitwise_not(img1_opening)

img1_canny_inverted = cv2.bitwise_not(img1_canny)

img1_closing_canny_inverted = cv2.morphologyEx(img1_canny_inverted, cv2.MORPH_CLOSE, element)

img1_opening_canny_inverted = cv2.morphologyEx(img1_canny_inverted, cv2.MORPH_OPEN, element)

element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

img1_gradient = cv2.morphologyEx(img1_canny, cv2.MORPH_GRADIENT, element2)

img1_tophat = cv2.morphologyEx(img1_canny, cv2.MORPH_TOPHAT, (7,7))

img1_blackhat = cv2.morphologyEx(img1_canny, cv2.MORPH_BLACKHAT, element)

img1_conv = cv2.filter2D(img1_canny, -1, element)

img1_conv_inverted = cv2.bitwise_not(img1_conv)
```

Declared function for distance transform with normalization and dilatation:
```python
def distance_transform(img: cv2.Mat) -> cv2.Mat:
    """
    Preform distance_transform.
    :param: img - binary image
    :returns: image after distance transform
    """
    img_distanceTransform = cv2.distanceTransform(src=img, distanceType=cv2.DIST_L2, maskSize=5)
    show_img(img_distanceTransform, 'img_distanceTransform')

    # Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
    cv2.normalize(img_distanceTransform, img_distanceTransform, 0, 1.0, cv2.NORM_MINMAX)
    show_img(img_distanceTransform, 'img_distanceTransform normalized')

    # Threshold to obtain the peaks. This will be the markers for the foreground objects
    _, img_distanceTransform = cv2.threshold(img_distanceTransform, 0.4, 1.0, cv2.THRESH_BINARY)

    # Dilate the dist image
    kernel1 = np.ones((3,3), dtype=np.uint8)
    img_distanceTransform = cv2.dilate(img_distanceTransform, kernel1)
    show_img(img_distanceTransform, 'img_dist dilate')

    return img_distanceTransform
```

<p align="center">
	<img src="./outputs/1_morphological_op.png">
	<img src="./outputs/1_dilate_sharp.png">
	<img src="./outputs/1_erode.png">
	<img src="./outputs/1_opening_closing.png">
	<img src="./outputs/1_morphological_op_compare.png">
</p>

Using morphological operations(dilate, erode, structuring element, distance transform, Morphological Gradient, top hat, black hat) and convolution on image 2 after Adaptive Threshold and bilateral blurring and on image 3 after Laplacian edge detection and median blurring:

```python
# img2
img2_dilate = cv2.dilate(img2_adaptiveThreshold,(-1, -1), 3)

img2_distanceTransform = cv2.distanceTransform(src=img2_adaptiveThreshold, distanceType=cv2.DIST_L2, maskSize=5)

img2_erode	= cv2.erode(img2_adaptiveThreshold, kernel=(3,3))

# img3
img3_dilate = cv2.dilate(img3_laplac,(-1, -1), 3)

img3_distanceTransform = cv2.distanceTransform(src=img3_laplac, distanceType=cv2.DIST_L2, maskSize=5)

img3_erode	= cv2.erode(img3_laplac, kernel=(3,3))
```

<p align="center">
	<img src="./outputs/2_morphological_op.png">
	<img src="./outputs/3_morphological_op.png">
</p>

Function for contour analysis:
```python
def contours(img: cv2.Mat, img_input: cv2.Mat, mode: Any, method: int) -> Tuple[cv2.Mat, int]:
    """
    Countour analysis
    :param: img - original image
    :param: img_input - image after morphological operation
    :param: mode - mode in cv2.findContours
    :param: method - method in cv2.findContours
    :returns: tuple of resulting image and number of cells
    """
    img_result = img.copy()
    img_contours, img_hierarchy = cv2.findContours(img_input, mode, method)
    cell_count = 0

    for i in range(0, len(img_contours)):
        if cv2.contourArea(img_contours[i]) > 10:
            cell_count += 1
            cv2.drawContours(img_result, img_contours, i, (0, 255, 0), 4)
        i += 1
    return img_result, cell_count
```

Function for displaying segmentated cells:
```python
def show_cells(img: cv2.Mat, img_result: cv2.Mat, no_cells: int) -> None:
    """
    Show original image and segmentated cells.
    :param: img - original image
    :param: img_result - segmentated image
    :param: cells - number of cells in image
    """
    plt.figure()
    f, axarr = plt.subplots(1,2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

    axarr[0].imshow(img_rgb)
    axarr[1].imshow(img_result_rgb)

    axarr[0].set_title('Original image')
    axarr[1].set_title(f'Cell segmentation - {no_cells} cells')

    axarr[0].axis('off')
    axarr[1].axis('off')

    plt.show()
```
Contours analysis of image 1 after dilating binary image from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_dilate.png">
</p>

Contours analysis of image 1 after dilating binary image with structuring element from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_dilate_structuring_element.png">
</p>

Contours analysis of image 1 after inverting dilated binary image from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_dilate_structuring_element_inverted.png">
</p>

Contours analysis of image 1 after inverting dilated(with structuring element) binary image from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_dilate_structuring_element_inverted.png">
</p>

Contours analysis of image 1 after dilating binary image from Canny Edge Detection and sharpening with a kernel of 4-neighbours: 
<p align="center">
	<img src="./outputs/1_cells_sharp4_canny_dilate.png">
</p>

Contours analysis of image 1 after dilating binary image from Canny Edge Detection and sharpening with a kernel of 8-neighbours: 
<p align="center">
	<img src="./outputs/1_cells_sharp8_canny_dilate.png">
</p>

Contours analysis of image 1 after inverting dilated binary image from Canny Edge Detection and sharpening with a kernel of 4-neighbours: 
<p align="center">
	<img src="./outputs/1_cells_sharp4_canny_dilate_inverted.png">
</p>

Contours analysis of image 1 after inverting dilated binary image from Canny Edge Detection and sharpening with a kernel of 8-neighbours: 
<p align="center">
	<img src="./outputs/1_cells_sharp8_canny_dilate_inverted.png">
</p>

Contours analysis of image 1 after dilating normalized image from distance transform and Canny Edge Detection and sharpening with a kernel of 8-neighbours: 
<p align="center">
	<img src="./outputs/1_cells_sharp8_distance_transform.png">
</p>

Contours analysis of image 1 after closing image from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_closing.png">
</p>

Contours analysis of image 1 after Canny Edge Detection and Gauss blurring (without morphological operations): 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny.png">
</p>

Contours analysis of image 1 after Morphological Gradient from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_gradient.png">
</p>

Contours analysis of image 1 after blackhat from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_blackhat.png">
</p>

Contours analysis of image 1 after convolution from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_conv.png">
</p>

Contours analysis of image 1 after inverting convolution from Canny Edge Detection and Gauss blurring: 
<p align="center">
	<img src="./outputs/1_cells_gauss_canny_conv_inverted.png">
</p>

Contours analysis of image 2 after dilating binary image from Adaptive Threshold and bilateral blurring: 
<p align="center">
	<img src="./outputs/2_cells_dilate.png">
</p>

Contours analysis of image 2 after eroding binary image from Adaptive Threshold and bilateral blurring: 
<p align="center">
	<img src="./outputs/2_cells.png">
</p>

Contours analysis of image 2 after Adaptive Threshold and bilateral blurring(without morphological operations): 
<p align="center">
	<img src="./outputs/2_cells_bilateral_adaptive.png">
</p>

Contours analysis of image 2 after dilateting normalized distance transform binary image from Adaptive Threshold and bilateral blurring: 
<p align="center">
	<img src="./outputs/2_cells_bilateral_distance_transform.png">
</p>

Contours analysis of image 3 after eroding binary image from Laplac edge detection and median blurring: 
<p align="center">
	<img src="./outputs/3_cells.png">
</p>

Contours analysis of image 3 after Laplacian Edge Detection and median blurring(no morphological operations): 
<p align="center">
	<img src="./outputs/3_cells_median_laplac.png">
</p>

Contours analysis of image 3 after dilateting image from Laplacian Edge Detection and median blurring(no morphological operations): 
<p align="center">
	<img src="./outputs/3_cells_median_laplac_dilate.png">
</p>

Contours analysis of image 3 after dilateting normalized image from distance transform from Laplacian Edge Detection and median blurring(no morphological operations): 
<p align="center">
	<img src="./outputs/3_cells_median_laplac_distance_transform.png">
</p>

## Conclusion  
We have used three methods on three images. The first image used Gaussian blurring, Canny Edge Detection to get binary image and we have used multiple morphological operations(dilate, erode, morphological gradient, tophat, blackhat, opening, closing, inverting images and so on) and convolution for contour analysis. The first image also used sharpening instead of a blurring. The second image used bilateral blurring, adaptive thresholding as binarization, and dilate/erode operation for contour analysis. Erode process showed more potential for segmentation because it segmented more cells. The third image used median blurring, Laplac edge detection, and erode operation for contour analysis.  

Based on previous experiments we can conclude that contours analysis shows nothing on image 1 after eroding, inverted closing, opening, inverted opening, inverted mask of canny and tophat.

