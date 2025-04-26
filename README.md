# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the required libraries.

### Step2
Convert the image from BGR to RGB.

### Step3
Apply the required filters for the image separately.

### Step4
Plot the original and filtered image by using matplotlib.pyplot.

### Step5
End the program.


## Program:
### Developed By   :Bhuvaneshwaran H
### Register Number:212223240018
</br>

### 1. Smoothing Filters
### Import libraries and load the image
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("bose.webp")  # replace with your filename
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert to RGB
```
i) Using Averaging Filter
```
kernel = np.ones((5, 5), np.float32) / 25  # 5x5 averaging kernel
image3 = cv2.filter2D(image2, -1, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title("Averaging Filter")
plt.axis("off")
plt.show()

```
ii) Using Weighted Averaging Filter
```
# Custom kernel for weighted averaging
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]], np.float32) / 16

weighted_avg = cv2.filter2D(image2, -1, kernel1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(weighted_avg)
plt.title("Weighted Average Filter")
plt.axis("off")
plt.show()

```
iii) Using Gaussian Filter
```
gaussian_blur = cv2.GaussianBlur(image2, (5, 5), 0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()

```
iv)Using Median Filter
```
median = cv2.medianBlur(image2, 5)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(median)
plt.title("Median Filter")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Load the image
image1 = cv2.imread("bose.webp")  # Change filename if needed
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
```
i) Using Laplacian Linear Kernal
```
# First smooth the image
kernel = np.ones((11,11), np.float32) / 121
smoothed_img = cv2.filter2D(image2, -1, kernel)

# Define a Laplacian sharpening kernel
kernel2 = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])

# Apply sharpening
sharpened_img = cv2.filter2D(image2, -1, kernel2)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(sharpened_img)
plt.title("Sharpened Image (Laplacian Kernel)")
plt.axis("off")
plt.show()

```
ii) Using Laplacian Operator
```
laplacian = cv2.Laplacian(image2, cv2.CV_64F)  # Use 64F to capture negative values

# To properly display Laplacian output, we need to normalize/convert:
laplacian = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')  # Laplacian result is usually shown in grayscale
plt.title("Laplacian Operator Result")
plt.axis("off")
plt.show()
```

## OUTPUT:
### 1. Smoothing Filters

i) Using Averaging Filter

![image](https://github.com/user-attachments/assets/3de775df-f7c1-4b22-8d83-b3c576217a78)



ii)Using Weighted Averaging Filter

![image](https://github.com/user-attachments/assets/b363193c-8305-44af-9f9c-39e947e141d0)

</br>

iii)Using Gaussian Filter

![image](https://github.com/user-attachments/assets/1685d897-cd2f-4db5-ab1f-b7e557a75770)

</br>

iv) Using Median Filter

![image](https://github.com/user-attachments/assets/1c563ee6-d523-4b39-9743-7883e15179f2)

</br>

### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal

![image](https://github.com/user-attachments/assets/fe873796-2063-460e-8764-123160f1ab4a)


ii) Using Laplacian Operator
![image](https://github.com/user-attachments/assets/1d3826f0-525c-4853-abf7-d4e41d35b642)


## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
