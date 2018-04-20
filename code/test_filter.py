import numpy as np
import cv2




def my_imfilter( image, filter ):
    # This function is intended to behave like the built in function cv2.filter2D()

    # Your function should work for color images. Simply filter each color
    # channel independently.

    # Your function should work for filters of any width and height
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This
    # restriction makes it unambigious which pixel in the filter is the center
    # pixel.

    # Boundary handling can be tricky. The filter can't be centered on pixels
    # at the image boundary without parts of the filter being out of bounds. 
    # You should simply recreate the default behavior of filter2D -- pad the input image with zeros, and
    # return a filtered image which matches the input resolution. A better
    # approach is to mirror the image content over the boundaries for padding.

    # # Uncomment if you want to simply call filter2D so you can see the desired
    # # behavior. When you write your actual solution, you can't use imfilter,
    # # filter2, conv2, etc. Simply loop over all the pixels and do the actual
    # # computation. It might be slow.
    
    #print(a.dtype)

    ################
    # Your code here
    ################
    return cv2.filter2D(image, -1, filter)


test_image = cv2.imread('../data/cat.bmp')
test_image = cv2.resize(test_image, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) ;
cv2.imshow("test",test_image)






## Identify filter
#This filter should do nothing regardless of the padding method you use.
identity_filter = np.zeros((3,3),np.float32)
identity_filter[2,2] = 1

identity_image = my_imfilter(test_image,identity_filter);

cv2.imshow("identity",identity_image)


## Small blur with a box filter
#This filter should remove some high frequencies
blur_filter = np.ones((3,3),np.float32)
blur_filter = blur_filter / sum(sum(blur_filter)) #making the filter sum to 1

blur_image = my_imfilter(test_image,blur_filter)
cv2.imshow("blur",blur_image)

# Large blur
# This blur would be slow to do directly, so we instead use the fact that
# Gaussian blurs are separable and blur sequentially in each direction.

# gaussian_filter = np.array([0.001133,0.002316,0.00445,0.008033,0.013627,0.021724,0.032542,0.04581
# ,0.0606,0.075333,0.088001,0.096603,0.099654,0.096603,
# 0.088001,0.075333,0.0606,0.04581,0.032542,0.021724,0.013627,0.008033,0.00445,0.002316,0.001133]);
gaussian_filter = np.array([0.001133,0.002316,0.00445,0.008033,0.013627,0.021724,0.032542,0.04581
 ,0.0606,0.075333,0.088001,0.096603,0.099654,0.096603,
 0.088001,0.075333,0.0606,0.04581,0.032542,0.021724,0.013627,0.008033,0.00445,0.002316,0.001133]);


large_blur_image = my_imfilter(test_image,gaussian_filter);

large_blur_image = np.transpose(large_blur_image,(1,0,2));


large_blur_image = my_imfilter(large_blur_image,gaussian_filter);
large_blur_image = np.transpose(large_blur_image,(1,0,2));

cv2.imshow("large_blur",large_blur_image);


## Oriented filter (Sobel Operator)
sobel_filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]); #should respond to horizontal gradients
sobel_image = my_imfilter(test_image, sobel_filter);
cv2.imshow("sobel_image",sobel_image +0.5);
#0.5 added because the output image is centered around zero otherwise and mostly black

## High pass filter (Discrete Laplacian)
laplacian_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]]);
laplacian_image = my_imfilter(test_image, laplacian_filter);
cv2.imshow("laplacian_image",laplacian_image +0.5 ); 
#0.5 added because the output image is centered around zero otherwise and mostly black


# High pass "filter" alternative
high_pass_image = test_image - blur_image; #simply subtract the low frequency content
cv2.imshow("high_pass_image",high_pass_image ); 

cv2.waitKey(0);