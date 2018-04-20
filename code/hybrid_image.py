import numpy as np
import cv2
from matplotlib import pyplot as plt


def create_padding(image, filter_size,filter_size_2) :
 
    h, w = image.shape
    print(h,w)

    k=filter_size
    zp = (k-1)//2

    zp2=0
    if(filter_size_2>0) :
        k2=filter_size_2
        zp2 = (k2-1)//2
 


    # print("padding")
    # print(zp)
    # print(zp2)
    # print(h+zp*2,w+zp2*2)
    padded_image = np.zeros((h+zp*2,w+zp2*2),np.uint8)

    start_h = zp    
    while(start_h <zp+h  ) :
        start_w = zp2
        while(start_w <zp2+w  ) :

   
            temp_h=start_h-zp
            temp_w =start_w-zp2

            padded_image[start_h,start_w] =image[temp_h,temp_w]

            start_w+=1
        start_h+=1


    return padded_image

def convolution(image, filter) :


    filter_size = len(filter)
    filter_size2 = 0
    total =0
    if(filter.ndim==2) :
            filter_size2 =len(filter[0]) 
   
            h,w = image.shape
            
            #output = np.zeros((filter_size,filter_size),np.uint8)
            
            #print(h,w)
            for i in  range(filter_size) :
                for j in  range(filter_size2) :
                    val=(filter[i][j]*image[i][j]) 

                    #val= val /  filter_size*filter_size
                    total+=val
    else :

        for i in  range(filter_size) :

            total+=(filter[i]*image[i])

    return total

def iterate_conv(image, filter,h,w) :
        
        filter_size = len(filter)
        


        filter_size_2 = 0
        if(filter.ndim==2) :
            filter_size_2 =len(filter[0]) 

        zp = (filter_size-1)//2
        zp2=0
        if(filter_size_2>0) :
            zp2 = (filter_size_2-1)//2
        
        output = np.zeros((h,w),np.uint8)

        h1,w1= image.shape
        
        flag_cast=0
        i=zp
        while(i<h1-zp) :
            j=zp2
            while(j<w1-zp2 ) :
                #print(i,j)
                #print(i-(zp) , i+zp+1 , j-(zp2) , j+zp2+1)
                data = image[ i-(zp) : i+zp+1 , j-(zp2) : j+zp2+1 ]
                val = convolution(data , filter)
                
               
                if(val<0) :
                    if(flag_cast==0) :
                        output=output.astype(np.int16)
                        flag_cast=1                        
                output[i-zp,j-zp2]=val
                #print(i-zp,j-zp)
                j+=1
            i+=1


            
                #filter[filter_size//2][filter_size//2]=output[i][j]
        
        return output

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
    return cv2.filter2D(image, -1, filter)
    #print(a.dtype)

    ################
    # Your code here
    ################
    filter_size = len(filter)
    filter_size_2 = 0
    if(filter.ndim==2) :
        filter_size_2 = len(filter[0])
    
    output = []
    if(len(image.shape)<3):
        h, w = image.shape
        padded= create_padding(image, filter_size,filter_size_2)    
        output=iterate_conv(padded, filter,h,w)              
    elif(len(image.shape)==3)  : 
        h, w,_ = image.shape
        b_ori,g_ori,r_ori = cv2.split(image)
        
        

        b= create_padding(b_ori, filter_size,filter_size_2)
        g= create_padding(g_ori, filter_size,filter_size_2)
        r= create_padding(r_ori, filter_size,filter_size_2)
    
        b=iterate_conv(b, filter,h,w)
        g=iterate_conv(g, filter,h,w)  
        r=iterate_conv(r, filter,h,w)
        
        output = np.dstack((b,g,r))
        img = cv2.filter2D(image, -1, filter)
        b_1,g_1,r_1 = cv2.split(img)

    return output 

def createGaussian( kernelSize, std_value ):
    kernel = np.zeros(kernelSize,np.float)
  
    pi = 3.14159265359
    std = std_value *std_value
    for x in  range (kernelSize) :
      
        kernel[x]  = (np.exp(-(x*x)/(2*std) ) /(np.sqrt(pi*2)*std_value  ))
        #kernel.append( np.exp((a*a)/(2*std) ))

    kernel = kernel/sum(kernel)
    return kernel
def combine(low1,high2,image2) :
    # np.seterr(over='ignore')
    h,w,_ = low1.shape
    output = np.zeros((h,w,3),np.uint8)
    for y in range (h) :
        for x in range (w) :
            val= int(low1[y,x,0])+int(high2[y,x,0])
            if(val>=255 ) :
                val=int(low1[y,x,0])
            val1= int(low1[y,x,1])+int(high2[y,x,1])
            if(val1>=255 ) :
                val1=int(low1[y,x,1])


            val2= int(low1[y,x,2])+int(high2[y,x,2])
            if(val2>=255 ) :
                val2=int(low1[y,x,2])


            if(val<=0 ) :
                val=int(low1[y,x,0])

          
            if(val1<=0 ) :
                val1=int(low1[y,x,1])



            if(val2<=0 ) :
                val2=int(low1[y,x,2])


            output[y,x,0] = val
            output[y,x,1] = val1
            output[y,x,2] = val2
       



    return output

def create_pyramid(combined) :
    n =5
    h,w, _ = combined.shape
    test_image1 = cv2.resize(combined, (0,0), fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR) 
    #cv2.imshow('test_image1',test_image)
    h1,w1, _ = test_image1.shape

    test_image2 = cv2.resize(combined, (0,0), fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR) 
    #cv2.imshow('test_image2',test_image)
    h2,w2, _ = test_image2.shape

    test_image3 = cv2.resize(combined, (0,0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR) 
    #cv2.imshow('test_image3',test_image)
    h3,w3,_ = test_image3.shape    


    pyramid = np.zeros((h,w+(n*3)+w1+w2+w3,3),np.uint8 )
    pyramid[0:h, 0:w+(n*4)+w1+w2+w3,:] = 255
    pyramid[0 : h, 0:w] = combined
    pyramid[h-h1 :h , w+n:n+w+w1 ] = test_image1
    pyramid[h-h2 :h , w+w1+n+n:n+n+w+w1+w2 ]  = test_image2
    pyramid[h-h3 :h , w+w1+w2+n+n+n:n+n+n+w+w1+w2+w3 ]  = test_image3

    return pyramid


def plot(img,image,name) :

    color = ('b','g','r')
    for i,col in enumerate(color):
         histr = cv2.calcHist([img],[i],None,[256],[0,256])
         
         if(image!=[]) :
             histr1 = cv2.calcHist([image],[i],None,[256],[0,256])

             #plt.subplot(221),plt.imshow(image)
             plt.subplot(221),plt.plot(histr1,color = col)
             plt.title('original histogram plot')
             plt.xlim([0,256])

             #plt.subplot(223),plt.imshow(img)
             plt.subplot(224),plt.plot(histr,color = col)
             plt.xlim([0,256])
             plt.title(name)
         else :
             #plt.subplot(222),plt.imshow(img)
             plt.subplot(221),plt.plot(histr,color = col)
             plt.xlim([0,253])
             plt.title('hybrid histogram plot')
    plt.savefig(name)
    plt.show()

pairs = []
## Setup
# read images and convert to floating point format
folder = "sample_testing"
image1 = cv2.imread('../html/data/einstein.bmp');
image2 = cv2.imread('../html/data/marilyn.bmp');


# Several additional test cases are provided for you, but feel free to make
# your own (you'll need to align the images in a photo editor such as
# Photoshop). The hybrid images will differ depending on which image you
# assign as image1 (which will provide the low frequencies) and which image
# you asign as image2 (which will provide the high frequencies)

## Filtering and Hybrid Image construction
cutoff_frequency = 15; #This is the standard deviation, in pixels, of the 
# Gaussian blur that will remove the high frequencies from one image and 
# remove the low frequencies from another image (by subtracting a blurred
# version from the original version). You will want to tune this for every
# image pair to get the best results.
# Generate your own Gaussian kernel <== CHECK THIS

###################################################################
# YOUR CODE BELOW. Use my_imfilter to create 'low_frequencies' and
# 'high_frequencies' and then combine them to create 'hybrid_image'
###################################################################


########################################################################
# Remove the high frequencies from image1 by blurring it. The amount of
# blur that works best will vary with different image pairs
########################################################################
gaussian_filter1 = createGaussian(11,cutoff_frequency)
low_frequencies = my_imfilter(image1,gaussian_filter1);
low_frequencies = np.transpose(low_frequencies,(1,0,2));
low_frequencies = my_imfilter(low_frequencies,gaussian_filter1);
low_frequencies = np.transpose(low_frequencies,(1,0,2)) 
#plot(low_frequencies,image1, folder+"/low_frequencies_plot")



########################################################################
# Remove the low frequencies from image2. The easiest way to do this is to
# subtract a blurred version of image2 from the original version of image2.
# This will give you an image centered at zero with negative values.
########################################################################
gaussian_filter1 = createGaussian(9,cutoff_frequency)
low_frequencies_2 = my_imfilter(image2,gaussian_filter1);
low_frequencies_2 = np.transpose(low_frequencies_2,(1,0,2));
low_frequencies_2 = my_imfilter(low_frequencies_2,gaussian_filter1);
low_frequencies_2 = np.transpose(low_frequencies_2,(1,0,2)) 
high_frequencies =np.zeros((image1.shape[0],image1.shape[1]),np.uint8 ) 
high_frequencies=( image2-low_frequencies_2  ) 
#plot(high_frequencies,image2,folder+"/high_frequencies_plot")

########################################################################
# Combine the high frequencies and low frequencies
########################################################################

hybrid_image =combine(low_frequencies,high_frequencies,image1) 

## Visualize and save outputs

#plot(hybrid_image,[],folder+"/hybrid_frequencies_plot")


cv2.imwrite(folder+'/low_frequencies.png',low_frequencies)
cv2.imwrite(folder+'/high_frequencies.png',high_frequencies)
cv2.imwrite(folder+'/hybrid_image.png',hybrid_image)
pyramid = create_pyramid(hybrid_image)
cv2.imwrite(folder+'/pyramid.png',pyramid)


cv2.imshow("low_frequencies",low_frequencies)
cv2.imshow("high_frequencies",high_frequencies+128 )
cv2.imshow("hybrid_image",hybrid_image)
cv2.imshow("pyramid",pyramid)

cv2.waitKey(0)

