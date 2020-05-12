#!/usr/bin/env python
# coding: utf-8
# # solution 1
# In[1]:


import cv2
import numpy as np
img = cv2.imread('C:/Users/Dr.Stark/Desktop/cv 2.jpg',1)
cv2.imshow('image',img)


# In[2]:


k = cv2.waitKey(0) 
if k == 26:  #right arrow key
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = 10
    cv2.add(hsv[:,:,2], value, hsv[:,:,2])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('out.png', image)
    
elif k == 27: # left arrow key
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = 10
    cv2.sub(hsv[:,:,2], value, hsv[:,:,2])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('out.png', image)
   


    


# # solution 3

# In[3]:


import cv2


img_grey = cv2.imread('C:/Users/Dr.Stark/Desktop/cv 2.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('C:/Users/Dr.Stark/Desktop/cv 3.jpg',img_grey)

thresh = 128

# black and white image 
Black_white = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]


cv2.imwrite('C:/Users/Dr.Stark/Desktop/cv 4.jpg',Black_white)


# In[4]:

#removing red and green scale from the image 
img[:,:,2] = np.zeros([img.shape[0], img.shape[1]])


img[:,:,1] = np.zeros([img.shape[0], img.shape[1]])


cv2.imwrite('C:/Users/Dr.Stark/Desktop/cv 5.jpg',img_)


# In[5]:





# # solution 4

# In[6]:


import cv2 
har = cv2.CascadeClassifier('E:cv2/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread('Desktop/test_image.jpg')


# In[7]:


faces = har.detectMultiScale(img, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if cv2.waitKey(0)==25:
            break


# In[8]:


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

