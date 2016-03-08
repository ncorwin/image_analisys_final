#!/usr/bin/env python

import numpy as np
import cv2 as cv2
import cv2.cv as cv
import sys
import math as m

x=0

############################################################################
def draw(location, mask, img):
    x,y = location

    b = 255
    g = 0
    r = 255

    if x != 0 and y != 0:
            
            cv2.circle(mask, location, 5, (b,g,r), -1)

    out = cv2.bitwise_and(img, mask)
   
    return np.array(out, dtype = "uint8")
############################################################################

def remove_bg1(raw_img):
	#create the weight based on total number of frames that have passed
	global x
	x +=1.0
	alpha = (1.- 1./x)

	#initialize bg_img
	if x ==1:
		bg_img = raw_img
	global bg_img
	
	#continuously updated average to create background
	bg_img=np.add(alpha*np.array(bg_img, dtype=float), (1.- alpha )*np.array(raw_img, dtype=float))
	bg_img = np.array(bg_img, dtype = "uint8")


	return bg_img
def remove_bg(raw_img, avg):

	cv2.accumulateWeighted(raw_img,avg,0.0005)
	bg_img = cv2.convertScaleAbs(avg)
	#cv2.imshow('bg_img',bg_img)

	return bg_img, avg

'''
######foreground practice###
	raw_hsv = cv2.cvtColor(raw_img,cv2.COLOR_BGR2HSV)
	bg_hsv = cv2.cvtColor(bg_img,cv2.COLOR_BGR2HSV)
	hmask = cv2.absdiff(raw_hsv[:,:,0], bg_hsv[:,:,0])
	smask = cv2.absdiff(raw_hsv[:,:,1], bg_hsv[:,:,1])
	vmask = cv2.absdiff(raw_hsv[:,:,2], bg_hsv[:,:,2])
	ret,hmask_thresh = cv2.threshold(hmask,20.,1.,cv2.THRESH_BINARY)
	ret,smask_thresh = cv2.threshold(smask,20.,1.,cv2.THRESH_BINARY)
	ret,vmask_thresh = cv2.threshold(vmask,20.,1.,cv2.THRESH_BINARY)
	mask1 = np.multiply(np.multiply(hmask_thresh, smask_thresh), vmask_thresh)

	# bgsub = cv2.BackgroundSubtractorMOG(10, 2, .1)

	# fgmask = bgsub.apply(raw_img)
	# print np.max(fgmask)

	####filter for only skin tones
	# h= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,0]	
	# s =cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,1]
	# ret, raw_h = cv2.threshold(h, 100., 1., cv2.THRESH_BINARY)
	# ret, raw_s = cv2.threshold(s, 20, 1., cv2.THRESH_BINARY)
	# hs_mask = np.multiply(raw_h, raw_s)
	
	
	
	
    ###Filter out colors with extreme values and no red for skin###
	# ret, rmask = cv2.threshold(img[:,:,2],40,1., cv2.THRESH_BINARY)
	# ret, r2mask = cv2.threshold(img[:,:,2],235.,1., cv2.THRESH_BINARY_INV)
	# rb_mask = np.multiply(rmask, r2mask)
	# img[:,:,0 ]=	np.multiply(img[:,:,0], rb_mask)
	# img[:,:,1 ]=	np.multiply(img[:,:,1], rb_mask)
	# img[:,:,2 ]=	np.multiply(img[:,:,2], rb_mask)
	# bmask = cv2.absdiff(img[:,:,0], bg_img[:,:,0])
	# gmask = cv2.absdiff(img[:,:,1], bg_img[:,:,1])
'''

def foreground(bg_img, raw_img):

	#take the background and subtract somehow from the foreground
	img = raw_img*1
	raw_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	bg_hsv = cv2.cvtColor(bg_img,cv2.COLOR_BGR2HSV)
	#raw_hsv = cv2.GaussianBlur(raw_hsv, (5,5), 2)
	#bg_hsv = cv2.GaussianBlur(bg_hsv, (5,5), 2)
	hmask = cv2.absdiff(raw_hsv[:,:,0], bg_hsv[:,:,0])
	smask = cv2.absdiff(raw_hsv[:,:,1], bg_hsv[:,:,1])
	vmask = cv2.absdiff(raw_hsv[:,:,2], bg_hsv[:,:,2])
	ret,hmask_thresh = cv2.threshold(hmask,1.,1.,cv2.THRESH_BINARY)
	ret,smask_thresh = cv2.threshold(smask,1.,1.,cv2.THRESH_BINARY)
	ret, vmask_thresh = cv2.threshold(vmask,1.,1.,cv2.THRESH_BINARY)
	hsv_mask = np.multiply(hmask_thresh, smask_thresh)
	#hsv_mask = np.multiply(hsv_mask, vmask_thresh)
	#hsv_mask = hmask_thresh
	
	##Greyscale mask that kinda worked except for bright lighting
	raw_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	bg_gray = cv2.cvtColor(bg_img,cv2.COLOR_BGR2GRAY)
	raw_gray = cv2.GaussianBlur(raw_gray, (5,5), 2)
	bg_gray = cv2.GaussianBlur(bg_gray, (5,5), 2)
	mask_g1 =cv2.absdiff(bg_gray,raw_gray)
	mask_g2 =cv2.absdiff(raw_gray,bg_gray)
	ret,mask_g1 = cv2.threshold(mask_g1,10.,1.,cv2.THRESH_BINARY)
	ret,mask_g2 = cv2.threshold(mask_g2,10.,1.,cv2.THRESH_BINARY)
	mask = np.multiply(mask_g1, mask_g2)
	
	
	#mask2_b = cv2.absdiff(img[:,:,0], bg_img[:,:,0])
	#mask2_g = cv2.absdiff(img[:,:,1], bg_img[:,:,1])
	#mask2_r = cv2.absdiff(img[:,:,2], bg_img[:,:,2])
	mask2_b = cv2.absdiff(bg_img[:,:,0], img[:,:,0])
	mask2_g = cv2.absdiff(bg_img[:,:,1], img[:,:,1])
	mask2_r = cv2.absdiff(bg_img[:,:,2], img[:,:,2])
	ret,mask2_b = cv2.threshold(mask2_b,5.,1.,cv2.THRESH_BINARY)
	ret,mask2_g = cv2.threshold(mask2_g,5.,1.,cv2.THRESH_BINARY)
	ret,mask2_r = cv2.threshold(mask2_r,5.,1.,cv2.THRESH_BINARY)
	mask2 = np.multiply(mask2_b, mask2_g)
	mask2 = np.multiply(mask2, mask2_r)
	
	###make changes here    
	mask = mask*1.0
	mask = np.multiply(hsv_mask, mask)
	#mask = cv2.bitwise_xor(mask, mask2)
	for i in range(7):
 		mask = cv2.dilate(mask*255., (50,50))/255.


	for i in range(6):
	  	mask = cv2.erode(mask*255, (50,50))/255.

 # 	for i in range(3):
 # 	 	mask = cv2.dilate(mask*255., (50,50))/255.

	fg_img = img*1.0
	fg_img[:,:,0 ]=	np.multiply(img[:,:,0], mask)
	fg_img[:,:,1 ]=	np.multiply(img[:,:,1], mask)
	fg_img[:,:,2 ]=	np.multiply(img[:,:,2], mask)

	cv2.imshow("fg_img", np.array(fg_img, dtype= "uint8"))

	return np.array(mask, dtype = "uint8")



def main():     
	###read the campera input###
	for i in range(5):
	    cap = cv2.VideoCapture(i)
	    #cap = cv2.VideoCapture(1)
	    if cap: break    
    
    #cap = cv2.VideoCaptur(1)
	null, raw_img =cap.read()
	#initialize the array 'avg'
	avg = np.float32(raw_img)
        out_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

###############################################################
        draw_layer = cv2.imread('draw_layer.jpg')
        #draw_img = np.multiply(draw_layer, raw_img)
################################################################   

	###Main Loop###
	while True:
		null, raw_img = cap.read()
		bg_img, avg = remove_bg(raw_img, avg)
		fg_mask = foreground(bg_img, raw_img)
                out_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

		#fg_gray = cv2.cvtColor(fg_img,cv2.COLOR_BGR2GRAY)
		contours, hier = cv2.findContours(fg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		contour_list = []
		idx = -1
		for i in range(len(contours)):
			if(cv2.contourArea(contours[i])>6000): #and cv2.contourArea(contours[i])<60000):
				contour_list.append(i)
				idx = i
		hull = []
		if idx != -1:
			#print len(contour_list), cv2.contourArea(contours[idx])
			cv2.drawContours(raw_img, contours, idx, [200,50,50], thickness = 3)
			hull = cv2.convexHull(contours[idx])
			cv2.drawContours(raw_img, hull, -1, [10,10,225], thickness = 10)
			#defects = cv2.convexityDefects(contours[idx], hull)

                        
#########################################################################
                tip = []
                for i in range(len(hull)):
                        measure = 10
                        nx,ny = hull[i,0]
                        if i < 2:
                                nx1,ny1 = hull[i+1,0]
                                nx2,ny2 = hull[i+2,0]
                                nx3,ny3 = hull[len(hull)-1,0]
                                nx4,ny4 = hull[len(hull)-2,0]
                        elif i >= len(hull) - 2:
                                nx1,ny1 = hull[0,0]
                                nx2,ny2 = hull[1,0]
                                nx3,ny3 = hull[i-1,0]
                                nx4,ny4 = hull[i-2,0]
                        else:
                                nx1,ny1 = hull[i+1,0]
                                nx2,ny2 = hull[i+2,0]
                                nx3,ny3 = hull[i-1,0]
                                nx4,ny4 = hull[i-2,0]

                        diff1 = abs(m.sqrt(m.pow(nx1-nx, 2)+m.pow(ny1-ny, 2)))
                        diff2 = abs(m.sqrt(m.pow(nx2-nx, 2)+m.pow(ny2-ny, 2)))
                        diff3 = abs(m.sqrt(m.pow(nx3-nx, 2)+m.pow(ny3-ny, 2)))
                        diff4 = abs(m.sqrt(m.pow(nx4-nx, 2)+m.pow(ny4-ny, 2)))

                        if diff1 < measure and diff2 < measure and diff3 < measure and diff4 < measure:
                                tip.append((nx, ny))

                if len(tip) > 0:
                        print tip[0]
                        cv2.circle(raw_img, tip[0], 25, (0,0,0), -1)
                        draw_img = draw(tip[0], draw_layer, out_img)
                else:
                        draw_img = draw((0,0), draw_layer, out_img)
#########################################################################
		if len(hull) != 0:
			#print hull[0:,0]
			print hull.shape
		cv2.imshow("raw_img", np.fliplr(raw_img))
                cv2.imshow("draw_img", np.fliplr(draw_img))
		#cv2.imshow("bg_img", bg_img)

		#break statement for image processing
		if (cv2.waitKey(1) & 0xFF == ord('q')):
			break

main()
cap.release()
cv2.destroyAllWindows()
