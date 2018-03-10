import numpy as np
import cv2
import dlib
import os
import glob
import sys
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


no_pts=8
n_bins = 59
rad=1



def get_samp_lbp(f):
    
    print("Processing file: {}".format(f))
    img = cv2.imread(f,0)

    dets = detector(img, 1)

    temp = []
    for d in dets:
        temp.append(d.height())
    d1 = dets[int(np.argmax(temp))]
    #print d1, temp
        
    #print d.top(),d.bottom(),d.left(),d.right()
    b1 = 10
    crop_img = img[d1.top()-b1:d1.bottom()+b1,d1.left()-b1:d1.right()+b1]

    norm_img = cv2.resize(crop_img, (150,150))

    dets = detector(norm_img, 1)

    temp = []
    for d in dets:
        temp.append(d.height())
    d1 = dets[int(np.argmax(temp))]

    shape = predictor(norm_img, d1)
        
    p0 = np.array([[shape.part(48).x, shape.part(48).y]])
    p0 = np.concatenate((p0, [[shape.part(54).x, shape.part(54).y]]), axis=0)
    p0 = np.concatenate((p0, [[shape.part(33).x, shape.part(33).y]]), axis=0)
    p0 = np.concatenate((p0, [[shape.part(21).x, shape.part(21).y]]), axis=0)
    p0 = np.concatenate((p0, [[shape.part(22).x, shape.part(22).y]]), axis=0)
    p0 = np.concatenate((p0, [[shape.part(31).x, shape.part(31).y]]), axis=0)
    p0 = np.concatenate((p0, [[shape.part(35).x, shape.part(35).y]]), axis=0)
    p0 = np.concatenate((p0, [[shape.part(57).x, shape.part(57).y]]), axis=0)
    p0 = np.concatenate((p0, [[shape.part(27).x, shape.part(27).y]]), axis=0)    
            
    c1=16
    c2=20

    left_lip_patch = cv2.resize(norm_img[p0[0,1]- c1: p0[0,1]+c1, p0[0,0]-c1 : p0[0,0]+c1],(32,32))
    right_lip_patch = cv2.resize(norm_img[p0[1,1]-c1 : p0[1,1]+c1, p0[1,0]-c1 : p0[1,0]+c1],(32,32))
    betw_ibrows = cv2.resize(norm_img[p0[3,1]-c1 : p0[3,1]+c1, p0[3,0]-c2 : p0[4,0]+c2], (40,32))
    left_cheek_patch = cv2.resize(norm_img[p0[5,1]-int(1.25*c1) : p0[5,1]+ int(0.75*c1) , p0[5,0]-(2*c2)-5 : p0[5,0]-5], (40,32))
    right_cheek_patch = cv2.resize(norm_img[p0[6,1]- int(1.25*c1) : p0[6,1]+ int(0.75*c1) , p0[6,0]+5 : p0[6,0]+(2*c2)+5], (40,32))
    below_lip_patch = cv2.resize(norm_img[p0[7,1]-int(0.75*c1): p0[7,1]+int(1.25*c1), p0[7,0]-c2 : p0[7,0]+c2], (40,32))
    nose_patch = cv2.resize(norm_img[p0[8,1]: p0[8,1]+int(1.5*c1), p0[8,0]-int(0.75*c1) : p0[8,0]+int(0.75*c1)], (32,32))
    
    temp_list = [left_lip_patch, right_lip_patch, betw_ibrows, left_cheek_patch, right_cheek_patch, below_lip_patch, nose_patch]

    lbp_data=[]
    hi=0
    while hi<len(temp_list):
        lbp=local_binary_pattern(temp_list[hi], no_pts, rad, 'nri_uniform')
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        lbp_data = np.append(lbp_data, hist)
        hi = hi+1

    cv2.imshow('norm_img',norm_img)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        sys.exit()


    return lbp_data    






for f in glob.glob("RafD Frontal Faces/*.jpg"):

    lbp_data=get_samp_lbp(f)

    try:
        samples_array = np.vstack([samples_array, lbp_data])

    except:
        samples_array = lbp_data
        print lbp_data.shape
        print sys.exc_info()[0]
        print 'except'

        
    if cv2.waitKey(100) & 0xFF == ord('q'):
        sys.exit()




z=0
y=0

while z<469:
    if not (z-4)%7:
        try:
            ra_neut = np.vstack([ra_neut, samples_array[z]])

        except:
            ra_neut = samples_array[z]
    z=z+1

while y<469:
    if (y-4)%7:
        try:
            ra_new = np.vstack([ra_new,np.subtract(samples_array[y],ra_neut[y/7])])
        except:
            ra_new = np.subtract(samples_array[y],ra_neut[y/7])
    y=y+1


print ra_new.shape
np.save('ra_lbp_nri',ra_new)


print 'Done'
