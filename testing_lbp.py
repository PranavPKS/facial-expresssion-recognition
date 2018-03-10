import numpy as np
import cv2
import dlib
import os
#import pickle
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
#from skimage import io

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#desc = LocalBinaryPatterns(8, 1)
no_pts=8
n_bins = 256
rad=1

os.chdir('Testing images/')

img = cv2.imread('hap(8).png',0)

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


print 'p0'         
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
print len(temp_list)


#hog_data = np.append( np.append( np.append( hog.compute(temp_list[0],winStride).T , hog.compute(temp_list[1],winStride).T ) , hog.compute(temp_list[2],winStride).T ) , hog.compute(temp_list[3],winStride).T )
lbp_data=[]
hi=0
while hi<len(temp_list):
    lbp=local_binary_pattern(temp_list[hi], no_pts, rad, method='ror')
    x = itemfreq(lbp.ravel())
    hi = hi+1

        
if cv2.waitKey(100) & 0xFF == ord('q'):
    sys.exit()    

#print hog_data
#print hog_data.shape
#cv2.imwrite('norm_img.jpg',norm_img)
#cv2.imwrite('left_lip_patch.jpg',left_lip_patch)
#cv2.imwrite('right_lip_patch.jpg',right_lip_patch)
#cv2.imwrite('betw_ibrows.jpg',betw_ibrows)
#cv2.imwrite('left_cheek_patch.jpg',left_cheek_patch)
#cv2.imwrite('right_cheek_patch.jpg',right_cheek_patch)
#cv2.imwrite('below_lip_patch.jpg',below_lip_patch)
#cv2.imwrite('nose_patch.jpg',nose_patch)
#val,rt = svm.predict(hog_data.reshape(1,svm.getVarCount()))
#print expressions[int(rt[0][0]-1)]
                    
