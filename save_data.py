import numpy as np
import cv2
import dlib
import os
import glob
#from skimage import io
import pickle
import sys



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


winSize = (32,32)
blockSize = (8,8)
blockStride = (4,4)
cellSize = (4,4)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 16
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors
winStride = (2,2)



def get_samp_hog(f):
    
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

    hog_data=[]
    hi=0
    while hi<len(temp_list):
        hog_data = np.append(hog_data, hog.compute(temp_list[hi],winStride).T)
        hi = hi+1

    cv2.imshow('norm_img',norm_img)
    cv2.imshow('below_lip_patch',below_lip_patch)
    cv2.imshow('left_cheek_patch',left_cheek_patch)
    cv2.imshow('nose_patch',nose_patch)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        sys.exit()


    return hog_data    





expressions =['Anger','Disgust','Fear','Happiness','Sadness','Surprise']

k=0


for exp in expressions:
    for f in glob.glob(os.path.join("Selected Cohn_Kanade images",exp,"*/*.png")):

        k=k+1

        if not k%8:
            hog_data=get_samp_hog(f)

            try:
                samples_array = np.vstack([samples_array, hog_data])

            except:
                samples_array = hog_data
                print hog_data.shape
                print sys.exc_info()[0]
                print 'except'

        
            if cv2.waitKey(100) & 0xFF == ord('q'):
                sys.exit()


k=0

#for exp in expressions:
for f in glob.glob(os.path.join("Selected Cohn_Kanade images/Happiness","*/*.png")):

    k=k+1

    if k%8 == 1:
        hog_data=get_samp_hog(f)
        samples_array = np.vstack([samples_array, hog_data])

            
        if cv2.waitKey(100) & 0xFF == ord('q'):
            sys.exit()


print samples_array.shape
np.save('samples_hog_ck_new',samples_array)
#outFile = open('some.txt','wb')
#pickle.dump(samples_array,outFile)
#outFile.close()

print 'Done'
