import numpy as np
import cv2
import dlib
import os
import glob
from skimage import io
import sys
from randomizing import *


#Initialize all variables and objects used  
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

##HOG Descriptor Object
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#compute(img[, winStride[, padding[, locations]]]) -> descriptors
winStride = (2,2)




expressions =['Anger','Disgust','Fear','Happiness','Sadness','Surprise']

##defining all the class labels as numbers
label=np.empty(342 , dtype='int')
label[0:43],label[43:93],label[93:147],label[147:221],label[221:272],label[272:342] = 1,2,3,4,5,6

##From randomizing get the required random indices
train_indices , test_indices = randy_fn()


hog_access=[]
##k is defined for getting the eighth images only.. Check the code
##m is used for initialization of samples_list and also fo error handling
k=0
m=0

for exp in expressions:
    for j,f in enumerate(glob.glob(os.path.join("Selected Cohn_Kanade images",exp,"*/*.png"))):

        k=k+1
        if not k%8:
        
            print("Processing file: {}".format(f))
            img = io.imread(f)
            ##Detector object for the img
            dets = detector(img, 1)

            ##Just to get the biggest image i.e., the face
            temp = []
            for d in dets:
                temp.append(d.height())
            d1 = dets[int(np.argmax(temp))]
            
            ##b1 can be set if the boundary has to be extended
            
            b1 = 10
            crop_img = img[d1.top()-b1:d1.bottom()+b1,d1.left()-b1:d1.right()+b1]

            ##Normalized image with a specific size
            norm_img = cv2.resize(crop_img, (150,150))
            ##Detector object for the norm_img
            dets = detector(norm_img, 1)

            temp = []
            for d in dets:
                temp.append(d.height())
            d1 = dets[int(np.argmax(temp))]

            shape = predictor(norm_img, d1)
            
            ##48 -left lip corner, 54-right lip corner, 33-nose tip, 21 and 22 are the inner eyebrow corners
            ##add all required points by concatenating
            p0 = np.array([[shape.part(48).x, shape.part(48).y]])
            p0 = np.concatenate((p0, [[shape.part(54).x, shape.part(54).y]]), axis=0)
            p0 = np.concatenate((p0, [[shape.part(33).x, shape.part(33).y]]), axis=0)
            p0 = np.concatenate((p0, [[shape.part(21).x, shape.part(21).y]]), axis=0)
            p0 = np.concatenate((p0, [[shape.part(22).x, shape.part(22).y]]), axis=0)

            
            ##c1 and c2 are the constants that are set to get the required size of those patches
            c1=16
            c2=20

            left_lip_patch = cv2.resize(norm_img[p0[0,1]- c1: p0[0,1]+c1, p0[0,0]-c1 : p0[0,0]+c1],(2*c1,2*c1))
            right_lip_patch = cv2.resize(norm_img[p0[1,1]-c1 : p0[1,1]+c1, p0[1,0]-c1 : p0[1,0]+c1],(2*c1,2*c1))
            betw_ibrows = cv2.resize(norm_img[p0[3,1]-c1 : p0[3,1]+c1, p0[3,0]-c2 : p0[4,0]+c2], (2*c2,2*c1))

            
            temp_list = [left_lip_patch, right_lip_patch, betw_ibrows]

            ##append all the hog features of those patches into hog_data
            hog_data = np.append( np.append( hog.compute(temp_list[0],winStride).T , hog.compute(temp_list[1],winStride).T ) , hog.compute(temp_list[2],winStride).T )
            

            try:
                samples_array = np.vstack([samples_array, hog_data])

            except:
                ##Initialization and error handling 
                ##If it comes more than 1 time here, It will exit by displaying the error
                samples_array = hog_data
                #print hog_data.shape
                #print 'except'
                if m==1:
                    print sys.exc_info()[0]
                    sys.exit()
                m=1
            
            
            #for i in range(0,5):
                #cv2.putText(norm_img,'.',(p0[i,0],p0[i,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        
            cv2.imshow('norm_img',norm_img)
            cv2.imshow('betw_ibrows',betw_ibrows)
        
            if cv2.waitKey(100) & 0xFF == ord('q'):
                sys.exit()


#Let's Shuffle and construct the required training and test data set

for p in train_indices:
    try:
        train_input = np.vstack([train_input,samples_array[p]])
        train_resp = np.append(train_resp,label[p])
    except:
        train_input = samples_array[p]
        train_resp = np.array([label[p]])

for q in test_indices:
    try:
        test_input = np.vstack([test_input,samples_array[q]])
        test_resp = np.append(test_resp,label[q])
    except:
        test_input = samples_array[q]
        test_resp = np.array([label[q]])


##Create and set the required params
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(2.67)
svm.setGamma(5.383)


svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))

##Train the classifier
svm.train(train_input, cv2.ml.ROW_SAMPLE, train_resp)

##Save it as a dat file for future reference
svm.save('svm_data.dat')

##Predict the results on the test_input
result = svm.predict(test_input)

##Calculate its Accuracy
mask = (result[1].T == test_resp)
correct = np.count_nonzero(mask)
print 'Accuracy',correct*100.0/mask.shape[1]
