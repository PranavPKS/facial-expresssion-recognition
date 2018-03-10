import numpy as np
import cv2
import dlib
import os
import glob
from skimage import io
import sys
import pickle
from randomizing import *


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




expressions =['Anger','Disgust','Fear','Happiness','Sadness','Surprise']

label=np.empty(342 , dtype='int')
label[0:43],label[43:93],label[93:147],label[147:221],label[221:272],label[272:342] = 1,2,3,4,5,6

#train_indices , test_indices = randy_fn()

#svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
 #                   svm_type = cv2.ml.SVM_C_SVC,
  #                  C=2.67, gamma=5.383 )

#capture=[]
#access=[]
#hog_access=[]
k=0
m=0

for exp in expressions:
    #capture=[]
    for j,f in enumerate(glob.glob(os.path.join("Selected Cohn_Kanade images",exp,"*/*.png"))):

        k=k+1
        if not k%8:
        
            print("Processing file: {}".format(f))
            img = io.imread(f)

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
            p0 = np.concatenate((p0, [[shape.part(35).x, shape.part(35).y]]), axis=0)
            
            
            c1=16
            c2=20

            left_lip_patch = cv2.resize(norm_img[p0[0,1]- c1: p0[0,1]+c1, p0[0,0]-c1 : p0[0,0]+c1],(32,32))
            right_lip_patch = cv2.resize(norm_img[p0[1,1]-c1 : p0[1,1]+c1, p0[1,0]-c1 : p0[1,0]+c1],(32,32))
            betw_ibrows = cv2.resize(norm_img[p0[3,1]-c1 : p0[3,1]+c1, p0[3,0]-c2 : p0[4,0]+c2], (40,32))
            cheek_patch = cv2.resize(norm_img[p0[5,1]-c1 : p0[5,1]+c1 , p0[5,0] : p0[5,0]+(2*c2)], (32,32))
            
            temp_list = [left_lip_patch, right_lip_patch, betw_ibrows, cheek_patch]

            hog_data = np.append( np.append( np.append( hog.compute(temp_list[0],winStride).T , hog.compute(temp_list[1],winStride).T ) , hog.compute(temp_list[2],winStride).T ) , hog.compute(temp_list[3],winStride).T )
            

            try:
                samples_array = np.vstack([samples_array, hog_data])

            except:
                samples_array = hog_data
                print hog_data.shape
                print sys.exc_info()[0]
                print 'except'
                if m==1:
                    sys.exit()
                m=1
            
            #hog_temp=[]
            #hog_temp.append(hog_data)
            #hog_temp.append(label[k/8])
            
            #hog_access.append(hog_temp)

            #capture.append(temp_list)
            
            #for i in range(0,5):
                #cv2.putText(norm_img,'.',(p0[i,0],p0[i,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        
            cv2.imshow('norm_img',norm_img)
            cv2.imshow('left_lip_patch',left_lip_patch)
            cv2.imshow('cheek_patch',cheek_patch)
        
            if cv2.waitKey(100) & 0xFF == ord('q'):
                sys.exit()
    #access.append(capture)



#svm = cv2.ml.SVM_create()
#svm.setType(cv2.ml.SVM_C_SVC)
#svm.setKernel(cv2.ml.SVM_LINEAR)
#svm.setC(2.67)
#svm.setGamma(5.383)


#svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
#svm.train(samples_array, cv2.ml.ROW_SAMPLE, label)

#svm.train(train_input, train_resp, params=svm_params)
#svm.save('svm_main_data.dat')
outFile = open('new_exp.txt','wb')
pickle.dump(samples_array,outFile)
outFile.close()


print 'Done'
#result = svm.predict(test_input)

#mask = (result[1].T == test_resp)
#correct = np.count_nonzero(mask)
#print 'Accuracy',correct*100.0/mask.shape[1]
