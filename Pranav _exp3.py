import cv2
#from sklearn import svm
from sklearn.linear_model import SGDClassifier
#from sklearn.kernel_approximation import RBFSampler
from dependancy2 import *
#import time
from initial import *


#Initialize all variables and objects used  
#labels for the CK database
label1=np.empty(352 , dtype='int')
label1[0:43],label1[43:93],label1[93:147],label1[147:221],label1[221:272],label1[272:342],label1[342:352] = 1,2,3,4,5,6,7

expressions =['Angry','Disgust','Fear','Happy','Sad','Surprised','Neutral']

#labels for the RafD database
label2= np.array([[1],[2],[3],[4],[5],[6]])
label2 = np.repeat(label2,67,axis=1).T.reshape(1,-1)
label2 = np.asarray(label2).reshape(-1)
ltt=np.empty(10)
ltt.fill(7)
label2 = np.concatenate([label2,ltt])

##Load the databases and make it to float32
#samp1 = np.load('samples_hog1.npy')
#samp2 = np.load('samples_hog_rafd.npy')

samp1 = np.load('s3.npy')
samp2 = np.load('r3.npy')

samp3 = np.zeros((10,40572))
samp1= np.vstack([samp1,samp3])
samp2= np.vstack([samp2,samp3])
print 'stay neutral'
samp4,avg_ini = get_init_samp()

samp1 = samp1.astype(np.float32)
samp2 = samp2.astype(np.float32)

#rbf_feature = RBFSampler(gamma=1, random_state=1)

#samp1_features = rbf_feature.fit_transform(samp1)
#samp2_features = rbf_feature.fit_transform(samp2)

##Train the classifier

#clf = svm.SVC(gamma = 0.0001, C=100, kernel ='linear')
#clf.fit(samp,label)

##Using SGD Classifier with the hinge loss and L2 regularization
##you will get an SVM that can be updated online/incremental

clf = SGDClassifier(loss="hinge", penalty="l2")
clf.partial_fit(samp1,label1,classes=np.unique(label1))
clf.partial_fit(samp2,label2,classes=np.unique(label2))



print 'trained'

##Create and set the required params

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#out = cv2.VideoWriter('output.avi',-1, 2.0, (640,480))

lk_params = dict( winSize  = (50,50), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

##Initialize the Thresholds which can be used for smile detection
#th1 =1
#th2 =1



##get the initial frame
ret, frame = cap.read()
frame = cv2.flip(frame,1)
old_out_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

##call detect_face_n_landmarks to obtain the required points in ref, which is used as a reference to detect misdirection
ref,faceSize = detect_face_n_landmarks(frame)
p0 = ref
#print faceSize
if faceSize:
    #print 'entered'
    faceROI = cropFaceROI(frame,p0[2],faceSize)
    faceROI = cv2.resize(faceROI, (300, 300))
    old_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

    ref1,f = detect_face_n_landmarks(faceROI)
    p2 = ref1     
else:
    p2 = np.array([])

#fno=0
#seconds=0

while(cap.isOpened()):
    #start = time.time()
    #fno=fno+1
    #print fno
    ##get the next frame
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame_out_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #frame_copy = frame.copy()
    #print p0
    
    ##if face is not detected in the previous iteration p0 will be empty
    ##which has to be reinitialized by checking the presence of face in next frame
    if not p0.size:
        ref,faceSize = detect_face_n_landmarks(frame)
        p0 = ref
        #print 0,faceSize
    ##if face is present it has to tracked with Optical flow
    
    if p0.size:
        dist_1 = dist_pts(p0[0], ref[0])
        dist_2 = dist_pts(p0[1], ref[1])
        dist_3 = dist_pts(p0[2], ref[2])
        dist_4 = dist_pts(p0[3], ref[3])
        dist_5 = dist_pts(p0[4], ref[4])
        dist = dist_pts(p0[0], p0[1])
        #these distances are calculated to detect misdirection of the points due to optical flow
        #so that it can be changed to the actual points
        if(dist_1 > dist/3 or dist_2 > dist/3 or dist_3 > dist/3 or dist_4 > dist/3 or dist_5 > dist/3):
            ref,faceSize = detect_face_n_landmarks(frame)
            p0 = ref
            #print 1,faceSize
            
            
        ##To avoid errors using 'try' to follow the new p1 values
        ##if indexing has occured the 'except' will be executed where the points are reinitialized
        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_out_gray, frame_out_gray, p0, None, **lk_params)
            #print 'entered'
            faceROI = cropFaceROI(frame,p1[2],faceSize)
            faceROI = cv2.resize(faceROI, (300,300))
            ROI_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

            if not p2.size:
                    ref1,f = detect_face_n_landmarks(faceROI)
                    p2 = ref1
            
            if p2.size:

                dist_1 = dist_pts(p2[0], ref1[0])
                dist_2 = dist_pts(p2[1], ref1[1])
                dist_3 = dist_pts(p2[2], ref1[2])
                dist_4 = dist_pts(p2[3], ref1[3])
                dist_5 = dist_pts(p2[4], ref1[4])
                dist = dist_pts(p2[0], p2[1])
                #these distances are calculated to detect misdirection of the points due to optical flow
                #so that it can be changed to the actual points
                if(dist_1 > dist/3 or dist_2 > dist/3 or dist_3 > dist/3 or dist_4 > dist/3 or dist_5 > dist/3):
                    p2,f = detect_face_n_landmarks(faceROI)
                    #print 'p2',p2,'f',f   
    
                try:
                    p3, st, err = cv2.calcOpticalFlowPyrLK(old_gray, ROI_gray, p2, None, **lk_params)
                    #print 'p3',p3
                    cv2.putText(faceROI,'.',(p3[0,0],p3[0,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[1,0],p3[1,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[2,0],p3[2,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[3,0],p3[3,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[4,0],p3[4,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)

                    ##Get the hog features data by calling get_hog()
                    hog_data = get_hog(ROI_gray,p3)
                    #print hog_data.shape
                    ##Predict the results on the test_input
                    #val,rt = svm.predict(hog_data.reshape(1,svm.getVarCount()))
                    print expressions[clf.predict( np.subtract(hog_data,avg_ini).reshape(1,-1) )[0]-1]
                                 
                    #if smile(p2,p3,th1,th2):
                        ##prints smile if a smile is detected
                        #cv2.putText(faceROI,'SMILE',(100,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1)       
                    
                    p2 = p3
                except:
                    #print 'except'
                    p2,f = detect_face_n_landmarks(faceROI)
    
            old_gray = ROI_gray.copy()    
                      
            cv2.imshow('face_ROI',faceROI)


            ##the current p1 has to p0 fot the next iteration 
            p0 = p1
        except:
            #print 'no face detected'
            p0,faceSize = detect_face_n_landmarks(frame)
            #print 2,faceSize
        
                
    #cv2.imshow('img',frame)
    
    #out.write(frame)
    ##press 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    ##the next iteration requires the current frame as it old frame reference
    old_out_gray = frame_out_gray.copy()
    #end = time.time()
    #seconds = seconds + (end - start)
    #if fno == 100:
        #break
    
#print 100/seconds    
cap.release()
#out.release()
cv2.destroyAllWindows()
