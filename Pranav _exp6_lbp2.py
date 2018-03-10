import cv2
from sklearn import svm
from dependancy3 import *
from initial1 import *
import matplotlib.pyplot as plt
import time


##Initialize all variables and objects used  
expressions =['Angry','Disgust','Fear','Happy','Sad','Surprised','Neutral']

##labels for the CK database
label1=np.empty(342 , dtype='int')
label1[0:43],label1[43:93],label1[93:147],label1[147:221],label1[221:272],label1[272:342] = 1,2,3,4,5,6



##labels for the RafD database
label2= np.array([[1],[2],[3],[4],[5],[6]])
label2 = np.repeat(label2,67,axis=1).T.reshape(1,-1)
label2 = np.asarray(label2).reshape(-1)

##Get the feature matrix, initial average neutral feature vector and the average position of points in it
print 'stay neutral'
ini_samp,avg_ini,avg_p3 = get_init_samp()


##Load the predefined SVM classifier

clf = np.load('clf_new_lbp.npy')

##Load the CK+ 416 images feature set: 342 peak (from all) and 74 neutral(from happiness)
#ckk = np.load('samples_hog_ck_new.npy')
print 'loaded'
##Stack the initial neutral face feature matrix 
#ckk = np.vstack([ckk, ini_samp])
##Set the labels
#label =np.empty(416 + ini_samp.shape[0] , dtype='int')
#label[0:342], label[342:416 + ini_samp.shape[0]] = 0,1
##Train the SVM classifier
#clf1 = svm.SVC(kernel='linear', C=100,probability=True).fit(ckk, label)

print 'trained'





##Create and set the required params

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


##Set the Optical flow parameters
lk_params = dict( winSize  = (20,20), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


##get the initial frame and its points
ret, frame = cap.read()
frame = cv2.flip(frame,1)
old_out_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

##call detect_face_n_landmarks to obtain the required points in ref, which is used as a reference to detect misdirection
ref,faceSize = detect_face_n_landmarks(frame)
p0 = ref

if faceSize:
    faceROI = cropFaceROI(frame,p0[2],faceSize)
    faceROI = cv2.resize(faceROI, (150, 150))
    old_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

    ref1,f = detect_face_n_landmarks(faceROI)
    p2 = ref1     
else:
    p2 = np.array([])


##Probability bar graph initialization
index = np.arange(6)
bar_width = 0.35
rects = plt.bar(index,[1,1,1,1,1,1],bar_width)
plt.ion()
plt.xlabel('Expressions')
plt.ylabel('Probability')
plt.xticks(index, ('Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness','Surprise'))
plt.legend()
plt.tight_layout()
fig = plt.figure()
plt.show()

##Update function for the bar graph
def update_line(y):
    for rect, h in zip(rects, y):
        rect.set_height(h)
    fig.canvas.draw()

fno=0
seconds=0


while(cap.isOpened()):
    start = time.time()
    fno=fno+1
    
    
    ##get the next frame
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame_out_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ##if face is not detected in the previous iteration p0 will be empty
    ##which has to be reinitialized by checking the presence of face in next frame
    if not p0.size:
        ref,faceSize = detect_face_n_landmarks(frame)
        p0 = ref
        
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
        
            
            
        ##To avoid errors using 'try' to follow the new p1 values
        ##if indexing has occured the 'except' will be executed where the points are reinitialized
        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_out_gray, frame_out_gray, p0, None, **lk_params)
            faceROI = cropFaceROI(frame,p1[2],faceSize)
            faceROI = cv2.resize(faceROI, (150,150))
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
                
    
                try:
                    ##Get the new tracked points 
                    p3, st, err = cv2.calcOpticalFlowPyrLK(old_gray, ROI_gray, p2, None, **lk_params)
                    ##Display some points in the window
                    cv2.putText(faceROI,'.',(p3[0,0],p3[0,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[1,0],p3[1,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[2,0],p3[2,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[3,0],p3[3,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                    cv2.putText(faceROI,'.',(p3[4,0],p3[4,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)

                    ##Get the hog features data by calling get_hog()
                    lbp_data = get_lbp(ROI_gray,p3)
                    #if clf1.predict( hog_data.reshape(1,-1) )[0]:
                        #update_line(np.float32([0,0,0,0,0,0]))
                    
                    lbp_data = np.subtract(lbp_data,avg_ini)    #subtract the average neutral of the responder's face
                    dd = get_d(p3,avg_p3)   #get the differences between the points in the current expression and average neutral points
                    lbp_data = np.append(lbp_data,dd)   #Complete feature vector

                        ##Predict the results on the test_input
                    
                    print expressions[clf.item().predict( lbp_data.reshape(1,-1) )[0]-1]
                        #prob = np.float32(clf.item().predict_proba( hog_data.reshape(1,-1) )[0])
                        #update_line(prob)   ##Update the bar graph
                    
                    ##the current p3 has to p2 fot the next iteration 
                    p2 = p3
                except:
                    p2,f = detect_face_n_landmarks(faceROI)
                    
            ##the next iteration requires the current frame as it old frame reference
            old_gray = ROI_gray.copy()    

            ##Display the window  
            cv2.imshow('face_ROI',faceROI)
            

            ##the current p1 has to p0 fot the next iteration 
            p0 = p1
        except:
            p0,faceSize = detect_face_n_landmarks(frame)
        
                
    ##hold 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    ##the next iteration requires the current frame as it old frame reference
    old_out_gray = frame_out_gray.copy()
    end = time.time()
    seconds = seconds + (end - start)
    if fno == 100:
        break

print 100/seconds
##Close all the windows
plt.close('all')
cap.release()
cv2.destroyAllWindows()
