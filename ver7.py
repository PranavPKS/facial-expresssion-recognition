import cv2
from sklearn import svm
from dependancy2 import *
from initial import *
import matplotlib.pyplot as plt
import time

train = 1
if train == 1:
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

    clf = np.load('clf_new.npy')    
    ##Load the CK+ 416 images feature set: 342 peak (from all) and 74 neutral(from happiness)
    ckk = np.load('samples_hog_ck_new.npy')
    print 'loaded'
    ##Stack the initial neutral face feature matrix 
    ckk = np.vstack([ckk, ini_samp])
    ##Set the labels
    label =np.empty(416 + ini_samp.shape[0] , dtype='int')
    label[0:342], label[342:416 + ini_samp.shape[0]] = 0,1
    ##Train the SVM classifier
    clf1 = svm.SVC(kernel='linear', C=100,probability=True).fit(ckk, label)
    #np.save('clf1.npy',clf1)
    #personparam = {'avg_p3':avg_p3,'avg_ini':avg_ini}
    #np.save('personparam.npy',personparam)
    print 'trained'

###Load the predefined SVM classifier
#personparam = np.load('personparam.npy').item()
#avg_ini = personparam['avg_ini']
#avg_p3 = personparam['avg_p3']

#clf1 = np.load('clf1.npy')


#Initialize all params and objects used  
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
winStride = (2,2)
    
##Create and set the required params

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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
    #print y
    for rect, h in zip(rects, y):
        rect.set_height(h)
    fig.canvas.draw()

time_log = []
##get the initial frame and its points
ret,frame = cap.read()
frame = cv2.flip(frame,1)
p0,faceSize = detect_face_n_landmarks(frame)
facePointRatio = faceSize/(p0[2,1] - p0[8,1])
frame_out_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
old_out_gray = frame_out_gray.copy()
##Set the Optical flow parameters
lk_params = dict( winSize  = (int(faceSize/8),int(faceSize/8)), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


while(cap.isOpened()):
    start_time = time.time()
    ##get the next frame
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame_out_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    ##To avoid errors using 'try' to follow the new p1 values
    ##if indexing has occured the 'except' will be executed where the points are reinitialized
    try:
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_out_gray, frame_out_gray, p0, None, **lk_params)  
        p4 = normalizeFacePoints(p1.copy(),faceSize.copy())        

        faceROI = cropFaceROI(frame_out_gray,p1[2],faceSize)   
        ROI_gray = cv2.resize(faceROI, (150,150))
        #cv2.imshow('face',ROI_gray)
        
        ##Get the hog features data by calling get_hog()
        #hog_data = get_hog(ROI_gray,p4.copy())
        hog_data = get_hog1(ROI_gray,p4,hog,winStride)
        if clf1.predict( hog_data.reshape(1,-1) )[0]:
            update_line(np.float32([0,0,0,0,0,0]))
        else:
            hog_data = np.subtract(hog_data,avg_ini)    #subtract the average neutral of the responder's face
            dd = get_d(p1,avg_p3)   #get the differences between the points in the current expression and average neutral points
            hog_data = np.append(hog_data,dd)   #Complete feature vector

            ##Predict the results on the test_input
        
            #print expressions[clf.item().predict( hog_data.reshape(1,-1) )[0]-1]
            prob = np.float32(clf.item().predict_proba( hog_data.reshape(1,-1) )[0])
            update_line(prob)   ##Update the bar graph
        p0 = p1
    except:
        print 'detecting again..................'
        p0,faceSize = detect_face_n_landmarks(frame)
    if max(err)>5:
        print 'Tracking error==========================..'
        p0,faceSize = detect_face_n_landmarks(frame)

    ##the next iteration requires the current frame as it old frame reference
    old_out_gray = frame_out_gray.copy()    
    try:

        ##Display some points in the window
        p5 = p0
        cv2.putText(frame,'.',(p5[0,0],p5[0,1]),cv2.FONT_HERSHEY_SIMPLEX, .4,(0,255,255),1)
        cv2.putText(frame,'.',(p5[1,0],p5[1,1]),cv2.FONT_HERSHEY_SIMPLEX, .4,(0,255,255),1)
        cv2.putText(frame,'.',(p5[2,0],p5[2,1]),cv2.FONT_HERSHEY_SIMPLEX, .4,(0,255,255),1)
        cv2.putText(frame,'.',(p5[3,0],p5[3,1]),cv2.FONT_HERSHEY_SIMPLEX, .4,(0,255,255),1)
        cv2.putText(frame,'.',(p5[4,0],p5[4,1]),cv2.FONT_HERSHEY_SIMPLEX, .4,(0,255,255),1)

        cv2.imshow('face_ROI',frame)     
        # change face Size
        faceSize = facePointRatio * (p0[2,1] - p0[8,1])
        #print 1/np.mean(time_log[len(time_log)-30:len(time_log)])        
    except:
        print 'landmark points not available'
    ##hold 'q' to exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    end_time = time.time()
    time_log.append(end_time - start_time)
    

##Close all the windows
plt.close('all')
cap.release()
cv2.destroyAllWindows()
