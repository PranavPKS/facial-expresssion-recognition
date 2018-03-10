import cv2
from dependancy2 import *

def get_init_samp():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

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
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (2,2)


    
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
        
    ##Frame number
    fno=0

    while(cap.isOpened()):
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
                ##Get the new tracked points 
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
                        cv2.putText(faceROI,'.',(p3[0,0],p3[0,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                        cv2.putText(faceROI,'.',(p3[1,0],p3[1,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                        cv2.putText(faceROI,'.',(p3[2,0],p3[2,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                        cv2.putText(faceROI,'.',(p3[3,0],p3[3,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)
                        cv2.putText(faceROI,'.',(p3[4,0],p3[4,1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1)

                        ##Get the hog features data by calling get_hog()
                        hog_data = get_hog(ROI_gray,p3,hog,winStride)

                        ##Define the average position of points in the sequence 
                        try:
                            avg_count = avg_count + 1 
                            avg_p3 = (((avg_count-1)*avg_p3) + p3)/avg_count
                        except:
                            avg_p3 = p3
                            avg_count =1

                        ##Define the initial average neutral feature vector
                        try:
                            ini_samp = np.vstack([ini_samp, hog_data])

                        except:
                            ini_samp = hog_data

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
        if fno == 50:
            break
    ##Close all the windows
    cap.release()
    cv2.destroyAllWindows()
    avg_ini = np.mean(ini_samp,axis=0)
    return ini_samp,avg_ini,avg_p3
