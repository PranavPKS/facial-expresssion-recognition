import math
import dlib
import numpy as np
import cv2

#Initialize all variables and objects used  
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def dist_pts(point1,point2):

    """ Given the points as input, returns the distance between them """
    distance = math.hypot(point1[0] - point2[0] , point1[1] - point2[1])
    return distance


def get_d(p0,p1):
    """ Given two set points, returns the differences between them as an array """
    i=0
    d=np.array([])
    while i<p0.shape[0]:
        d = np.append(d,(np.float32(p0[i][0]) - np.float32(p1[i][0]))/150)
        d = np.append(d,(np.float32(p0[i][1]) - np.float32(p1[i][1]))/150)
        i=i+1
    return d
    
    
def smile(ref,p0,th1,th2):
    """
    Input: the two array of points and the thresholds 
    Output: returns 1 for a smile and 0 for a non-smile
    """
    
    lu = ref[0,1] - p0[0,1] #movement of left corner in upward direction
    ru = ref[1,1] - p0[1,1] #movement of right corner in upward direction
    nu = ref[2,1] - p0[2,1] #movement of nose tip in upward direction
    
    # change of lip width    
    refLipWidth = ref[1,0] - ref[0,0]
    updatedLipWidth = p0[1,0] - p0[0,0]
    lipWidthChange = updatedLipWidth - refLipWidth
    
    # lu and ru - relative lip corner motion
    lu = lu - nu
    ru = ru - nu

    if lipWidthChange>th2 and lu>th1 and ru>th1:
        return 1
    else:
        return 0


    
def detect_face_n_landmarks(frame):   
    """
    Input: nd-array of the image
    Output: returns the points in an array p0
    """
    dets = detector(frame, 0)
    
    if not dets:
        return np.array([]),0
    else:
        temp = []
        for d in dets:
            temp.append(d.height())

        d1 = dets[int(np.argmax(temp))]

        shape = predictor(frame, d1)

        ##48 -left lip corner, 54-right lip corner, 33-nose tip, 21 and 22 are the inner eyebrow corners
        ##31 and 35 are nose bottom corners, 57-bottom lip center, 27-nose top tip
        ##add all required points by concatenating
        p0 = np.array([[shape.part(48).x, shape.part(48).y]])
        p0 = np.concatenate((p0, [[shape.part(54).x, shape.part(54).y]]), axis=0)
        p0 = np.concatenate((p0, [[shape.part(33).x, shape.part(33).y]]), axis=0)
        p0 = np.concatenate((p0, [[shape.part(21).x, shape.part(21).y]]), axis=0)
        p0 = np.concatenate((p0, [[shape.part(22).x, shape.part(22).y]]), axis=0)
        p0 = np.concatenate((p0, [[shape.part(31).x, shape.part(31).y]]), axis=0)
        p0 = np.concatenate((p0, [[shape.part(35).x, shape.part(35).y]]), axis=0)
        p0 = np.concatenate((p0, [[shape.part(57).x, shape.part(57).y]]), axis=0)
        p0 = np.concatenate((p0, [[shape.part(27).x, shape.part(27).y]]), axis=0)
        
        p0 = p0.astype(np.float32)
        return p0,d1.height()

def cropFaceROI (frame,nosePos,faceSize):
    """
    input: frame,nose coordianates, face_height
    output: the nd-array of the region of interest
    """
    f = faceSize/2
    ltx = nosePos[0] - f
    rtx = nosePos[0] + f
    lty = nosePos[1] - (1.25*f)
    lby = nosePos[1] + f
    
    faceROI = frame[lty:lby,ltx:rtx]
    return faceROI


def get_hog(norm_img,p0,hog,winStride):

    """
    input: nd-array of the image
    output: returns the HOG data of the given patches in an appended format 
    """

    ##c1 and c2 are the constants that are set to get the required size of those patches
    c1=16
    c2=20
    p0 = p0.astype(int)
    left_lip_patch = cv2.resize(norm_img[p0[0,1]- c1: p0[0,1]+c1, p0[0,0]-c1 : p0[0,0]+c1],(32,32))
    right_lip_patch = cv2.resize(norm_img[p0[1,1]-c1 : p0[1,1]+c1, p0[1,0]-c1 : p0[1,0]+c1],(32,32))
    betw_ibrows = cv2.resize(norm_img[p0[3,1]-c1 : p0[3,1]+c1, p0[3,0]-c2 : p0[4,0]+c2], (40,32))
    left_cheek_patch = cv2.resize(norm_img[p0[5,1]-int(1.25*c1) : p0[5,1]+ int(0.75*c1) , p0[5,0]-(2*c2)-5 : p0[5,0]-5], (40,32))
    right_cheek_patch = cv2.resize(norm_img[p0[6,1]- int(1.25*c1) : p0[6,1]+ int(0.75*c1) , p0[6,0]+5 : p0[6,0]+(2*c2)+5], (40,32))
    below_lip_patch = cv2.resize(norm_img[p0[7,1]-int(0.75*c1): p0[7,1]+int(1.25*c1), p0[7,0]-c2 : p0[7,0]+c2], (40,32))
    nose_patch = cv2.resize(norm_img[p0[8,1]: p0[8,1]+int(1.5*c1), p0[8,0]-int(0.75*c1) : p0[8,0]+int(0.75*c1)], (32,32))
    
    temp_list = [left_lip_patch, right_lip_patch, betw_ibrows, left_cheek_patch, right_cheek_patch, below_lip_patch, nose_patch]

    ##append all the hog features of those patches into hog_data    
    hog_data=[]
    hi=0
    while hi<len(temp_list):
        hog_data = np.append(hog_data, hog.compute(temp_list[hi],winStride).T)
        hi = hi+1
    hog_data = hog_data.astype(np.float32)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()
    return hog_data
