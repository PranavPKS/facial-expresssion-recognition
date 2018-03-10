import numpy as np
import cv2
import dlib
import os
import glob
#from skimage import io
import pickle
import sys
import scipy.io



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_points(f):
    
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
            
    cv2.imshow('norm_img',norm_img)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        sys.exit()


    return p0 





#os.chdir('RafD Frontal Faces')
#mat = scipy.io.loadmat('dataRaFD.mat')

neut = []
expr = []

for i,f in enumerate(glob.glob("RafD Frontal Faces/*.jpg")):


    if not (i-4)%7:
        neut.append(get_points(f))
    else:
        expr.append(get_points(f))


print len(neut)
print len(expr)
j=0

while j<len(expr):
    i=0
    d=np.array([])
    while i<9:
        d = np.append(d,(np.float32(expr[j][i][0]) - np.float32(neut[j/6][i][0]))/150)
        d = np.append(d,(np.float32(expr[j][i][1]) - np.float32(neut[j/6][i][1]))/150)
        i=i+1


    try:
        samples_array = np.vstack([samples_array, d])

    except:
        samples_array = d
        print d.shape
        print d
        print sys.exc_info()[0]
        print 'except'

        
    if cv2.waitKey(100) & 0xFF == ord('q'):
        sys.exit()

    j=j+1
print samples_array.shape
np.save('r3_geo',samples_array)
#outFile = open('some.txt','wb')
#pickle.dump(samples_array,outFile)
#outFile.close()

print 'Done'
