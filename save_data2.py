import numpy as np
import cv2
import dlib
import os
import glob
import sys



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





expressions =['Anger','Disgust','Fear','Happiness','Sadness','Surprise']

k=0
m=0
n=0

for exp in expressions:
    for f in glob.glob(os.path.join("Selected Cohn_Kanade images",exp,"*/*.png")):

        k=k+1

        if not k%8:
            p0=get_points(f)
            m=1

        if k%8 == 1:   
            p1 = get_points(f)
            n=1

        if m==1 and n==1:
            i=0
            m=0
            n=0
            d=np.array([])
            while i<9:
                d = np.append(d,(np.float32(p0[i][0]) - np.float32(p1[i][0]))/150)
                d = np.append(d,(np.float32(p0[i][1]) - np.float32(p1[i][1]))/150)
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



print samples_array.shape
np.save('s3_geo',samples_array)
#outFile = open('some.txt','wb')
#pickle.dump(samples_array,outFile)
#outFile.close()

print 'Done'
