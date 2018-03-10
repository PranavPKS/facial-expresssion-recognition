import numpy as np


##Hardcoded Randomizing function for the database which has the 'Selected Cohn_Kanade images'

def randy_fn():    
    y1 = np.arange(43)
    np.random.shuffle(y1)
    
    y2 = np.arange(43,93)
    np.random.shuffle(y2)
    
    randy1 = np.append(y1[0:32],y2[0:37])
    randy2 = np.append(y1[32:43],y2[37:50])
    
    y3 = np.arange(93,147)
    np.random.shuffle(y3)
    
    randy1 = np.append(randy1,y3[0:40])
    randy2 = np.append(randy2,y3[40:54])
    
    y4 = np.arange(147,221)
    np.random.shuffle(y4)
    
    
    randy1 = np.append(randy1,y4[0:55])
    randy2 = np.append(randy2,y4[55:74])
    
    y5 = np.arange(221,272)
    np.random.shuffle(y5)
    
    
    randy1 = np.append(randy1,y5[0:38])
    randy2 = np.append(randy2,y5[38:51])
    
    y6 = np.arange(272,342)
    np.random.shuffle(y6)
    
    randy1 = np.append(randy1,y6[0:52])
    randy2 = np.append(randy2,y6[52:70])

    return randy1,randy2
