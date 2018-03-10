import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold


tot = np.load('tot_samp_new.npy')[342:744]
lab = np.load('tot_label.npy')[342:744]
#a = np.array([])

def kfld():
    kf = KFold(402, n_folds=5,shuffle=True,)
    for train_index, test_index in kf:
        X_train, X_test = tot[train_index], tot[test_index]
        y_train, y_test = lab[train_index], lab[test_index]
    clf = svm.SVC(kernel='linear', C=100,decision_function_shape='ovr').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #mask = (y_pred == y_test)
    #correct = np.count_nonzero(mask)
    return y_test,y_pred

i=0
test_lab=np.array([])
pred_lab=np.array([])

while (i<10):
    t,p = kfld()
    #b=kfld()
    try:
        test_lab = np.vstack([test_lab,t])
        pred_lab = np.vstack([pred_lab,p])
    except:
        test_lab = t
        pred_lab = p
        
    #a = np.append(a,b)
    i=i+1

print pred_lab.shape
#print np.mean(a)
