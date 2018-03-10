import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix


tot = np.load('tot_samp_new.npy')[342:744]
lab = np.load('tot_label.npy')[342:744]


def kfld():
    kf = KFold(402, n_folds=5,shuffle=True,)
    for train_index, test_index in kf:
        X_train, X_test = tot[train_index], tot[test_index]
        y_train, y_test = lab[train_index], lab[test_index]
    clf = svm.SVC(kernel='linear', C=100,decision_function_shape='ovr').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print cm_norm
    #mask = (y_pred == y_test)
    #correct = np.count_nonzero(mask)
    return cm_norm

i=0
conf_mat = np.array([])


while (i<10):
    t = kfld()
    try:
        conf_mat = conf_mat + t
    except:
        conf_mat = t
        
    #a = np.append(a,b)
    i=i+1

res = conf_mat/10
print res
#print np.mean(a)
