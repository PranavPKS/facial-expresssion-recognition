import numpy as np
from sklearn import svm
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


tot = np.load('tot_samp_new.npy')[0:342]
lab = np.load('tot_label.npy')[0:342]
target_names = np.array(['Anger','Disgust','Fear','Happiness','Sadness','Surprise'])

kf = KFold(342, n_folds=5,shuffle=True,)
for train_index, test_index in kf:
    X_train, X_test = tot[train_index], tot[test_index]
    y_train, y_test = lab[train_index], lab[test_index]
clf = svm.SVC(kernel='linear', C=100,decision_function_shape='ovr').fit(X_train, y_train)
y_pred = clf.predict(X_test)
#mask = (y_pred == y_test)
#correct = np.count_nonzero(mask)
#print correct*100/y_test.shape[0]

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.18)


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
