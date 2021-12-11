from libsvm.svm import *
from libsvm.svmutil import svm_read_problem, svm_train, svm_predict
import scipy

trainy, trainx = svm_read_problem('training.txt', return_scipy=True)
testy, testx = svm_read_problem('test.txt', return_scipy=True)

# linear
m = svm_train(trainy, trainx, '-t 0')
p_label, p_acc, p_val = svm_predict(testy, testx, m)
print(f'Linear Kernel Accuracy: {p_acc}')


# polynomial
m = svm_train(trainy, trainx, '-t 1')
p_label, p_acc, p_val = svm_predict(testy, testx, m)
print(f'Polynomial Kernel Accuracy: {p_acc}')


# rbf
m = svm_train(trainy, trainx, '-t 2')
p_label, p_acc, p_val = svm_predict(testy, testx, m)
print(f'RBF Kernel Accuracy: {p_acc}')


# sigmoid
m = svm_train(trainy, trainx, '-t 3')
p_label, p_acc, p_val = svm_predict(testy, testx, m)
print(f'Sigmoid Kernel Accuracy: {p_acc}')
