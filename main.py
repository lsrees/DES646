from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils import *
from demo import *

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")         
import cv2

data_train = pd.read_csv("train_angle.csv")
data_test = pd.read_csv("test_angle.csv")

X, Y = data_train.iloc[:, :data_train.shape[1] - 1], data_train['target']

model = SVC(kernel='rbf', decision_function_shape='ovo',probability=True)
model.fit(X, Y)


predictions = evaluate(data_test, model, show=True)


cm = confusion_matrix(data_test['target'], predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


correct_feedback(model, 0, 'test_angle.csv')

cv2.destroyAllWindows()
