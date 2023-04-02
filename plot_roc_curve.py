from keras.models import load_model
from keras.models import Model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from keras.utils import to_categorical
from Data_Generator import *
import matplotlib.pyplot as plt
classes={
         0 :" No Diabtic_Retinopathy",
         1 :" Mild",
         2 : "Moderate",
         3 : "Severe",
         4 :"Proliferative DR"}


#loadinf model's weights
#load the modeks weight after training , then save the model in suitbale name in format .h5 
#lets assume that model was save as following : transfer_learning_inception.h5
model=load_model("transfer_learning_inception.h5")
model.summary()

y_true = validation_generator.classes
num_classes=validation_generator.num_classes
y_true = to_categorical(y_true, num_classes=num_classes)
print("shape of Y_true", y_true.shape)
print(validation_generator.classes, validation_generator.class_indices)

# Get the predicted probabilities for the validation set
y_pred = model.predict(validation_generator)

y_test=y_true
pred=y_pred
score = round(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)),2)
print(score)
report = classification_report(y_test.argmax(axis=1), pred.argmax(axis=1))
print(report)

print(y_pred.shape)
# Calculate the FPR, TPR, and threshold values for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 5
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
         ''.format(roc_auc["micro"]))
for i in range(n_classes):
    # plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
    plt.plot(fpr[i], tpr[i], lw=lw, label="ROC"+classes[i] + "(area = {1:0.2f})"
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()
