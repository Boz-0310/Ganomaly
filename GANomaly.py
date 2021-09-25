# In[]
import imp
import time
from keras import layers
import keras
import keras.backend as K
import tensorflow as tf
# In[] Models
niter = 600
bz = 16
width = 256
height = 256
channels = 3
show_summary=0
# In[] Generators Encoder
input_layer = layers.Input(name='input', shape=(height, width, channels))

# Encoder
x = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='conv_1', kernel_regularizer = 'l2')(input_layer)
x = layers.LeakyReLU(name='leaky_1')(x)

x = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='conv_2', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_1')(x)
x = layers.LeakyReLU(name='leaky_2')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_3', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_2')(x)
x = layers.LeakyReLU(name='leaky_3')(x)


x = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_4', kernel_regularizer = 'l2')(x)
x = layers.BatchNormalization(name='norm_3')(x)
x = layers.LeakyReLU(name='leaky_4')(x)

x = layers.GlobalAveragePooling2D(name='g_encoder_output')(x)

g_e = keras.models.Model(inputs=input_layer, outputs=x)

if show_summary==1:
    g_e.summary()

# In[] Generator
input_layer = layers.Input(name='input', shape=(height, width, channels))

x = g_e(input_layer)

y = layers.Dense(height * width * 2, name='dense')(x) # 2 = 128 / 8 / 8
y = layers.Reshape((height//8, width//8, 128), name='de_reshape')(y)

y = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', name='deconv_1', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_1')(y)

y = layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', name='deconv_2', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_2')(y)

y = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', name='deconv_3', kernel_regularizer = 'l2')(y)
y = layers.LeakyReLU(name='de_leaky_3')(y)

y = layers.Conv2DTranspose(channels, (1, 1), strides=(1,1), padding='same', name='decoder_deconv_output', kernel_regularizer = 'l2', activation='tanh')(y)

g = keras.models.Model(inputs=input_layer, outputs=y)
if show_summary==1:
    g.summary()
# In[] Encoder
input_layer = layers.Input(name='input', shape=(height, width, channels))

z = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='encoder_conv_1', kernel_regularizer = 'l2')(input_layer)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='encoder_conv_2', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_1')(z)
z = layers.LeakyReLU()(z)


z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='encoder_conv_3', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_2')(z)
z = layers.LeakyReLU()(z)

z = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='conv_41', kernel_regularizer = 'l2')(z)
z = layers.BatchNormalization(name='encoder_norm_3')(z)
z = layers.LeakyReLU()(z)

z = layers.GlobalAveragePooling2D(name='encoder_output')(z)

encoder = keras.models.Model(input_layer, z)
if show_summary==1:
    encoder.summary()
# In[] feature extractor
input_layer = layers.Input(name='input', shape=(height, width, channels))

f = layers.Conv2D(32, (5,5), strides=(1,1), padding='same', name='f_conv_1', kernel_regularizer = 'l2')(input_layer)
f = layers.LeakyReLU(name='f_leaky_1')(f)

f = layers.Conv2D(64, (3,3), strides=(2,2), padding='same', name='f_conv_2', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_1')(f)
f = layers.LeakyReLU(name='f_leaky_2')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_3', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_2')(f)
f = layers.LeakyReLU(name='f_leaky_3')(f)


f = layers.Conv2D(128, (3,3), strides=(2,2), padding='same', name='f_conv_4', kernel_regularizer = 'l2')(f)
f = layers.BatchNormalization(name='f_norm_3')(f)
f = layers.LeakyReLU(name='feature_output')(f)

feature_extractor = keras.models.Model(input_layer, f)
if show_summary==1:
    feature_extractor.summary()
# In[] gan trainer
class AdvLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class CntLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
class EncLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(g_e(ori) - encoder(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)
    
# model for training
input_layer = layers.Input(name='input', shape=(height, width, channels))
gan = g(input_layer) # g(x)

adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cnt_loss = CntLoss(name='cnt_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])

gan_trainer = keras.models.Model(input_layer, [adv_loss, cnt_loss, enc_loss])

# loss function
def loss(yt, yp):
    return yp

losses = {
    'adv_loss': loss,
    'cnt_loss': loss,
    'enc_loss': loss,
}

lossWeights = {'cnt_loss': 1.0, 'adv_loss': 1.0, 'enc_loss': 1.0}

# compile
gan_trainer.compile(optimizer = 'adam', loss=losses, loss_weights=lossWeights)
if show_summary==1:
    gan_trainer.summary()

# In[] discriminator
input_layer = layers.Input(name='input', shape=(height, width, channels))

f = feature_extractor(input_layer)

d = layers.GlobalAveragePooling2D(name='glb_avg')(f)
d = layers.Dense(1, activation='sigmoid', name='d_out')(d)
    
d = keras.models.Model(input_layer, d)
if show_summary==1:
    d.summary()

d.compile(optimizer='adam', loss='binary_crossentropy')

# In[] Load Data
# from keras.datasets import mnist
import cv2
import numpy as np
import custom_dataset

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
data_train, data_test = custom_dataset.custom_data
x_train, y_train = data_train
x_test, y_test = data_test

x_ok = x_train[y_train == 0]  
x_test = x_test[((y_test == 0) | (y_test == 1))] 
y_test = y_test[(y_test == 0) | (y_test == 1)]

def reshape_x(x):
    if(channels != 1):
        new_x = np.empty((len(x), height,width,channels))
    else:
        new_x = np.empty((len(x), height,width))
    for i, e in enumerate(x):
        new_x[i] = cv2.resize(e, ( width,height))
    if(channels != 1):
        return new_x / 127 - 1 # /127 -1 normalization 
    else:
        return np.expand_dims(new_x, axis=-1) / 127 - 1 #expand_dims多一個維度
  
x_ok = reshape_x(x_ok)
x_test = reshape_x(x_test)

x_ok.max()
# In[] Training
def get_data_generator(data, batch_size=32):
    datalen = len(data)
    cnt = 0
    while True:
        idxes = np.arange(datalen)
        np.random.shuffle(idxes)
        cnt += 1
        for i in range(int(np.ceil(datalen/batch_size))):
            train_x = np.take(data, idxes[i*batch_size: (i+1) * batch_size], axis=0)
            y = np.ones(len(train_x))
            yield train_x, [y, y, y]

train_data_generator = get_data_generator(x_ok, bz)

for i in range(niter):
    start_time = time.time()
    ### get batch x, y ###
    x, y = train_data_generator.__next__()
        
    ### train disciminator ###
    d.trainable = True
        
    fake_x = g.predict(x)
        
    d_x = np.concatenate([x, fake_x], axis=0)
    d_y = np.concatenate([np.zeros(len(x)), np.ones(len(fake_x))], axis=0)
        
    d_loss = d.train_on_batch(d_x, d_y)

    ### train generator ###
    
    d.trainable = False        
    g_loss = gan_trainer.train_on_batch(x, y,return_dict=True)
    if(type(g_loss)==dict):
        for k in g_loss.keys():
            g_loss[k]=round(g_loss[k], 4)

    speed =time.time() - start_time
    print(f'niter: {i+1}, G_loss: {g_loss}, D_loss: {d_loss:10.4f}, time: {speed:10.4f}s')
    
# In[] Evaluation
encoded = g_e.predict(x_test) #輸入圖像產生的編碼
gan_x = g.predict(x_test)  #Generator產生的圖形
encoded_gan = g_e.predict(gan_x) #Generator產生的圖形所產生的編碼
score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
score = (score - np.min(score)) / (np.max(score) - np.min(score)) # map to 0~1

from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 5
plt.scatter(range(len(x_test)), score, c=['skyblue' if x == 1 else 'pink' for x in y_test])
# In[]
i = 0 # or 1
fig,ax = plt.subplots(1, 2)
image = np.reshape(gan_x[i:i+1], (height,width,channels))
image = image * 127 + 127
ax[0].set_title("Generator")
ax[0].imshow(image.astype(np.uint8), cmap='gray')
image = np.reshape(x_test[i:i+1], (height,width,channels))
image = image * 127 + 127
ax[1].set_title("Input Image(label:"+str(y_test[i])+")")
ax[1].imshow(image.astype(np.uint8), cmap='gray')

# %%
from sklearn.metrics import f1_score, precision_score,roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from inspect import signature


# %%
import itertools

def plot_roc_curve(fpr, tpr, name_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(name_model+'_roc_curve.png')
    plt.show()
    # plt.close() 

def roc(labels, scores, name_model):
    """Compute ROC curve and ROC area for each class"""
    roc_auc = dict()
    # True/False Positive Rates.
    fpr, tpr, threshold = roc_curve(labels, scores)
    # print("threshold: ", threshold)
    roc_auc = auc(fpr, tpr)
    # get a threshod that perform very well.
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    # draw plot for ROC-Curve
    plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_threshold

auc_out, threshold = roc(y_test, score, 'name_model')


predictions = (score > threshold).astype(int)
cm = tf.math.confusion_matrix(labels=y_test, predictions=predictions).numpy()
TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]




diagonal_sum = cm.trace()
sum_of_all_elements = cm.sum()
print("auc: ", auc_out)
print("threshold: ", threshold)
print("Accuracy: ", diagonal_sum / sum_of_all_elements )
print("False Alarm Rate: ", FP/(FP+TP))
print("Leakage Rate: ", FN/(FN+TN))
print("precision_score: ",precision_score(y_test, predictions))
print("recall_score: ", TP/(TP+FN))
#         F1 = 2 * (precision * recall) / (precision + recall)
print("F1-Score: ", f1_score(y_test, predictions))
# %%
def pre_rec_curve(labels, scores, show=False):
    average_precision = average_precision_score(labels, scores)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    if show:
        precision, recall, _ = precision_recall_curve(labels, scores)
        step_kwargs = ({
            'step': 'post'
        } if 'step' in signature(plt.fill_between).parameters else {})
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall,
                         precision,
                         alpha=0.2,
                         color='b',
                         **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        plt.savefig("pre_rec_curve.png")
#         plt.close()
    return {'average_precision': average_precision}

pre_rec_curve=pre_rec_curve(y_test, score, show=True)
#%%
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'_cm.png')
    # plt.close()
class_names = ["normal", "defect"]
plot_confusion_matrix(cm, class_names, title='Confusion matrix')