#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid


# In[14]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
from keras import backend as K
import tensorflow as tf


# In[15]:


import telegram
from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler 
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters


# In[16]:


updater = Updater("5396188533:AAHGs1nLTAwoFGvxyES27ssEUX8FRwYPwfE",use_context=True)
bot = telegram.Bot("5396188533:AAHGs1nLTAwoFGvxyES27ssEUX8FRwYPwfE")
def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello sir, Welcome to the Bot.")
    
def OpenDoor(update: Update, context: CallbackContext):
    update.message.reply_text("Door IS Opend")

def CloseDoor(update: Update, context: CallbackContext):
    update.message.reply_text("Door Is Closed")

def unknown(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Sorry '%s' is not a valid command" % update.message.text)

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('open', OpenDoor))
updater.dispatcher.add_handler(CommandHandler('close', CloseDoor))

updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))
updater.dispatcher.add_handler(MessageHandler(
    Filters.command, unknown))  # Filters out unknown commands
updater.start_polling()


# In[ ]:


def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


# In[18]:


class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# In[19]:


siamese_model = tf.keras.models.load_model('face2.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


# In[20]:


for image in os.listdir(os.path.join('application_data', 'verification_images')):
    validation_img = os.path.join('application_data', 'verification_images', image)
    print(validation_img)


# In[21]:


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        print(result)
        results.append(result)
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    
    # Verification Threshold: Proportion of positive predictions / total positive samples 
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    verified = verification > verification_threshold
    
    return verified


# In[22]:


def DetectFace(path):
        print(path)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
                faces = img[y:y + h, x:x + w]
                cv2.imwrite(path, faces)
            return 1
        else :
          return 0


# In[24]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         h, s, v = cv2.split(hsv)

#         lim = 255 - 10
#         v[v > lim] = 255
#         v[v <= lim] -= 10
        
#         final_hsv = cv2.merge((h, s, v))
#         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        #if DetectFace(os.path.join('application_data', 'input_image', 'input_image.jpg')):
        verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
        #else: 
         #   print("NO Face Detected")
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




