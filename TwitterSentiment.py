import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import datetime,os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from tqdm import tqdm
import preprocessor as processor


#To run code on CPU only uncomment the following line:
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

#next 3 lines to accomodate exceptions with embedddings
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

TWEET_FILE="data/tweets.csv"
TWEET_FILE_BALANCED="data/tweets_balanced.csv"
EMBEDDINGS_FOLDER="data/universal-sentence-encoder-multilingual-large_3"
MODEL_FILE ="data/model.h5"

#################################################################################################################################################################
#wordcloud picture
#################################################################################################################################################################
def plot_wc(data):
  text = data['tweet_text'].values
  stopwords = set(STOPWORDS)
  stopwords.update(["https", "co"])
  wordcloud = WordCloud(stopwords=stopwords, background_color="white",mode='RGB',width=1000,height=1000,max_words=1000,random_state=1, contour_width=1, contour_color='steelblue').generate(str(text))
  plt.figure(figsize=(10, 10))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()

#################################################################################################################################################################
#generate a file with a balanced number of positives,negatives and neutral tweets
#################################################################################################################################################################
def generate_balanced_dataset(plot=True):
  tw_file_bal=Path(TWEET_FILE_BALANCED)
  if tw_file_bal.is_file():
    df=pd.read_csv(TWEET_FILE_BALANCED, sep=';')
    df=df.dropna()
    df=shuffle(df)
    if plot:
      plot_wc(df)
    return df
  else:
    tw_file=Path(TWEET_FILE)
    if not tw_file.is_file():
      raise Exception("Sorry, file {0} is missing".format(TWEET_FILE)) 
    print("Generating file {0}".format(TWEET_FILE_BALANCED))
    df=pd.read_csv(TWEET_FILE, sep=';')
    df=df.dropna()
    df=shuffle(df)
    if plot:
      plot_wc(df)
    tot_len=df.shape[0]#num of rows
    #select rows with sentiment neutral (0)
    df_neu = df.loc[df['sentiment']=='NEUTRAL'].head(tot_len)
    len_neu=len(df_neu)
    #select rows wwith sentiment positive (1)
    df_pos = df.loc[df['sentiment']=='POSITIVE'].head(tot_len)
    len_pos=len(df_pos)
    #select rows wwith sentiment negattive (-1)
    df_neg = df.loc[df['sentiment']=='NEGATIVE'].head(tot_len)
    len_neg=len(df_neg)
    if plot:
      y=[len_neg,len_neu,len_pos]
      x=["Negatives","Neutral","Positives"]
      plt.bar(x,y,color=['red', 'blue', 'green'])
      plt.show()
    min_val=min(len_neu,len_pos,len_neg)
    print ("Each cathegory of tweets (positive,negative,neutral) will have {0} rows. {1} in total".format(min_val,min_val*3))
    #equalize dataset to the min value per cathegory
    df_all=pd.concat([df_neu.head(min_val),df_pos.head(min_val),df_neg.head(min_val)])
    df_all=shuffle(df_all)
    df_all.to_csv(TWEET_FILE_BALANCED, sep=';',index=False)
    if plot:
      plot_wc(df_all)
      y=[min_val,min_val,min_val]
      plt.bar(x,y,color=['red', 'blue', 'green'])
      plt.show()
    return df_all
#################################################################################################################################################################
#embed twwets
#################################################################################################################################################################
def embed_tweets(x,embeddings):
  x_emb = []
  for r in tqdm(x):
    r=processor.clean(r)
    emb = embeddings(r)
    reshaped_emb = tf.reshape(emb, [-1]).numpy()
    x_emb.append(reshaped_emb)
  x_emb = np.array(x_emb)
  return x_emb

def train(x,y):
  #Training dataset – The part of data that is used for model fitting
  #Validation dataset – The part of data used for tuning hyperparameters during training
  #Test dataset – The part of data used for evaluating a model’s performance after its training
  train_ratio = 0.90
  validation_ratio = 0.05
  test_ratio = 0.05
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 - train_ratio)
  x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
  #proceed to tweet embeddings
  embeddings = hub.KerasLayer(EMBEDDINGS_FOLDER)
  x_train_embed=embed_tweets(x_train,embeddings)
  x_test_embed=embed_tweets(x_test,embeddings)
  x_val_embed=embed_tweets(x_val,embeddings)
  print(x_train_embed.shape, y_train.shape)
  #Defining the NN model
  #We will create a ffive-layer deep learning network model.
  #Layer  Nodes
  #1       128 Dense layer
  #2       regularization layer dropping NN 50% weights at each epoch
  #3       64 Dense layer
  #4       regularization layer dropping NN 50% weights at each epoch
  #5       3 Dense layer wwitth an output per catheogory (neg,neutr,pos)

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(128, activation = 'relu',input_shape = (x_train_embed.shape[1],)))
  model.add(tf.keras.layers.Dropout(rate=0.5))
  model.add(tf.keras.layers.Dense(64, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(rate=0.5))
  model.add(tf.keras.layers.Dense(3, activation = 'softmax'))
  model.summary()
  model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics=['accuracy'])
  #To analyze the network performance we need to define a callback function which will be called at each epoch during training. We will be collecting the logs in the log folder.
  logdir = os.path.join("log",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir,histogram_freq = 1)
  #early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=1)
#Model training
  r = model.fit(x_train_embed, y_train, batch_size = 32, epochs = 20,verbose = 1,
    validation_data = (x_val_embed, y_val),callbacks = [tensorboard_callback])
  model.save(MODEL_FILE)
  #evaluation
  test_scores = model.evaluate(x_test_embed, y_test)
  print('Test Loss: ', test_scores[0])
  print('Test accuracy: ', test_scores[1] * 100)
  plt.plot(r.history['accuracy'], label='accuracy')
  plt.plot(r.history['val_accuracy'], label='val_acc')
  plt.plot(r.history['loss'], label='loss')
  plt.plot(r.history['val_loss'], label='val_loss')
  plt.legend()
  plt.show()
  return model

#################################################################################################################################################################
#main
#################################################################################################################################################################

try:
  model={}
  model_file=Path(MODEL_FILE)
  if not model_file.is_file():
    df=generate_balanced_dataset()
    x = df['tweet_text']
    y= pd.get_dummies(df['sentiment'])
    print(y)
    model=train(x,y)
  else:
    model = tf.keras.models.load_model(MODEL_FILE)
    model.summary()
  embeddings = hub.KerasLayer(EMBEDDINGS_FOLDER)
  #Predict on unseen Data
  neg_tweet="Andatevene tutti , maledetti mercenari, fate schifo non meritate quella maglia , siete una vergogna"
  neg_tweet=processor.clean(neg_tweet)
  print(neg_tweet)
  emb = embeddings(neg_tweet)
  pred=model.predict(emb)
  print(pred)

  pos_tweet="Forza e bellezza....  così insieme non si erano mai viste"
  pos_tweet=processor.clean(pos_tweet)
  print(pos_tweet)
  emb = embeddings(pos_tweet)
  pred=model.predict(emb)
  print(pred)

  neu_tweet="@sport Lo stadio Noucamp riaprirà al pubblico dopo un anno di restauri. #noucamp"
  neu_tweet=processor.clean(neu_tweet)
  print(neu_tweet)
  emb = embeddings(neu_tweet)
  pred=model.predict(emb)
  print(pred)
except Exception as ex:
  print(ex) 





