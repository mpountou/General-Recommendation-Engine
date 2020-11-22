import keras
import tensorflow as tf 
from keras.callbacks import ModelCheckpoint

class cf_deep():

  def __init__(self,preprocess_pack,handler,cols,user_input):
   
    self.user_input = user_input
    self.handler = handler
    self.cols = cols
    df , t_users, t_items , minmax , enc = preprocess_pack
    self.df = df
    self.t_users = t_users
    self.t_items = t_items
    self.min_rating , self.max_rating = minmax
    self.enc = enc
    self.model = cf_deep_model(t_users,t_items,5,self.min_rating,self.max_rating).model

  def convertInput(self,train,test):
    TRAIN = train.values
    TEST = test.values
    X_train = [TRAIN[:, 0], TRAIN[:, 1]]
    y_train = TRAIN[:, 2]
    X_test = [TEST[:, 0], TEST[:, 1]]
    y_test = TEST[:, 2]
    return X_train,y_train,X_test,y_test

  def recommend(self,itemsTopredict):
    train,test = self.handler.rec_split(self.df)
    X_train,y_train,X_test,y_test = self.convertInput(train,test)
    self.train(X_train,y_train,X_test,y_test)
    self.load_best_model()
    
    canTransform = self.enc.classes_
    itemstp = []
    for i in range(len(itemsTopredict)):
      if itemsTopredict[i] in canTransform:
        itemstp.append(itemsTopredict[i])
    predItems = self.enc.transform(itemstp)
    a = [self.user_input for i in range(len(predItems))]
    X_pred = [np.array(a), np.array(predItems)]
    y_pred = np.array(self.model.predict(X_pred))
    df = pd.DataFrame(columns=[self.cols[1],'dl_pred'])
    df[self.cols[1]] = self.enc.inverse_transform(predItems)
    df['dl_pred'] = y_pred
    c = df[self.cols[1]].tolist()
    for i in range(len(itemsTopredict)):
      if itemsTopredict[i] not in c:
        itemId = itemsTopredict[i]
        df = df.append(pd.DataFrame(data=[[itemId,np.nan]],columns=[self.cols[1],'dl_pred']))

    return df

  def split_and_predict(self):
    # split data to train and test set
    train,test = self.handler.split(self.df,self.user_input)
    X_train,y_train,X_test,y_test = self.convertInput(train,test)
    self.train(X_train,y_train,X_test,y_test)
    self.load_best_model()
    y_pred = np.array(self.model.predict(X_test))
    df = pd.DataFrame(columns=[self.cols[1],'y_true','y_pred'])
    df[self.cols[1]] = test[self.cols[1]].tolist()
    df['y_true'] = y_test
    df['y_pred'] = y_pred

    return df
    
  def train(self,X_train,y_train,X_test,y_test):
    filepath ="/content/drive/My Drive/models/weights_dl.hdf5"
    # Keep only a single checkpoint, the best over test accuracy.
    checkpoint = ModelCheckpoint(filepath,
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              mode='min')   

    self.model.compile(optimizer= 'adam', loss=tf.keras.losses.MeanAbsoluteError())

    train = self.model.fit(x=X_train, y=np.array(y_train),
            batch_size=32,
            epochs=10,
            verbose=1,
            validation_data=(X_test, np.array(y_test)),
            shuffle=True,
            callbacks=[checkpoint])
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(train.history) 
    # save to csv: 
    hist_csv_file = '/content/drive/My Drive/models/history_dl.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f) 

  def load_best_model(self):
    filepath ="/content/drive/My Drive/models/weights_dl.hdf5"
    self.model = cf_deep_model(self.t_users,self.t_items,5,self.min_rating,self.max_rating).model
    self.model.load_weights(filepath)
