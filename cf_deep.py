''' 
@author: Mpountou
@year: 2020-2021
'''

import keras
import tensorflow as tf 
from keras.callbacks import ModelCheckpoint

class cf_deep():

  def __init__(self,preprocess_pack,handler,cols,user_input):
    self.evdf = -1
    self.user_input = user_input
    self.handler = handler
    self.cols = cols
    df , t_users, t_items , minmax , enc , enc2 = preprocess_pack
    self.df = df
    self.t_users = t_users
    self.t_items = t_items
    self.min_rating , self.max_rating = minmax
    self.enc = enc
    self.enc2 = enc2
    self.model = cf_deep_model(t_users,t_items,70,self.min_rating,self.max_rating).model
    self.recdf = -1
  def make_recommendation(self):
    # make local variables
    if type(self.recdf) == type(pd.DataFrame()):
      return self.recdf
    input_user = self.user_input
    data =   self.df
    cols =   self.cols

    if type(self.recdf)  == type(pd.DataFrame()):
      return self.recdf
   
    train,test = self.handler.rec_split(self.df)
   
    X_train,y_train,X_test,y_test = self.convertInput(train,test)

    X_train[0] = X_train[0].astype('float64')
    X_train[1] = X_train[1].astype('float64')
    X_test[0] = X_test[0].astype('float64')
    X_test[1] = X_test[1].astype('float64')
    y_train = np.array(y_train).astype('float64')
    y_test = np.array(y_test).astype('float64')
    
    self.train(X_train,y_train,X_test,y_test)
    
    self.load_best_model()

    train_items = np.unique(X_train[1])
    print(len(train_items))

    # get unrated ratings per user
    all_users = data[cols[0]].unique()
    all_items = set(data[cols[1]].unique())

    print(len(all_items))
    # dataframe for recommendation
    df = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
    for i in all_users:
      u_data = data.loc[data[cols[0]] == i]
      rated = set(u_data[cols[1]].unique())
      unrated = list(all_items - rated)
      if len(unrated) > 0 :
        a = [i for k in range(len(unrated))]
        X_pred = [np.array(a).astype('float64'), np.array(unrated).astype('float64')]
        y_rec = self.model.predict(X_pred)
        p_data = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
        p_data[columns[0]] = self.enc2.inverse_transform(a)
        p_data[columns[1]] = self.enc.inverse_transform(unrated)
        p_data['y_rec'] = y_rec
        df = df.append(p_data)


    df = df.reset_index().drop(columns='index')
 
    self.recdf = df
    return df

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

  def evaluate_system(self,all_train,all_test):
    if type( self.evdf) == type(pd.DataFrame()):
      return  self.evdf
    # make local 
    cols = self.cols
    df = self.df

    train = all_train.copy()
    test = all_test.copy()
    train[cols[0]] = self.enc2.transform(train[cols[0]].tolist())
    test[cols[0]] = self.enc2.transform(test[cols[0]].tolist())

    train[cols[1]] =  self.enc.transform(train[cols[1]].tolist())
    test[cols[1]] = self.enc.transform(test[cols[1]].tolist())

    # convert input for neural network
    X_train,y_train,X_test,y_test = self.convertInput(train,test)

    # fix datatype tensor error
    X_train[0] = X_train[0].astype('float64')
    X_train[1] = X_train[1].astype('float64')
    X_test[0] = X_test[0].astype('float64')
    X_test[1] = X_test[1].astype('float64')
    y_train = np.array(y_train).astype('float64')
    y_test = np.array(y_test).astype('float64')

    # train and load
    self.train(X_train,y_train,X_test,y_test)
    self.load_best_model()

    # create prediction dataframe
    y_pred = np.array(self.model.predict(X_test))
    df = pd.DataFrame(columns=[cols[0],cols[1],'y_true','y_pred'])
    df[cols[0]] =  self.enc2.inverse_transform(test[cols[0]].tolist())
    df[cols[1]] =  self.enc.inverse_transform(test[cols[1]].tolist())
    df['y_true'] = y_test
    df['y_pred'] = y_pred

    # reset index  
    df = df.reset_index().drop(columns='index')
    self.evdf = df
    return df
  def split_and_predict(self):
    # split data to train and test set
    train,test = self.handler.split(self.df,self.user_input,0.2)
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

    self.model.compile(optimizer= 'adam', loss=tf.keras.losses.MeanSquaredError())
    #MeanSquaredError
    train = self.model.fit(x=X_train, y=np.array(y_train),
            batch_size=256,
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
    self.model = cf_deep_model(self.t_users,self.t_items,70,self.min_rating,self.max_rating).model
    self.model.load_weights(filepath)

  def hyb_eval(self,train,test):

    itemsTopredict = test[self.cols[1]].tolist()

    #print(test[self.cols[1]])
    u_test = test.copy()
    u_test[self.cols[1]] = self.enc.transform(test[self.cols[1]].tolist() )
    #print(test[self.cols[1]])
    u_train = train.copy()
    u_train[self.cols[1]] = self.enc.transform(train[self.cols[1]].tolist() )

    X_train,y_train,X_test,y_test = self.convertInput(u_train,u_test)


    X_train[0] = X_train[0].astype('float64')
    X_train[1] = X_train[1].astype('float64')
    X_test[0] = X_test[0].astype('float64')
    X_test[1] = X_test[1].astype('float64')
    y_train = np.array(y_train).astype('float64')
    y_test = np.array(y_test).astype('float64')

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

  def coverage(self,threshold=5):

    if type(self.recdf)  != type(pd.DataFrame()):
      pred_ratings = self.make_recommendation()
    else:
      pred_ratings = self.recdf

    already_rated = len(self.df)
    
    high_rated = len(pred_ratings.loc[pred_ratings['y_rec']>threshold])
    
    low_rated = len(pred_ratings.loc[pred_ratings['y_rec']<=threshold])

    unrated = len(pred_ratings.loc[pred_ratings['y_rec']==np.nan])
    
    cov_df = pd.DataFrame()

    cov_df['recommended'] = [high_rated]

    cov_df['not recommended'] = [low_rated]

    cov_df['cannot recommended'] = [unrated]

    cov_df['already rated'] = [already_rated]
    
    return cov_df
    
 
 
  def novelty(self,cat_,threshold=5,translator_=False):

    if type(self.recdf) != type(pd.DataFrame()):
      pred_ratings = self.make_recommendation()
    else:
      pred_ratings = self.recdf

    pred_ratings = pred_ratings.merge(cat_,on=columns[1],how='inner')

    categories = pred_ratings['category'].unique()

    c_ratings = []
    for i in range(len(categories)):
      ratings = []
      fr = pred_ratings.loc[pred_ratings['category'] == categories[i]]
      ratings.append(round(len(fr.loc[fr['y_rec'] >=threshold])  / len(fr.loc[fr['y_rec'] >=0]),2) )
      ratings.append(round(len(fr.loc[fr['y_rec'] <threshold])  / len(fr.loc[fr['y_rec'] >=0]) ,2))
      c_ratings.append(ratings)

    df = pd.DataFrame(data=c_ratings , columns=['προτείνεται','δεν προτείνεται'])
    if type(translator_) == bool:
      return df

    categories_gr = []
 
    for i in range(len(categories)):
      categories_gr.append(translator_.loc[translator_['category'] == categories[i]].index.tolist()[0])
    df['κατηγορίες'] = categories_gr

    df = df.set_index(keys='κατηγορίες')

    return df
    #4.5