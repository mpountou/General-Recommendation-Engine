''' 
@author: Mpountou
@year: 2020-2021
'''

from keras.models import Sequential
from keras.layers import Dense,Dropout
import tensorflow as tf
from sklearn import preprocessing
class cb_nn():
  def __init__(self,input_user,features):
    self.features = features
    self.recdf = -1
    self.dataset = dataset
    self.evdf = -1
    self.columns = columns

    self.input_user = input_user



  def make_recommendation(self):

    if type(self.recdf) == type(pd.DataFrame()):
      return self.recdf
    
    # make local variables
    dataset = self.dataset
    columns = self.columns
    features = self.features

    # get unrated ratings per user
    all_users = dataset[columns[0]].unique()
    all_items = set(dataset[columns[1]].unique())

    # dataframe for recommendation
    df = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
    for i in all_users:
      u_data = dataset.loc[dataset[columns[0]] == i]
      train_X = features.loc[u_data[columns[1]].tolist()]
   
      train_y = u_data[columns[2]].tolist()
 
      model = self.model_init()
      earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min',restore_best_weights=True)
      reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
 
      history = model.fit(np.array(train_X.values), np.array(train_y), epochs=10,callbacks=[earlyStopping,reduce_lr_loss], validation_split=0.1)


      rated = set(u_data[columns[1]].unique())
      unrated = list(all_items - rated)
      if len(unrated)>0:
        test_X = features.loc[unrated]
        y_rec = model.predict(test_X)
        p_data = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
        p_data[columns[0]] = [int(i) for k in range(len(unrated))]
        p_data[columns[1]] = unrated
        p_data['y_rec'] = y_rec
        df = df.append(pd.DataFrame(data=p_data.values,columns=[columns[0],columns[1],'y_rec']))

    df = df.reset_index().drop(columns='index')
 
    self.recdf = df

    return df  

  def evaluate_system(self,all_train,all_test):
    if type( self.evdf) == type(pd.DataFrame()):
      return  self.evdf
    # make local variables
    dataset = self.dataset
    columns = self.columns
    features = self.features

    all_users = dataset[columns[0]].unique()

    df = pd.DataFrame(columns=[columns[0],columns[1],'y_true','y_pred'])

    for i in all_users:

      u_train = all_train.loc[all_train[columns[0]] == i]
      u_test = all_test.loc[all_test[columns[0]] == i]

      f_train = features.loc[u_train[columns[1]].tolist()]
      f_test = features.loc[u_test[columns[1]].tolist()]

      train_y = u_train[columns[2]].tolist()
      test_y = u_test[columns[2]].tolist()
   
      model = self.model_init()
      earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min',restore_best_weights=True)
      reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1, min_delta=1e-4, mode='min')
    
  
   
      history = model.fit(np.array(f_train.values), np.array(train_y), epochs=25,callbacks=[earlyStopping,reduce_lr_loss], validation_split=0.2)

      y_pred = model.predict(f_test.values)

  
      u_df = pd.DataFrame(columns=[columns[0],columns[1],'y_true','y_pred'])
      u_df[columns[0]] = u_test[columns[0]].tolist()
      u_df[columns[1]] = u_test[columns[1]].tolist()
      u_df['y_true'] = u_test[columns[2]].tolist()
      u_df['y_pred'] = y_pred

      df = df.append(u_df)
    
    df = df.reset_index().drop(columns='index')
    self.evdf = df
    return df




  def recommend(self,itemsTopredict):
    # init regressor rf
    model = self.model_init()
    # train 
    # train 
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min',restore_best_weights=True)
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
 
    history = model.fit(np.array(self.rated_clothes.values), np.array(self.ratings), epochs=15,callbacks=[earlyStopping,reduce_lr_loss], validation_split=0.1)

    # predict
    pred_X = self.unrated_clothes.loc[itemsTopredict]
    y_pred = model.predict(pred_X)
    # result dataframe
    df = pd.DataFrame()
    df['clothId'] = pred_X.index.tolist()
    df['nn_pred'] = y_pred
   
    return df

  def split_and_predict(self):
    # init regressor rf
    model = self.model_init()
    # split data
    train_X, test_X, train_y, test_y = train_test_split(self.rated_clothes, self.ratings, test_size=0.2, random_state=1)
    # train 
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min',restore_best_weights=True)
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

    history = model.fit(np.array(train_X.values), np.array(train_y), epochs=10,callbacks=[earlyStopping,reduce_lr_loss], validation_split=0.1)
    # predict
    y_pred = model.predict(test_X)
    
    # result dataframe
    df = pd.DataFrame(columns=['clothId','y_pred','y_true'])
    df['clothId'] = test_X.index.tolist()
    df['y_true'] = test_y
    df['y_pred'] = y_pred

    return df
     
  def model_init(self):
    # make local
    dataset = self.dataset
    columns = self.columns

    model = Sequential()
    model.add(Dense(2048, input_shape=(self.features.shape[1], ), activation='relu', name='dense_1'))
    model.add(Dropout(0.2))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(1024, activation='relu', name='dense_2'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu', name='dense_3'))
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu', name='dense_4'))
    model.add(Dense(1, activation='sigmoid', name='dense_output'))
    min_rating = min(dataset[columns[2]].tolist())
    max_rating = max(dataset[columns[2]].tolist())
    model.add(Lambda(lambda x:   (max_rating-min_rating)*x + min_rating ))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    return model

  def convert_index(self,old):
    id_db = pd.read_pickle('/content/drive/My Drive/id_db.pkl')
    new_list = id_db['id'].tolist()
    old_list = id_db['db_id'].tolist()
    new = []
    for j in range(len(old)):
      for i in range(len(id_db)):
        if old[j] in old_list[i]:
          new.append(new_list[i])
    return new
    
  def hyb_eval(self,train,test):
    # init regressor rf
    model = self.model_init()
    d_train = self.rated_clothes.copy()
    d_train['rating'] = self.ratings.copy()
    train_X = d_train.loc[train['clothId'].tolist()]
  
    train_y = train_X['rating'].tolist()
  
    train_X = train_X.drop(columns='rating')
    d_test = self.rated_clothes.copy()
    d_test['rating'] = self.ratings
    test_X = d_test.loc[test['clothId'].tolist()]
   
    test_y = test_X['rating'].tolist()
    
    test_X = test_X.drop(columns='rating')
    # train 
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min',restore_best_weights=True)
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

    history = model.fit(np.array(train_X.values), np.array(train_y), epochs=10,callbacks=[earlyStopping,reduce_lr_loss], validation_split=0.05)
    # predict
    y_pred = model.predict(test_X)
    
    # result dataframe
    df = pd.DataFrame(columns=['clothId','nn_pred'])
    df['clothId'] = test_X.index.tolist()
    #df['y_true'] = test_y
    df['nn_pred'] = y_pred

    return df

 

  def coverage(self,threshold):

    if type(self.recdf) != type(pd.DataFrame()):
      pred_ratings = self.make_recommendation()
    else:
      pred_ratings = self.recdf

    already_rated = len(self.dataset)
    
    high_rated = len(pred_ratings.loc[pred_ratings['y_rec']>threshold])
    
    low_rated = len(pred_ratings.loc[pred_ratings['y_rec']<=threshold])

    unrated = len(pred_ratings.loc[pred_ratings['y_rec']==np.nan])
    
    cov_df = pd.DataFrame()

    cov_df['recommended'] = [high_rated]

    cov_df['not recommended'] = [low_rated]

    cov_df['cannot recommended'] = [unrated]

    cov_df['already rated'] = [already_rated]
    
    return cov_df
    
  
    
  def novelty(self,cat_,threshold,translator_=False):

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