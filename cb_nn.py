from keras.models import Sequential
from keras.layers import Dense

class cb_nn():
  def __init__(self,input_user):
    self.input_user = input_user
    self.input_user = input_user
    self.df = pd.read_pickle('/content/drive/My Drive/multi-user5.pkl')
    self.df = self.df.loc[self.df['UserId'] == input_user]
    self.df = self.df.drop_duplicates(subset='ImageSource',keep='first')
    self.prodNo = self.df['ProductNo'].tolist()
    self.ratings = self.df['gradeUser'].tolist()
    self.all_clothes = pd.read_pickle('/content/drive/My Drive/clothes_attr.pkl')
    self.rated_clothes = self.all_clothes.copy()
    self.rated_clothes['ratings'] = self.ratings

    z = []
    for i in range(4325):
      z.append([])

    for i in range(10):
      if i != self.input_user:
        rating_data = pd.read_pickle('/content/drive/My Drive/multi-user5.pkl')
        u_data = rating_data.loc[rating_data['UserId'] == i]
        u_data = u_data.drop_duplicates(subset='ImageSource',keep='first')
        for j in range(len(u_data)):
          if u_data.iloc[j]['gradeUser']!=2.5:
            z[j].append(u_data.iloc[j]['gradeUser'])
    for i in range(len(z)):
      if len(z[i])>0:
        z[i] = sum(z[i]) / len(z[i])
      else:
        z[i] = -1
    self.rated_clothes['avg_rating'] = z
    self.unrated_clothes = self.rated_clothes.copy()
    self.unrated_clothes = self.unrated_clothes.loc[self.unrated_clothes['ratings'] ==2.5]
    self.rated_clothes = self.rated_clothes.loc[self.rated_clothes['ratings'] != 2.5]
    self.ratings = self.rated_clothes['ratings'].tolist()
    self.rated_clothes = self.rated_clothes.drop(columns='ratings')
    self.unrated_clothes = self.unrated_clothes.drop(columns='ratings')

  def split_and_predict(self):
    # init regressor rf
    model = self.model_init()
    # split data
    train_X, test_X, train_y, test_y = train_test_split(self.rated_clothes, self.ratings, test_size=0.1, random_state=1)
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
    model = Sequential()
    model.add(Dense(128, input_shape=(412, ), activation='tanh', name='dense_1'))
    model.add(Dropout(0.01))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(128, activation='tanh', name='dense_2'))
    model.add(Dropout(0.01))
    model.add(Dense(128, activation='tanh', name='dense_3'))
    model.add(Dense(1, activation='linear', name='dense_output'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

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
    df = pd.DataFrame(columns=['clothId','y_pred'])
    df['clothId'] = test_X.index.tolist()
    #df['y_true'] = test_y
    df['y_pred'] = y_pred

    return df