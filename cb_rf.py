from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd


class cb_rf():
  
  def __init__(self,input_user):
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

  def recommend(self,itemsTopredict):
    # init regressor rf
    regr = RandomForestRegressor(n_estimators=100,criterion='mse', random_state=1)
    # train 
    regr.fit(self.rated_clothes, self.ratings)
    # predict
    pred_X = self.unrated_clothes.loc[itemsTopredict]
    y_pred = regr.predict(pred_X)
    # result dataframe
    df = pd.DataFrame()
    df['clothId'] = pred_X.index.tolist()
    df['rf_pred'] = y_pred
   
    return df

  def split_and_predict(self):
    # init regressor rf
    regr = RandomForestRegressor(n_estimators=80,max_depth=20,criterion='mse', random_state=1)
    # split data
    train_X, test_X, train_y, test_y = train_test_split(self.rated_clothes, self.ratings, test_size=0.1, random_state=1)
    # train 
    regr.fit(train_X,train_y)
    # predict
    y_pred = regr.predict(test_X)
   
    # result dataframe
    df = pd.DataFrame(columns=['clothId','y_pred','y_true'])
    df['clothId'] = test_X.index.tolist()
    df['y_true'] = test_y
    df['y_pred'] = y_pred

    return df
 