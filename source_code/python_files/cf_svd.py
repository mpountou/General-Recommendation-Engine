''' 
@author: Mpountou
@year: 2020-2021
'''

import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

class cf_svd():

  def __init__(self,evaluator_,input_user,dataset,columns):
    self.input_user = input_user
    self.evaluator_ = evaluator_
    self.dataset = dataset
    self.columns = columns
    self.evdf = -1
    self.all_ev = -1
    self.recdf = -1
    self.feature_num = [5,25,50]
    self.best_feat = -1
    
  

  def make_recommendation(self):

    if type( self.recdf) == type(pd.DataFrame()):
      return  self.recdf

    if self.best_feat == -1:
      print('BEST_FEAT IS -1 -> EVALUATE SYSTEM FIRST')
      self.best_feat = 30

    # make local variables
    input_user = self.input_user
    data =   self.dataset
    cols =   self.columns
    best_feat = self.best_feat

    # user item rating matrix
    matrix =  data_handler.create_matrix(dataset=data,columns=cols,fill_unrated_with=np.nan)
    # prediction ratings
    svd_features = int(len(matrix.index.tolist())/1.5)
    pred_mat = pd.DataFrame(columns= matrix.columns.tolist(),data = self.svd(matrix, k=svd_features))
    # get unrated ratings per user
    all_users = data[cols[0]].unique()
    all_items = set(data[cols[1]].unique())
    # dataframe for recommendation
    df = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
    for i in all_users:
      u_data = data.loc[data[cols[0]] == i]
      rated = set(u_data[cols[1]].unique())
      unrated = all_items - rated
      for j in unrated:
        df = df.append(pd.DataFrame(data=[[int(i),int(j),pred_mat[j][i]]],columns=[columns[0],columns[1],'y_rec']))

    df = df.reset_index().drop(columns='index')
 
    self.recdf = df
    return df  

  def svd(self,train, k):
    # get matrix values
    matrix = np.array(train)
    # mask nan or unavailable values
    mask = np.isnan(matrix)
    masked_arr = np.ma.masked_array(matrix, mask)
    # get means for every item
    item_means = np.mean(masked_arr, axis=0)    
    # replace nan with average
    matrix = masked_arr.filled(item_means)  
    x = np.tile(item_means, (matrix.shape[0],1))    
    # the above mentioned nan entries will be essentially zero now
    matrix = matrix - x    
    # U and V are user and item features
    U, s, V=np.linalg.svd(matrix, full_matrices=False)
    # take the k most significant features
    s=np.diag(s)   
    # get k most significant features
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:]    
    s_root=sqrtm(s)    
    Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV)    
    UsV = UsV + x    
    return UsV
  

  def recommend(self,itemsToPredict):
    matrix =  data_handler.create_matrix(dataset=self.dataset,columns=self.columns,fill_unrated_with=np.nan)
    pred_mat = pd.DataFrame(columns= matrix.columns.tolist(),data = self.svd(matrix, k=5))
    df = self.predict(pred_mat,itemsToPredict)
    c = df[self.columns[1]].tolist()
    for i in range(len(itemsToPredict)):
      if itemsToPredict[i] not in c:
        itemId = itemsToPredict[i]
        df = df.append(pd.DataFrame(data=[[itemId,np.nan]],columns=[self.columns[1],'svd_pred']))

    return df

  def predict(self,matrix,itemToPredict):
    df = pd.DataFrame(columns=[self.columns[1],'svd_pred'])
    for i in range(len(itemToPredict)):
      pred_item = itemToPredict[i]
      if pred_item in matrix.columns:
        y_pred = matrix[pred_item][self.input_user]
        if np.isnan(y_pred) == False:
          df = df.append(pd.DataFrame(data=[[int(pred_item),y_pred]],columns=[self.columns[1],'svd_pred']))
    
    return df    

  def evaluate_system(self,all_train,all_test):
    if type( self.evdf) == type(pd.DataFrame()):
      return  self.evdf
    # make variables local
    dataset = self.dataset
    columns = self.columns
    feature_num = self.feature_num
    evaluator_ = self.evaluator_

    # create user-item matrix with train data 
    matrix =  data_handler.create_matrix(dataset=all_train ,columns=columns ,fill_unrated_with=np.nan)
    
    # user index
    u_index = matrix.index.tolist()
    # item index
    i_index = matrix.columns.tolist()
    all_df = []
    all_rmse = []
    # heuristic feature picker
    for feat in feature_num:
      # predictions for test set
      pred_mat = pd.DataFrame(columns=i_index,data = self.svd(matrix, k=feat))
      
      # dataframe prediction for evaluation
      df = pd.DataFrame(columns=[columns[0],columns[1],'y_pred','y_true'])
      for i in range(len(all_test)):
        item_X = all_test.iloc[i][self.columns[1]]#.tolist()
        user_X = all_test.iloc[i][self.columns[0]]
        if item_X in pred_mat.columns and user_X in pred_mat.index.tolist():
          y_true =  all_test.iloc[i][self.columns[2]].tolist()
          y_pred = pred_mat[item_X][user_X]
          df = df.append(pd.DataFrame(data=[[int(user_X),int(item_X),y_pred,y_true]],columns=[columns[0],columns[1],'y_pred','y_true']))

      # reset index  
      df = df.reset_index().drop(columns='index')
      all_df.append(df)
      a,b = evaluator_.average_arpf_rm(data=df,cols=columns,threshold=4.5,model='svd')
      all_rmse.append(b['score'][0])

    print(all_rmse)

    pos = all_rmse.index(min(all_rmse))

      
    self.evdf = all_df[pos]
    self.all_ev = all_df
    df = self.evdf
    self.best_feat = feature_num[pos]
    return df

    def hyb_eval(self,train,test):
      # create matrix without test data
      matrix = data_handler.create_matrix(train,self.columns,fill_unrated_with=np.nan)
      itemsToPredict = test[self.columns[1]].tolist()
      pred_mat = pd.DataFrame(columns= matrix.columns.tolist(),data = self.svd(matrix, k=50))
      df = self.predict(pred_mat,itemsToPredict)
      c = df[self.columns[1]].tolist()
      for i in range(len(itemsToPredict)):
        if itemsToPredict[i] not in c:
          itemId = itemsToPredict[i]
          df = df.append(pd.DataFrame(data=[[itemId,np.nan]],columns=[self.columns[1],'svd_pred']))

      return df  




    def split_and_predict(self):
      # get all ratings of user input
      input_data = self.dataset.loc[self.dataset[self.columns[0]] == self.input_user]
      # split ratings to train and test
      train_X, test_X, train_y, test_y = train_test_split(input_data, input_data['rating'].tolist(), test_size=0.2, random_state=0)
      # copy dataset
      tmp_dataset = self.dataset.copy()
      # remove test data
      tmp_dataset = tmp_dataset.drop( index = test_X.index.tolist())
      # reset test index
      test_X = test_X.reset_index()
      # create user-item matrix with train data 
      matrix =  data_handler.create_matrix(dataset=tmp_dataset,columns=self.columns,fill_unrated_with=np.nan)
      # user index
      u_index = matrix.index.tolist()
      # item index
      i_index = matrix.columns.tolist()
      # predictions for test set
      pred_mat = pd.DataFrame(columns=i_index,data = self.svd(matrix, k=70))

      # 
      df = pd.DataFrame(columns=['clothId','y_pred','y_true'])
      for i in range(len(test_X)):
        item_X = test_X.iloc[i][self.columns[1]]#.tolist()
        
        if item_X in pred_mat.columns:
          y_true =  test_X.iloc[i][self.columns[2]].tolist()
          y_pred = pred_mat[item_X][self.input_user]
          df = df.append(pd.DataFrame(data=[[int(item_X),y_pred,y_true]],columns=['clothId','y_pred','y_true']))

     
    
      return df

  def hyb_eval(self,train,test):
    # create matrix without test data
    matrix = data_handler.create_matrix(train,self.columns,fill_unrated_with=np.nan)
    itemsToPredict = test[self.columns[1]].tolist()
    pred_mat = pd.DataFrame(columns= matrix.columns.tolist(),data = self.svd(matrix, k=105))
    df = self.predict(pred_mat,itemsToPredict)
    c = df[self.columns[1]].tolist()
    for i in range(len(itemsToPredict)):
      if itemsToPredict[i] not in c:
        itemId = itemsToPredict[i]
        df = df.append(pd.DataFrame(data=[[itemId,np.nan]],columns=[self.columns[1],'svd_pred']))

    return df  

  def coverage(self,threshold_):
    if type(self.recdf)  != type(pd.DataFrame()):
      pred_ratings = self.make_recommendation()
    else:
      pred_ratings = self.recdf

    already_rated = len(dataset)
    
    high_rated = len(pred_ratings.loc[pred_ratings['y_rec']>threshold_])
    
    low_rated = len(pred_ratings.loc[pred_ratings['y_rec']<=threshold_])

    unrated = len(pred_ratings.loc[pred_ratings['y_rec']==np.nan])
    
    cov_df = pd.DataFrame()

    cov_df['recommended'] = [high_rated]

    cov_df['not recommended'] = [low_rated]

    cov_df['cannot recommended'] = [unrated]

    cov_df['already rated'] = [already_rated]
    
    return cov_df
    
  def novelty(self,threshold_,cat_,translator_=False):

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
      ratings.append(round(len(fr.loc[fr['y_rec'] >=threshold_])  / len(fr.loc[fr['y_rec'] >=0]),2) )
      ratings.append(round(len(fr.loc[fr['y_rec'] <threshold_])  / len(fr.loc[fr['y_rec'] >=0]) ,2))
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