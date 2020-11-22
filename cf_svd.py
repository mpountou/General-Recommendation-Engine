import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

class cf_svd():

  def __init__(self,input_user,dataset,columns):
    self.input_user = input_user
    self.dataset = dataset
    self.columns = columns
    
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

  def split_and_predict(self):
    # get all ratings of user input
    input_data = self.dataset.loc[self.dataset[self.columns[0]] == self.input_user]
    # split ratings to train and test
    train_X, test_X, train_y, test_y = train_test_split(input_data, input_data['rating'].tolist(), test_size=0.1, random_state=5)
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
    pred_mat = pd.DataFrame(columns=i_index,data = self.svd(matrix, k=15))

    df = pd.DataFrame(columns=['clothId','y_pred','y_true'])
    for i in range(len(test_X)):
      item_X = test_X.iloc[i][self.columns[1]].tolist()
      if item_X in pred_mat.columns:
        y_true =  test_X.iloc[i][self.columns[2]].tolist()
        y_pred = pred_mat[item_X][self.input_user]
        df = df.append(pd.DataFrame(data=[[int(item_X),y_pred,y_true]],columns=['clothId','y_pred','y_true']))
    return df
  