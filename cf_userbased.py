''' 
@author: Mpountou
@year: 2020-2021
'''

# import all libraries
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

class cf_userbased:
  
  def __init__(self,dataset,columns,input_user,max_neighbors):
    self.dataset = dataset
    self.columns = columns
    self.input_user = input_user
    self.max_neighbors = max_neighbors
    self.evdf = -1
    self.recdf = -1
  def make_recommendation(self):
    if type( self.recdf) == type(pd.DataFrame()):
      return  self.recdf
      # make variables local
    dataset = self.dataset
    columns = self.columns

    all_users = dataset[columns[0]].unique()
    all_items = dataset[columns[1]].unique()
    matrix = data_handler.create_matrix(dataset,columns,fill_unrated_with=0)
    df = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
    for i in all_users:
      u_data = dataset.loc[dataset[columns[0]] == i]
      rated = u_data[columns[1]].tolist()
      unrated = list(set(all_items)-set(rated))
      self.input_user = i
      best_users,best_scores = self.user_similarities(matrix)
      rec_items,rec_ratings = self.predict(matrix,best_users,best_scores,unrated)
  
      # pred
      u_df = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
      u_df[columns[0]] = [i for j in range(len(rec_items))]
      u_df[columns[1]] = rec_items
      u_df['y_rec'] = rec_ratings
      df = df.append(u_df)

      # not pred
      items_left = list(set(unrated) - set(rec_items))
      u_df = pd.DataFrame(columns=[columns[0],columns[1],'y_rec'])
      u_df[columns[0]] = [i for j in range(len(items_left))]
      u_df[columns[1]] = items_left
      u_df['y_rec'] = [np.nan for j in range(len(items_left))]
 
      df = df.append(u_df)
      
    self.recdf = df

    return df  
  def evaluate_system(self,all_train,all_test):
    if type( self.evdf) == type(pd.DataFrame()):
      return  self.evdf
    # make variables local
    dataset = self.dataset
    columns = self.columns

    all_users = dataset[columns[0]].unique()
    all_items = dataset[columns[1]].unique()

    matrix = data_handler.create_matrix(all_train,columns,fill_unrated_with=0)
    df = pd.DataFrame(columns=[columns[0],columns[1],'y_pred','y_true'])
    for i in all_users:
      u_train = all_train.loc[all_train[columns[0]] == i]
      u_test = all_test.loc[all_test[columns[0]] == i]
      self.input_user = i
      best_users,best_scores = self.user_similarities(matrix)
      pred_items,pred_ratings = self.predict(matrix,best_users,best_scores,u_test[columns[1]])

      # pred
      u_df = pd.DataFrame(columns=[columns[0],columns[1],'y_pred','y_true'])
      u_df[columns[0]] = [i for j in range(len(pred_items))]
      u_df[columns[1]] = pred_items
      u_df['y_pred'] = pred_ratings
      u_rem = u_test.copy()
      u_df['y_true'] = u_rem.set_index(keys=columns[1]).loc[pred_items][columns[2]].tolist()
      df = df.append(u_df)

      # not pred
      items_left = list(set(u_test[columns[1]].tolist()) - set(pred_items))
      u_df = pd.DataFrame(columns=[columns[0],columns[1],'y_pred','y_true'])
      u_df[columns[0]] = [i for j in range(len(items_left))]
      u_df[columns[1]] = items_left
      u_df['y_pred'] = [np.nan for j in range(len(items_left))]
      u_rem = u_test.copy()
      u_df['y_true'] = u_rem.set_index(keys=columns[1]).loc[items_left][columns[2]].tolist()
      df = df.append(u_df)
  
    df = df.reset_index().drop(columns='index')
    self.evdf = df
    return df

  def common_ratings(self,rA,rB):
    # rating index of userA
    iA = np.nonzero(rA)[0]
    # rating index of userB
    iB = np.nonzero(rB)[0]
    # common index of two users
    ci = set(iA) & set(iB)
    return list(ci)

  def user_similarities(self,matrix):
    # all users expect input_user
    compared_user = matrix.index.tolist()
   
    compared_user.remove(self.input_user)
    # user correlations
    similarities = []
    # total common ratings
    t_common = []
    for i in range(len(compared_user)):
      # get input_user ratings 
      listA = np.array(matrix.loc[self.input_user].tolist())
      # get compared_user ratings
      listB = np.array(matrix.loc[compared_user[i]].tolist())
      # get all common index ratings 
      common_index = self.common_ratings(listA,listB)
      # if common ratings are more than 2 calculate similarity
      if len(common_index) >=2:
        # save total common ratings
        t_common.append(len(common_index))
        # save similarity between input_user and compared_user
        similarities.append(pearsonr(listA[common_index],listB[common_index])[0])
        #similarities.append(cosine_similarity([listA[common_index]],[listB[common_index]])[0][0])
      else:
        # save 1 and not 0 because of division errors
        t_common.append(1)
        # not enough common ratings ~ zero similarity
        similarities.append(0)

    # mean value of common ratings 
    mean_common = np.mean(t_common)
    # discount similarity
    discount_cor = []
 
    # calculate discounts for every similarity
    for i in range(len(similarities)):
      cor = similarities[i] * ( min(mean_common,t_common[i]) / mean_common )
      discount_cor.append(cor)

    # convert to numpy
    discount_cor = np.array(discount_cor)
    # index of best similarities
    best_index = np.argsort(discount_cor)[::-1][:len(discount_cor)]
    # convert to numpy
    compared_user = np.array(compared_user)
    # find users with best similarity
    best_users = compared_user[best_index[:self.max_neighbors]]
    # find scores of users with best similarity
    best_scores = discount_cor[best_index[:self.max_neighbors]]

    return best_users,best_scores

 
  def recommend(self,itemsToPredict):
    # create matrix without test data
    matrix = data_handler.create_matrix(self.dataset,self.columns,fill_unrated_with=0)
    available_data = matrix.columns
    #for i in range(len(itemsToPredict)):
    #  if itemsToPredict[i] not in available_data:
    #    itemsToPredict.remove(itemsToPredict[i])
    
    best_users,best_scores = self.user_similarities(matrix)
    
    c,pred = self.predict(matrix,best_users,best_scores,itemsToPredict)
    df = pd.DataFrame()
    df[self.columns[1]] = c
    df['ub_pred'] = pred
    
    for i in range(len(itemsToPredict)):
      
      if itemsToPredict[i] not in c:
        itemId = itemsToPredict[i]
        df = df.append(pd.DataFrame(data=[[itemId,np.nan]],columns=[self.columns[1],'ub_pred']))

    return df
  def sys_eval(self,test_split_size):
    if self.max_neighbors <=0:
      print('Neighbor number must be larger than 0')
      return -1,-1
    all_df = []
    for i in range(100):
      # split data to train and test set
      train,test = handler.split(self.dataset,i,test_split_size)
      train_X = pd.DataFrame(train[self.columns[:3]])
      train_y = train[self.columns[2]].tolist()
      test_X = test[self.columns[:3]]
      test_y = test[self.columns[2]].tolist()
  
      # create matrix without test data
      matrix = data_handler.create_matrix(train,self.columns,fill_unrated_with=0)
      # find most similar users
      best_users,best_scores = self.user_similarities(matrix)
      common_predict_items,pred = self.predict(matrix,best_users,best_scores,test_X[self.columns[1]])

      # dataframe with pred ratings
      df1 = pd.DataFrame()
      df1[columns[1]] = common_predict_items
      pred = list(map(lambda x: round(x,2),pred))
      df1['y_pred'] = pred
      df1[self.columns[1]] = df1[self.columns[1]].astype('float64')
      # data frame with true ratings
      df2 = test_X.loc[test_X[self.columns[1]].isin(common_predict_items)][[self.columns[1],self.columns[2]]]
      df2 = df2.rename(columns={"rating": "y_true"})
      df2[self.columns[1]] = df2[self.columns[1]].astype('float64')
      # data frame with both pred and true ratings
      df = pd.merge(df1,df2,'inner',on=self.columns[1])
      all_df.append(df)
    return all_df
  def split_and_predict(self,test_split_size):
    if self.max_neighbors <=0:
      print('Neighbor number must be larger than 0')
      return -1,-1
    # split data to train and test set
    train,test = handler.split(self.dataset,self.input_user,test_split_size)
    train_X = pd.DataFrame(train[self.columns[:3]])
    train_y = train[self.columns[2]].tolist()
    test_X = test[self.columns[:3]]
    test_y = test[self.columns[2]].tolist()
 
    # create matrix without test data
    matrix = data_handler.create_matrix(train,self.columns,fill_unrated_with=0)
    # find most similar users
    best_users,best_scores = self.user_similarities(matrix)
    common_predict_items,pred = self.predict(matrix,best_users,best_scores,test_X[self.columns[1]])

    # dataframe with pred ratings
    df1 = pd.DataFrame()
    df1[columns[1]] = common_predict_items
    df1[self.columns[1]] = df1[self.columns[1]].astype('float64')
    pred = list(map(lambda x: round(x,2),pred))
    df1['y_pred'] = pred

    # data frame with true ratings
    df2 = test_X.loc[test_X[self.columns[1]].isin(common_predict_items)][[self.columns[1],self.columns[2]]]
    df2 = df2.rename(columns={"rating": "y_true"})
    df2[self.columns[1]] = df2[self.columns[1]].astype('float64')
    # data frame with both pred and true ratings
    df = pd.merge(df1,df2,'inner',on=self.columns[1])

    return df

  def predict(self,matrix,best_users,best_scores,itemsToPredict):
    common_indexes = set(np.nonzero(matrix.loc[best_users[0]].tolist())[0])
    # find common ratings for all neighbors
    if self.max_neighbors >1:
      for i in range(1,len(best_users)):
        common_indexes = common_indexes & set(np.nonzero(matrix.loc[best_users[i]].tolist())[0])

    # find  ratings vector for input user
    input_user_index = np.array(matrix.loc[self.input_user].tolist())
    
    # find unrated items
    input_user_index = np.where(input_user_index == 0)[0]
    
    # find items that input user not rated and compared users rated
    common_indexes = common_indexes & set(input_user_index)
    
    # convert to list
    common_indexes = list(common_indexes)

    # common items
    common_items = np.array(matrix.columns.tolist())[common_indexes]

    # predict test items
    common_predict_items = list(set(common_items) & set(itemsToPredict))

    # average ratings of all compared users
    mean_compared_users = []
    for i in range(len(best_users)):
      ratings = np.array(matrix.loc[best_users[i]].tolist())
      index_rating = np.nonzero(ratings)[0]
      mean_compared_users.append(np.mean(ratings[index_rating]))
    
    # index of input user ratings
    ratings = np.array(matrix.loc[self.input_user].tolist())
    index_rating = np.nonzero(ratings)[0]
    
    # average ratings of input user
    mean_input_user = np.mean(ratings[index_rating])
    
    # pred ratings will be saved here
    pred = []

    # for all common ratings
    for i in range(len(common_predict_items)):
      # find all grades for specific item
      grades = []
      for k in range(len(best_users)):
        grades.append(matrix[common_predict_items[i]][best_users[k]])
      # length of ratings
      totalGrades = len(grades)
      weighted_grades = []
      sum1 = 0
      sum2 = 0
      # calculate weighted grade
      for j in range(totalGrades):
        weighted_grades.append(grades[j] -mean_compared_users[j])
        sum1 += weighted_grades[j] * best_scores[j]
        sum2 += best_scores[j]
      pred.append(mean_input_user + sum1/sum2)
    return common_predict_items,pred

  def hyb_eval(self,train,test):
    # create matrix without test data
    matrix = data_handler.create_matrix(train,self.columns,fill_unrated_with=0)
    available_data = matrix.columns
 
    itemsToPredict = test[self.columns[1]].tolist()
 
    best_users,best_scores = self.user_similarities(matrix)
 
    c,pred = self.predict(matrix,best_users,best_scores,itemsToPredict)
 
    df = pd.DataFrame()
    df[self.columns[1]] = c
    df['ub_pred'] = pred
 
    for i in range(len(itemsToPredict)):
      if itemsToPredict[i] not in c:
        itemId = itemsToPredict[i]
        df = df.append(pd.DataFrame(data=[[itemId,np.nan]],columns=[self.columns[1],'ub_pred']))
    return df

  def coverage(self,threshold):
    # make variables local
    dataset = self.dataset
    columns = self.columns

    if type(self.recdf)  != type(pd.DataFrame()):
      pred_ratings = self.make_recommendation()
    else:
      pred_ratings = self.recdf

    already_rated = len(dataset)
    
    high_rated = len(pred_ratings.loc[pred_ratings['y_rec']>threshold])
    
    low_rated = len(pred_ratings.loc[pred_ratings['y_rec']<=threshold])

    unrated = len(pred_ratings.loc[np.isnan(pred_ratings['y_rec'])])
    
    cov_df = pd.DataFrame()

    cov_df['recommended'] = [high_rated]

    cov_df['not recommended'] = [low_rated]

    cov_df['cannot recommended'] = [unrated]

    cov_df['already rated'] = [already_rated]
    
    return cov_df
    
  def novelty(self,threshold,cat_,translator_=False):
    # make variables local
    dataset = self.dataset
    columns = self.columns

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
# 5.5