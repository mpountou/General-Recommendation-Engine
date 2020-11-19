# import all libraries
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
''' 
@author: Mpountou
@year: 2020
'''
 
class cf_userbased:
  """
    A class to make collaborative user based recommendations
    ...
    
    Attributes
    ----------
    input_user : int
       number of userId
    max_neighbors : array
       number of neighbors used for collaborative filtering
    matrix : 2d-array
       a user - item with ratings matrix

    Methods
    -------
    common_ratings()
       retruns the indexes of common ratings given of two users
    user_similarities()
       returns dataframe for deep learning model based, total users, total items and minmax of ratings
    split_and_predict()
       splits the data to train and test and predicts the test  
    """ 

  def common_ratings(self,rA,rB):
    # rating index of userA
    iA = np.nonzero(rA)[0]
    # rating index of userB
    iB = np.nonzero(rB)[0]
    # common index of two users
    ci = set(iA) & set(iB)
    return list(ci)

  def user_similarities(self,input_user,max_neighbors,matrix):
    # all users expect input_user
    compared_user = matrix.index.tolist()
    compared_user.remove(input_user)
    # user correlations
    similarities = []
    # total common ratings
    t_common = []
    for i in range(len(compared_user)):
      # get input_user ratings 
      listA = np.array(matrix.loc[input_user].tolist())
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
    best_users = compared_user[best_index[:max_neighbors]]
    # find scores of users with best similarity
    best_scores = discount_cor[best_index[:max_neighbors]]

    return best_users,best_scores

 


  def split_and_predict(self,input_user,max_neighbors,dataset,columns,test_split_size):
    if max_neighbors <=0:
      print('Neighbor number must be larger than 0')
      return -1,-1
    # split data to train and test set
    train,test = handler.split(dataset,input_user)
    train_X = pd.DataFrame(train[columns[:3]])
    train_y = train[columns[2]].tolist()
    test_X = test[columns[:3]]
    test_y = test[columns[2]].tolist()
 
    # create matrix without test data
    matrix = data_handler.create_matrix(train,columns,fill_unrated_with=0)
    # find most similar users
    best_users,best_scores = self.user_similarities(input_user,max_neighbors,matrix)
    # find common ratings for first neighbor
    common_indexes = set(np.nonzero(matrix.loc[best_users[0]].tolist())[0])
    # find common ratings for all neighbors
    if max_neighbors >1:
      for i in range(1,len(best_users)):
        common_indexes = common_indexes & set(np.nonzero(matrix.loc[best_users[i]].tolist())[0])
   
    # find  ratings vector for input user
    input_user_index = np.array(matrix.loc[input_user].tolist())
    
    # find unrated items
    input_user_index = np.where(input_user_index == 0)[0]
    
    # find items that input user not rated and compared users rated
    common_indexes = common_indexes & set(input_user_index)
    
    # convert to list
    common_indexes = list(common_indexes)

    # common items
    common_items = np.array(matrix.columns.tolist())[common_indexes]

    # predict test items
    common_predict_items = list(set(common_items) & set(test_X[columns[1]]))
    
    # average ratings of all compared users
    mean_compared_users = []
    for i in range(len(best_users)):
      ratings = np.array(matrix.loc[best_users[i]].tolist())
      index_rating = np.nonzero(ratings)[0]
      mean_compared_users.append(np.mean(ratings[index_rating]))
    
    # index of input user ratings
    ratings = np.array(matrix.loc[input_user].tolist())
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

    # dataframe with pred ratings
    df1 = pd.DataFrame()
    df1[columns[1]] = common_predict_items
    pred = list(map(lambda x: round(x,0),pred))
    df1['y_pred'] = pred

    # data frame with true ratings
    df2 = test_X.loc[test_X[columns[1]].isin(common_predict_items)][[columns[1],columns[2]]]
    df2 = df2.rename(columns={"rating": "y_true"})

    # data frame with both pred and true ratings
    df = pd.merge(df1,df2,'inner',on=columns[1])

    return df
 