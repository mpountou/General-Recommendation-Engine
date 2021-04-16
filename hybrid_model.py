''' 
@author: Mpountou
@year: 2020-2021
'''

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import linear_model
import pandas as pd
import numpy as np

class hybrid_v2():

  def __init__(self,dataset,cols,models,all_train,all_test):
    self.models = models
    self.dataset = dataset
    self.cols = cols
    self.linear_reg = -1
    self.poly_reg = - 1
    self.all_train = all_train
    self.all_test = all_test
    self.recdf = -1
    

  def make_recommendation(self):
    if type(self.recdf) == type(pd.DataFrame()):
      return self.recdf
    # make variables local
    models = self.models
    dataset = self.dataset
    cols = self.cols

    if type(self.linear_reg) == type(Pipeline([('polynomial',PolynomialFeatures(degree=1)),('modal',LinearRegression())])):
      linear_reg = self.linear_reg
      #poly_reg = self.poly_reg
    else:
      self.evaluate_system()
      linear_reg = self.linear_reg
      #poly_reg = self.poly_reg
    
    df = models[0].make_recommendation()

    df = df.rename(columns={'y_rec':'pred_0'})
    for i in range(1,len(models)):
          df_ = models[i].make_recommendation()
          df_ = df_.rename(columns={'y_rec':'pred_'+str(i)})
          df = df.merge(df_,on=[columns[0],columns[1]],how='inner')
    
    X = df[df.columns[2:]].values
    
    y_rec0 = X.mean(axis=1)
    y_rec1 = self.linear_reg.predict(X)
    #y_rec2 = self.poly_reg.predict(X)

    recdf = df[df.columns[:2]]
    recdf['y_rec'] = y_rec1
    #recdf['y_rec1'] = y_rec1
    #recdf['y_rec2'] = y_rec2

    self.recdf = recdf
    return recdf
    

  def evaluate_system(self):
    # make variables local
    models = self.models
    dataset = self.dataset
    cols = self.cols
    all_train = self.all_train
    all_test = self.all_test

    df = models[0].evaluate_system(all_train,all_test)
    df = df.rename(columns={'y_pred':'pred_0'})
    for i in range(1,len(models)):
      df_ = models[i].evaluate_system(all_train,all_test)
      df_ = df_.rename(columns={'y_pred':'pred_'+str(i)})
      df = df.merge(df_,on=[columns[0],columns[1],'y_true'],how='inner')
    
    self.stat = df
    
    X = pd.DataFrame()
    for i in range(len(models)):
      X['pred_'+str(i)] = df['pred_'+str(i)]
     
    y = df['y_true']

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.7, random_state=42)

    average_df = pd.DataFrame()
    average_df[columns[0]] = df[columns[0]]
    average_df[columns[1]] = df[columns[1]]
    average_df['y_pred'] = X.values.mean(axis=1)
    average_df['y_true'] = df['y_true'].tolist()

    # REGRESSION DEGREE = 1 
    lr_df = pd.DataFrame()
    lr_df[columns[0]] = df[columns[0]]
    lr_df[columns[1]] = df[columns[1]]
    input_=[('polynomial',PolynomialFeatures(degree=1)),('modal',LinearRegression())]
    pipe=Pipeline(input_)
    pipe.fit(train_X.values,train_y)
    y_pred= pipe.predict(X)
    lr_df['y_pred'] = y_pred
    lr_df['y_true'] = df['y_true'].tolist()
    self.linear_reg = pipe

    # REGRESSION DEGREE = 2 
    #pr_df = pd.DataFrame()
    #pr_df[columns[0]] = df[columns[0]]
    #pr_df[columns[1]] = df[columns[1]]
    #input_=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
    #pipe=Pipeline(input_)
    #pipe.fit(train_X.values,train_y)
    #y_pred= pipe.predict(X)
    #pr_df['y_pred'] = y_pred
    #pr_df['y_true'] = df['y_true'].tolist()
    #self.poly_reg = pipe

    self.train_X = train_X
    self.test_X = test_X
    self.train_y = train_y
    self.test_y = test_y

    return average_df,lr_df


  def coverage(self):
    if type(self.recdf)  != type(pd.DataFrame()):
      pred_ratings = self.make_recommendation()
    else:
      pred_ratings = self.recdf

    already_rated = len(dataset)
    
    high_rated = len(pred_ratings.loc[pred_ratings['y_rec']>6])
    
    low_rated = len(pred_ratings.loc[pred_ratings['y_rec']<=6])

    unrated = len(pred_ratings.loc[pred_ratings['y_rec']==np.nan])
    
    cov_df = pd.DataFrame()

    cov_df['recommended'] = [high_rated]

    cov_df['not recommended'] = [low_rated]

    cov_df['cannot recommended'] = [unrated]

    cov_df['already rated'] = [already_rated]
    
    return cov_df

  def novelty(self,cat_,translator_=False):

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
      ratings.append(round(len(fr.loc[fr['y_rec'] >=6])  / len(fr.loc[fr['y_rec'] >=0]),2) )
      ratings.append(round(len(fr.loc[fr['y_rec'] <6])  / len(fr.loc[fr['y_rec'] >=0]) ,2))
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
