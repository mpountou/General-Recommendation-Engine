import difflib
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

class recommendation_query():

  def __init__(self,df,input_user,models):
    self.input_user = input_user
    self.df = df 
    self.models = models

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

  def search(self):
    # available search words in dataset
    all_searchWords = self.df['SearchWords'].tolist()

    self.df = self.df.loc[self.df['UserId']==self.input_user]
    # make search words to lower case
    all_searchWords = [x.lower() for x in all_searchWords]

    # remove duplicate search words
    all_searchWords = set(all_searchWords)

    # user searching clothes category
    input_searchWords = input("Please enter your search words:  ")

   
    # find matches based on dataset searchwords
    matches = difflib.get_close_matches(input_searchWords, all_searchWords, len(all_searchWords) )
    self.matches = matches
    # print matches
    print('Result found: ',matches)
    
    # merge result of matches
    dataFrames = []
    for i in matches:
      dataFrames.append(self.df.loc[self.df['SearchWords'] == i.upper()])

    # all available clothes based on user search
    query_result = pd.concat(dataFrames)
    liked = query_result.loc[query_result['gradeUser'] >= 4]
    sw = liked['SearchWords'].tolist()
    prodNo = liked['ProductNo'].tolist()
    id = self.convert_index(prodNo)
    img_src = liked['ImageSource'].tolist()
    ldf = pd.DataFrame()
    ldf['SearchWords'] = sw
    ldf['clothId'] = id
    ldf['ImageSource'] = img_src

    query_result = query_result.loc[query_result['gradeUser'] ==2.5]
    sw = query_result['SearchWords'].tolist()
    prodNo = query_result['ProductNo'].tolist()
    img_src = query_result['ImageSource'].tolist()
    id = self.convert_index(prodNo)
    df = pd.DataFrame()
    df['SearchWords'] = sw
    df['clothId'] = id
    df['ImageSource'] = img_src
    return df,ldf


  def show_search(self):
    res = pd.read_pickle('/content/drive/My Drive/id_category.pkl')
    
    preds = []
    result,ldf = self.search()
    #print(result)
    print(len(result))
    for i in range(len(self.models)):
      preds.append(models[i].recommend(result['clothId'].unique()))
    df = preds[0]
    for i in range(1,len(preds)):
      df = pd.merge(df,preds[i],on='clothId',how='inner')
    df['mean'] = df.drop(columns='clothId').mean(axis=1)
    df = pd.merge(df,result,on='clothId')

    #print(df)
    #print(self.matches)
    liked_df = []
    
    
    matched_df = []
    for i in range(len(self.matches)):
     matched_df.append(df.loc[df['SearchWords'] == self.matches[i].upper()])
     liked_df.append(ldf.loc[ldf['SearchWords'] == self.matches[i].upper()])
    return matched_df,liked_df


  def show_rec(self,liked_df,matched_df):

    for i in range(3,4):
      #print('CATEGORY: ',self.matches[i])
      print('LIKED CLOTHES :')

      f_size = len(liked_df[i]) // 8
      print(f_size)
      print(len(liked_df[i]))
      if len(liked_df[i]) % 8 > 0:
        f_size += 1 
      print(f_size)
      fig, axs = plt.subplots(f_size, 8, sharex='col', sharey='row',figsize=(15,3*f_size),
                          gridspec_kw={'hspace': 0, 'wspace': 0})
      if f_size == 1:
        axs = np.expand_dims(axs, axis=0)
      x = np.linspace(0, 0 * np.pi, 300)
      y = np.sin(x ** 2)

      for j in range(f_size):
        for k in range(j*8,8*(j+1)):
          if (len(liked_df[i]) >= 8*(j+1)) or (len(liked_df[i]) > k):
            cloth = liked_df[i].iloc[k]
            rec_cloth = cloth['ImageSource']
            image = io.imread(rec_cloth)
            pos = k % 8
            axs[j][pos].imshow(image)
          else:
            axs[j][pos].plot(x,y)


      print('RECOMMENDED CLOTHES :')

      f_size = len(matched_df[i]) // 8
      print(f_size)
      print(len(matched_df[i]))
      if len(matched_df[i]) % 8 > 0:
        f_size += 1 
      print(f_size)
      fig, axs = plt.subplots(f_size, 8, sharex='col', sharey='row',figsize=(15,3*f_size),
                          gridspec_kw={'hspace': 0, 'wspace': 0})
      if f_size == 1:
        axs = np.expand_dims(axs, axis=0)
      x = np.linspace(0, 0 * np.pi, 300)
      y = np.sin(x ** 2)

      for j in range(f_size):
        for k in range(j*8,8*(j+1)):
          if (len(matched_df[i]) >= 8*(j+1)) or (len(matched_df[i]) > k):
            cloth = matched_df[i].iloc[k]
            rec_cloth = cloth['ImageSource']
            image = io.imread(rec_cloth)
            pos = k % 8
            axs[j][pos].imshow(image)
          else:
            axs[j][pos].plot(x,y)

     
'''      
    for i in range(4):
      cloth = liked_df[1].iloc[i]
      rec_cloth = cloth['ImageSource']
      print('-----------------')  
      image = io.imread(rec_cloth)
      plt.imshow(image)
      plt.show()
    for i in range(4):
      cloth = matched_df[1].iloc[i]
      rec_cloth = cloth['ImageSource']
      print('-----------------')  
      image = io.imread(rec_cloth)
      plt.imshow(image)
      plt.show()
'''