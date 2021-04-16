''' 
@author: Mpountou
@year: 2020-2021
'''

# import all libraries
import seaborn as sns
from matplotlib import pyplot
sns.set_theme()
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

class evaluator:

  def calculate_metrics(self,df,threshold):

    y_true = df['y_true'].tolist()
    y_pred = df['y_pred'].tolist()

    # calculate rmse
    rmse = mean_squared_error(y_true,y_pred)**(1/2)

    # calculate mae
    mae = mean_absolute_error(y_true,y_pred)   
    # convert predictions to binary based on threshold

    pred = [1 if x >=threshold else 0 for x in y_pred]
    true = [1 if x >=threshold else 0 for x in y_true]
    # calculate accuracy
    acc_ = accuracy_score(true,pred)
    # calculate recall
    rec_ = recall_score(true, pred)
    # calculate precision
    prec_ =precision_score(true, pred)
    # calculate f1 score
    f1_ = f1_score(true, pred)

    return rmse,mae,acc_,rec_,prec_,f1_

  def average_arpf_rm(self,data,cols,threshold,model):

    rmse = []
    mae = []

    acc = []  
    rec = []
    pre = []
    f1 = []

    all_users = data[cols[0]].unique()
    for i in all_users:

      u_data = data.loc[data[cols[0]] == i]

      y_pred = u_data['y_pred']
      y_test = u_data['y_true']

      labels_pred = [1 if i>=threshold else 0 for i in y_pred]
      labels_true = [1 if i>=threshold else 0 for i in y_test]

      acc.append(accuracy_score(labels_true,labels_pred))
      rec.append(recall_score(labels_true,labels_pred))
      pre.append(precision_score(labels_true,labels_pred))
      f1.append(f1_score(labels_true,labels_pred))

      rmse.append(mean_squared_error(y_test,y_pred)**(1/2))
      mae.append(mean_absolute_error(y_test,y_pred))



    df = pd.DataFrame(columns=['model','metric','score'])

    df = df.append(pd.DataFrame(data=[[model,'accuracy',np.mean(acc)]] , columns=['model','metric','score']))
    df = df.append(pd.DataFrame(data=[[model,'recall',np.mean(rec)]] , columns=['model','metric','score']))
    df = df.append(pd.DataFrame(data=[[model,'precision',np.mean(pre)]] , columns=['model','metric','score']))
    df = df.append(pd.DataFrame(data=[[model,'f1_score',np.mean(f1)]] , columns=['model','metric','score']))

    # reset index  
    df = df.reset_index().drop(columns='index')

    df2 = pd.DataFrame(columns=['model','metric','score'])

    df2 = df2.append(pd.DataFrame(data=[[model,'rmse',np.mean(rmse)]] , columns=['model','metric','score']))
    df2 = df2.append(pd.DataFrame(data=[[model,'mae',np.mean(mae)]] , columns=['model','metric','score']))

    # reset index  
    df2 = df2.reset_index().drop(columns='index')

    return df,df2

  def nei_metric_df(self,user,neigh,df,threshold):
    # calculate matrics
    rmse,mae,acc_,rec_,prec_,f1_ = self.calculate_metrics(df,threshold)
    # create a dataframe with those metrics, scores and neighbors
    df = pd.DataFrame(columns=['user','metric','score','neighbors']) 
    # append accuracy
    df = df.append(pd.DataFrame(data=[[user,'accuracy',acc_,neigh]],columns=['user','metric','score','neighbors']))
    # append recall
    df = df.append(pd.DataFrame(data=[[user,'recall',rec_,neigh]],columns=['user','metric','score','neighbors']))
    # append precision
    df = df.append(pd.DataFrame(data=[[user,'precision',prec_,neigh]],columns=['user','metric','score','neighbors']))
    # append f1_score
    df = df.append(pd.DataFrame(data=[[user,'f1_score',f1_,neigh]],columns=['user','metric','score','neighbors']))
    df2 = pd.DataFrame(columns=['user','metric','score','neighbors']) 
    # append accuracy
    df2 = df2.append(pd.DataFrame(data=[[user,'rmse',rmse,neigh]],columns=['user','metric','score','neighbors']))
    # append recall
    df2 = df2.append(pd.DataFrame(data=[[user,'mae',mae,neigh]],columns=['user','metric','score','neighbors']))
    
    return df,df2
    
  def user_metric_df(self,user,df,threshold):
    # calculate matrics
    rmse,mae,acc_,rec_,prec_,f1_ = self.calculate_metrics(df,threshold)
    # create a dataframe with those metrics and scores
    df = pd.DataFrame(columns=['user','metric','score']) 
    # append accuracy
    df = df.append(pd.DataFrame(data=[[user,'accuracy',acc_]],columns=['user','metric','score']))
    # append recall
    df = df.append(pd.DataFrame(data=[[user,'recall',rec_]],columns=['user','metric','score']))
    # append precision
    df = df.append(pd.DataFrame(data=[[user,'precision',prec_]],columns=['user','metric','score']))
    # append f1_score
    df = df.append(pd.DataFrame(data=[[user,'f1_score',f1_]],columns=['user','metric','score']))

    df2 = pd.DataFrame(columns=['user','metric','score']) 
    # append accuracy
    df2 = df2.append(pd.DataFrame(data=[[user,'rmse',rmse]],columns=['user','metric','score']))
    # append recall
    df2 = df2.append(pd.DataFrame(data=[[user,'mae',mae]],columns=['user','metric','score']))
    
    return df,df2

  def visualize_bars(self,df,axe_x,axe_y,hue,title,x_dist,hei_,asp_):
    sns.color_palette("husl", 8)
    sns.set(font_scale=2)
    g = sns.catplot(
    data=df, kind="bar",palette="mako",
    x=axe_x, y=axe_y, hue=hue,
    ci="sd",  height=hei_,legend_out = True,aspect=asp_
    ) 
    ax = g.facet_axis(0,0)
    sns.set(font_scale=1)
    for p in ax.patches:
        ax.text(p.get_x() + x_dist, 
                p.get_height() * 1.02, 
                '{0:.2f}'.format(p.get_height()), 
                color='black', rotation='horizontal', size='large')
    g.despine(left=True)
    g.set_axis_labels("", "Percent (%)")
    g.legend.set_title(""+title)


  def visualize_pie(self,df,df2):
    sns.set(font_scale=3)
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(35,15),nrows=1, ncols=2)

    colors = ['#3cd070','#fc4f30', '#57595D' , '#80F5FF']
    explode = (0.1, 0.1, 0.1,0.1)
  
    sizes = []
    sizes.append(df['recommended'].tolist()[0])
    sizes.append(df['not recommended'].tolist()[0])
    sizes.append(df['cannot recommended'].tolist()[0])
    
    sizes.append(df['already rated'].tolist()[0])
    labels = ['προτείνεται','δεν προτείνεται','δε μπορεί να αξιολογηθεί','ήδη αξιολογημένο']

    if (sizes[2] == 0):
      del colors[2]
      del sizes[2]
      del labels[2]
      explode = list(explode)
      del explode[2]
      explode = tuple(explode)
    #plt.tight_layout()
    
    n_labels = []
    for i in range(len(sizes)):
      n_labels.append(labels[i]+str(' ')+str(int(round(100*sizes[i]/sum(sizes),0))) + str('%'))
    ax[0].pie(sizes, startangle=90, colors=colors ,labels=n_labels, autopct='%1.f%%', explode=explode, shadow=True,textprops={'color':"w"})
    ax[0].set_title('Κάλυψη (Coverage)',fontsize=40, fontweight='bold')
    ax[0].legend( loc=(0.1, -0.1),fontsize=30,frameon=True)
    
    df2.plot.barh(color=['#3cd070','#fc4f30'],stacked=True,ax=ax[1]);
    df_perc = df2.copy().reset_index()

    for n in df2: 
      for i, (cs, ab, pc) in enumerate(zip(df_perc.iloc[:, 1:].cumsum(1)[n],  
                                          df_perc[n], df_perc[n])): 
          plt.text(cs - ab / 2, i, str(np.round(pc, 2)) + '%',  
                  va = 'center', ha = 'center',color='white')
    ax[1].set_title('Ποικιλια (Diversity)',fontsize=40, fontweight='bold')
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2,fontsize=30)
    for label in ( ax[1].get_xticklabels() +  ax[1].get_yticklabels()):
	    label.set_fontsize(30)
    plt.ylabel('Κατηγορίες', fontsize=35)
    fig.suptitle('Αξιολόγηση συστήματος προτάσεων', fontsize=25, y=1.07, fontweight='bold', x=0.37)
