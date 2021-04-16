
eval = evaluator()

df = pd.DataFrame(columns=['user','metric','score','neighbors'])
df2 = pd.DataFrame(columns=['user','metric','score','neighbors'])
for u in range(5,6):
  for i in range(1,10):
    model = cf_userbased(dataset = dataset,columns = columns,input_user = u,max_neighbors = i)
    grade_df= model.split_and_predict(test_split_size = 0.2) 
    a,b = eval.nei_metric_df(user=u,neigh=i,df=grade_df,threshold=3)
    df = df.append(a)
    df2 = df2.append(b)
# visualize scores
eval.visualize_bars(df=df,axe_x='metric',axe_y='score',hue='neighbors')
eval.visualize_bars(df=df2,axe_x='metric',axe_y='score',hue='neighbors')
