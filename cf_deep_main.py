handler = data_handler(path,columns)

preprocess_pack = handler.deep_preprocessing(dataset)

eval = evaluator()

df = pd.DataFrame(columns=['user','metric','score'])
df2 = pd.DataFrame(columns=['user','metric','score'])

for u in range(5):
  dl = cf_deep(preprocess_pack,handler,columns,u)
  grade_df = dl.split_and_predict()
  a,b = eval.user_metric_df(user=u,df=grade_df,threshold=3)
  df = df.append(a)
  df2 = df2.append(b)
eval.visualize_bars(df = df,axe_x='metric',axe_y='score',hue='user')
eval.visualize_bars(df = df2,axe_x='metric',axe_y='score',hue='user')



