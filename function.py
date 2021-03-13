class ModelData:
  def __init__(self):
    self.dataset=None

  def Load(self):
    # 모드 별로 설명하고 원하는 모드 번호를 입력받는 코드 
    choice=int(input('\nWhich of the two modes do you want?\nEnter the number of the mode you want.\n1. Exercise mode\n2. Practice mode\n'))
    if choice == 1 : # 연습모드를 선택했을 때, 예제데이터 불러오는 함수 실행
      print('You have selected Exercise mode.')
      self._LoadExampleData()
    elif choice == 2: # 실습모드를 선택했을 때, 파일 불러오기
      print('You have selected Practice mode.\n')
      self._LoadPracticeData()

  def _LoadExampleData(self): # 예제데이터를 불러오는 함수
    url = 'https://raw.githubusercontent.com/Seo-Junh0/first-git/master/NiFeCr_Total_Hardness.csv' # 예제데이터 위치(깃허브)
    self.dataset = pd.read_csv(url, index_col=0) # csv 형식의 예제데이터를 데이터프레임 형태로 불러오기 

  def _LoadPracticeData(self):
    # 실습모드에서 파일불러오는 방법은 1. 로컬드라이브에서 불러오기 2. 구글드라이브에서 불러오기 두가지로 구성
    location=int(input('\nNow where do you want to load your file from?\n1. Local drive\n2. Google drive\n'))
    if location == 1:
      self._LocalData()
    elif location == 2:
      self._DriveData()

  def _LocalData(self):
    # 2. 구글드라이브에서 불러오기
    import pathlib
    from google.colab import files
    uploaded = files.upload()
    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
    import io
    self.dataset= pd.read_csv(io.BytesIO(uploaded[fn]), index_col=0)

  def _DriveData(self):
    # 1. 로컬드라이브에서 불러오기
    filepath=input("\nInput your file path(in your google drive)\n")
    filename = filepath
    self.dataset= pd.read_csv(filename)

  def ScanData(self): # 불러온 데이터가 전부 유효하거나 숫자인지 확인하는 함수
    check_for_nan1 = self.dataset.isnull().values.any()
    df=self.dataset.apply(pd.to_numeric, errors = 'coerce')
    check_for_nan2 = df.isnull().values.any()
    if check_for_nan1==True:
      print('Warning! Some of your data is abnormal.')
    elif check_for_nan2==True:
      print('Caution! Your data contains a non-numeric format.')
    else :
      print("Your data is all numeric. So it's available.")

class DataSelect:

  def __init__(self,dataset):
    self.dataset=dataset
    self.target_name=None
    self.feature_names= None
    self.selected_feature_names = None
    self.data_to_use=None

  def TargetSelect(self):
    index=int(input("Column number of the target variable : "))-1
    self.target_name=self.dataset.columns[index] # target_index를 통해 타겟 변수의 이름 구하기
    self.feature_data=self.dataset.drop(self.target_name, axis=1) # 데이터에서 타겟 변수 열을 제외하기
    self.feature_names=list(self.feature_data.columns) # 타겟 변수를 제외한 특징들의 이름 리스트
    print("You chose '%s' as the target variable.\n"%self.target_name)

  def FeatureSelect(self):
    selected_feature_indexs=input('Enter the index of features you want to use among the above features (Use "," to separate each index number) : \n').split(',')
    self.selected_feature_names = [self.feature_names[((int (i))-1)] for i in selected_feature_indexs] # 구한 인덱스 값으로 해당 선택된 특징의 이름 구하여 리스트 만들기
    print("You chose %s as the input feature.\n"%self.selected_feature_names)
  
  def PearsonHeatmap(self):
    pearson = self.dataset.corr(method = 'pearson') # 데이터의 각 특징 간의 피어슨 상관계수 구하기
    plt.figure(figsize = (8,8)) # 상관계수 그림 크기 설
    sns_plot_pearson = sns.heatmap(pearson.apply(lambda x: x ** 2), square=True, cmap='Reds') # 상관계수 값의 제곱한 값을 기준으로 히트맵이미지 그리기
    sns_plot_pearson.set_title('The Table for Correlation')

  def ScatterPlot(self):
    # 이름 앞에 인덱스 변호 추가 하기
    total_range=[]
    feature_data=self.dataset.drop(self.target_name, axis=1) # 데이터에서 타겟 변수 열을 제외하기
    for i in range(1,self.dataset.shape[1]):
      total_range.append(str(i)+". "+self.feature_names[i-1])
    feature_data.columns=total_range
    # 타겟 변수와 특징 사이의 산점도 나타내기
    feature_data[self.target_name]=self.dataset[self.target_name].values
    column=5
    part=(len(total_range)-1)//column
    for i in range(0,part):
      sns_scatterplot=sns.pairplot(data=feature_data, diag_kind="kde",  x_vars=total_range[i*column:(i+1)*column], y_vars=[self.target_name])
      if i==0:
        sns_scatterplot.fig.subplots_adjust(top=0.9)
        sns_scatterplot.fig.suptitle("The Graph for Linearity (between '%s' and remaining features)"%self.target_name)
    sns_scatterplot=sns.pairplot(data=feature_data, diag_kind="kde",  x_vars=total_range[part*column:], y_vars=[self.target_name])

  def ExtractDataToUse(self):
    self.data_to_use=self.dataset.loc[:,self.selected_feature_names+[self.target_name]] # 선택된 특징과 타겟 변수로 구성된 데이터 만들기
    return self.data_to_use # 선택된 특징와 타겟 변수로만 이루어진 데이터 확인하기

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

class ANN:

  activfunc_list=['softmax','sigmoid', 'tanh', 'relu', 'elu'] # 지원하는 활성화 함수 리스트
  optimizerfunc_list=[keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
                      keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                      keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=True),
                      keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)] # 지원하는 최적화 함수 리스트

  lossfunc=['mse','categorical_crossentropy']
  outlayer_activfunc_list=['linear','softmax']
  metrics=[['mae','mse'],['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]]

  def __init__(self, data_to_use, target_name):
    self.purpose=int(input("Which model do you want to create, classification or regression?\n1.Regression  2.Classification\n"))-1

    fraction=float(input('Proportion of Train Data you want(e.g. 0.8) : '))
    # 훈련용과 평가용 데이터를 나눌 비율을 입력받고, 해당 비율로 나누기
    Train_set=data_to_use.sample(frac=fraction,random_state=0).sample(frac=1)
    Test_set=data_to_use.drop(Train_set.index).sample(frac=1)
    # 나눠진 두 데이터셋에서 타겟 변수 분리하기
    Train_labels = pd.DataFrame(Train_set.pop(target_name))
    Test_labels = pd.DataFrame(Test_set.pop(target_name))
    # 타겟 변수(레이블) 인코딩, 타겟변수가 분리된 두 데이터셋 정규화,
    
    self.number_of_hidden=None
    self.setup_data=[]

    self.model=None
    self.history=None
    self.optimizer=None

    self.train_data=Train_set
    self.train_labels=pd.get_dummies(Train_labels) if self.purpose else Train_labels
    self.test_data=Test_set
    self.test_labels=pd.get_dummies(Test_labels) if self.purpose else Test_labels
  
  def SetUp(self):
    self.number_of_hidden=int(input('\nThe number of hidden layers you want to add to the model : ')) # 원하는 은닉층 수 입력
    # 입력층에 쓰일 활성화 함수 입력
    print('\nThe Input Layer')
    temp=[self._WhichActivfunc(),len(self.train_data.keys())]
    self.setup_data.append(temp)
    # 은닉층에 쓰일 활성화 함수와 노드 수 입력
    for i in range(self.number_of_hidden):
      print('\nThe Hidden Layer %i'%(i+1)) # 입력받은 은닉층 수만큼 프린트
      temp=[self._WhichActivfunc(),int(input('Number of nodes : '))]
      self.setup_data.append(temp)
    print('\nThe Outer Layer\n')

  def _WhichActivfunc(self):
    hiddenlayer_activfunc=int(input('Activation function (1.Softmax  2.Sigmoid  3.tanh  4.ReLU  5.ELU) : '))-1 # 각각의 활성화 함수 번호 입력
    return ANN.activfunc_list[hiddenlayer_activfunc]

  def DesignLayer(self):
    self.model=keras.Sequential()
    for i in range(self.number_of_hidden+1): # 층 수만큼 반복
      self.model.add(layers.Dense(self.setup_data[i][1], activation=self.setup_data[i][0])) # 입력받은 노드의 개수와 활성화 함수를 바탕으로 층 추가
    self.model.add(layers.Dense(len(self.train_labels.keys()),activation=ANN.outlayer_activfunc_list[self.purpose])) # 출력층 구성
    optimizer_num=int(input("Which optimization function will you use?\n1.Gradient descent\n2.RMSprop\n3.NAG\n4.NAdam\n"))-1
    self.model.compile(loss=ANN.lossfunc[self.purpose], optimizer=ANN.optimizerfunc_list[optimizer_num], metrics=ANN.metrics[self.purpose]) # 오차를 측정하는 방법으로 mae와 mse를 사용

  def _Norm(self,x): # 열 별로 정규화해주는 함수
    train_stats=x.describe() # 데이터 x의 기본적인 통계값 계산하여 train_stats에 저장
    train_stats = train_stats.transpose()
    return (x - train_stats['mean']) / train_stats['std'] # 그중 평균과 분산을 이용하여 정규화

  def Train(self):
    # 원하는 반복 횟수 입력
    EPOCHS = int(input('\nNumber of times to repeat model training (Epoch) : '))
    # 입력받은 반복 횟수만큼 학습
    self.history = self.model.fit(
      self._Norm(self.train_data), self.train_labels,
      epochs=EPOCHS, validation_split = 0.2, verbose=0, validation_data=(self._Norm(self.train_data), self.train_labels),
      callbacks=[PrintDot()]) # 무작위로 검증데이터를 20% 선택하여 학습 진행

  def PlotByEpoch(self):
    hist = pd.DataFrame(self.history.history) # 반복 횟수별로 검증정확도를 데이터프레임 형태로 바꾸기
    hist['epoch'] = self.history.epoch # epoch 값 나타내는 열 추가

    plt.figure(figsize=(6,4)) # 그래프 크기 설정

    plt.xlabel('Epoch') # x축 이름 붙이기
    plt.ylabel('Loss') # y축 이름 붙이기
    plt.plot(hist['epoch'], hist['loss'],
           label='Train Error') # 학습데이터를 기준으로 정확도 채점했을 때 그래프
    plt.plot(hist['epoch'], hist['val_loss'],
           label = 'Val Error') # 검증데이터를 기준으로 정확도 채점했을 때 그래프
    plt.legend()
  
  def Evaluate(self):
    loss, *arg=self.model.evaluate(self._Norm(self.test_data), self.test_labels, verbose=2)
    if self.purpose:
      precision=arg[2]
      recall=arg[1]
      f1_score=2*(precision*recall)/(precision+recall)
      print('\nLoss : {0}\nAccuracy : {1}\nF1-score : {2}'.format(loss, arg[0], f1_score))
    else:
      print('\nMean Squared Error : {0}\nMean Absolute Error : {1}'.format(loss, arg[0]))

  def ResultPlot(self):
    test_predictions = self.model.predict(self._Norm(self.test_data)).flatten() # 평가용 데이터를 이용해 예측값을 직접 구하기
    # 평가용 데이터를 이용해 구한 예측값을 평가용 레이블의 실측값과 비교하는 그래프 그리기
    plt.scatter(self.test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

  def Save(self):
    ModelName=input("Name your model to save to 'ANN_Model' folder : ") # 모델 이름 입력
    keras_model_path = "/content/drive/MyDrive/ANN_Model/%s"%ModelName # 각자의 드라이브에 ANN_Model 폴더를 만들어 그 안에 저장
    self.model.save(keras_model_path)  # 케라스 API 모델 저장
