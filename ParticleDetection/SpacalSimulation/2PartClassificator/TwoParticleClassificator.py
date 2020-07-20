def train(enThreshold, KFold, model, variant, learning_rate, BATCH_SIZE_Factor, EPOCHS, loss_function, optimizer):
    '''
    # enThreshold: Many cells have very small values of desposited energy. Replace values below a certain threshold with zero.
    # KFold: ratio of train and test. Example: KFold of "5" would be train 80% and test 20%, KFold of "2" would be train and test 50%.
    # model: number of net chosen from the file "models.py".
    # variant: variant of the net chosen from the file "models.py".
    # learning_rate: learning rate for optimization function.
    # BATCH_SIZE_Factor: it is just prepared, but does not work in this version.
    # EPOCHS: Number of passes through all the dataset.
    # loss_function: 0 is CrossEntropyLoss and 1 NLLLoss, but you can edit the code to add as many as you want.
    # optimizer: 0 is Adam, 1 is SGD and 2 is Adamax, but you can edit the code to add as many as you want.    
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mlflow
    from models import modelSelector
    from sklearn import preprocessing
    from sklearn.metrics import confusion_matrix

    import sys

    from timeit import default_timer as timer #Calculate time in GPU and CPU.
    import torch.nn as nn # torch.nn gives us access to some helpful neural network things, such as fully-connected layers, convolutional layers (for imagery), recurrent layers, ...
    import torch.nn.functional as F # handy functions like RELu (activation functions).
    from sklearn.model_selection import train_test_split #The name is self explainatory.

    
    spacal_df = pd.read_hdf('spacal_at_10mm_neutrals.h5') #ToDo: Write the path where you store the hdf files.

    #Change ids for integrity
    spacal_df['class'] = spacal_df['class'].map(lambda x: np.where(x == 2, 0, x))
    spacal_df['class'] = spacal_df['class'].map(lambda x: np.where(x == 10, 1, x))

    particleNum = 2


    #Replacement with threshold value:
    spacal_df['EnergyFront'] = spacal_df['EnergyFront'].map(lambda x: np.where(x < enThreshold, 0., x))
    spacal_df['EnergyRear'] = spacal_df['EnergyRear'] .map(lambda x: np.where(x < enThreshold, 0., x))


    #Get maximums and minimums in every cell:
    maxEnergyFront = np.array([])
    minEnergyFront = np.array([])
    maxEnergyRear = np.array([])
    minEnergyRear = np.array([])


    for row in spacal_df.index:
        maxEnergyFront = np.append(maxEnergyFront, np.amax(spacal_df["EnergyFront"][row]))
        minEnergyFront = np.append(minEnergyFront, np.amin(spacal_df["EnergyFront"][row]))
        maxEnergyRear = np.append(maxEnergyRear, np.amax(spacal_df["EnergyRear"][row]))
        minEnergyRear = np.append(minEnergyRear, np.amin(spacal_df["EnergyRear"][row]))
        

    spacal_df["MaxEnergyFront"] = maxEnergyFront
    spacal_df["MinEnergyFront"] = minEnergyFront
    spacal_df["MaxEnergyRear"] = maxEnergyRear
    spacal_df["MinEnergyRear"] = minEnergyRear


    
    #Once the data has been checked, it is possible to convert front and rear cells to one single image multichannel:
    spacal_df["EnergyFrontRear"] = list(map(lambda x,y: np.append(x,y), spacal_df["EnergyFront"], spacal_df["EnergyRear"]))


    # we can put our network on our GPU. To do this, we can just set a flag like:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    


    ##############################
    ###Section to record losses###
    ##############################

    #Add more utility to keep track of loss values and plot them at each epoch:
    from IPython.display import clear_output

    resultsDF = pd.DataFrame()


    class Logger:
      def __init__(self):
        self.train_loss_batch = []
        self.train_loss_epoch = []
        self.test_loss_batch = []
        self.test_loss_epoch = []
        self.train_batches_per_epoch = 0
        self.test_batches_per_epoch = 0
        self.epoch_counter = 0

      def fill_train(self, loss):
        self.train_loss_batch.append(loss)
        self.train_batches_per_epoch += 1

      def fill_test(self, loss):
        self.test_loss_batch.append(loss)
        self.test_batches_per_epoch += 1

      def finish_epoch(self):
        self.train_loss_epoch.append(np.mean(
            self.train_loss_batch[-self.train_batches_per_epoch:]
        ))
        self.test_loss_epoch.append(np.mean(
            self.test_loss_batch[-self.test_batches_per_epoch:]
        ))
        self.train_batches_per_epoch = 0
        self.test_batches_per_epoch = 0
        
        clear_output()
      
        print("epoch #{} \t train_loss: {:.8} \t test_loss: {:.8}".format(
                  self.epoch_counter,
                  self.train_loss_epoch[-1],
                  self.test_loss_epoch[-1]
              ))
        
        self.epoch_counter += 1



    #############################
    ###Other functions section###
    #############################


    #This reshape class will be needed to reshape convolutions.
    class Reshape(torch.nn.Module):
        def __init__(self, *shape):
            super(Reshape, self).__init__()
            self.shape = shape

        def forward(self, x):
            return x.reshape(*self.shape)


    #Convert train and test data to tensors
    def setData(X_train, X_test, y_train, y_test, maxEnergyFrontRear, originalLen):
        
        X_train, X_test = X_train/maxEnergyFrontRear, X_test/maxEnergyFrontRear

        try:
            #Case 1: index 0 is in X_train
            X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True) 
            newIndex = []
            for i in range(originalLen):
                try:
                    newIndex.append(X_test[i])
                except:
                    pass
            X_test = torch.tensor(newIndex, dtype=torch.float32, requires_grad=True)
        except:
            #Case 2: index 0 is in X_test
            X_test = torch.tensor(X_test,  dtype=torch.float32, requires_grad=True) 
            newIndex = []
            for i in range(originalLen):
                try:
                    newIndex.append(X_train[i])
                except:
                    pass
            X_train = torch.tensor(newIndex, dtype=torch.float32, requires_grad=True)

        try:
            #Case 1: index 0 is in y_train
            y_train = torch.tensor(y_train, dtype=torch.long) 
            newIndex = []
            for i in range(originalLen):
                try:
                    newIndex.append(y_test[i])
                except:
                    pass
            y_test = torch.tensor(newIndex, dtype=torch.long)
        except:
            #Case 2: index 0 is in X_test
            y_test = torch.tensor(y_test,  dtype=torch.long) 
            newIndex = []
            for i in range(originalLen):
                try:
                    newIndex.append(y_train[i])
                except:
                    pass
            y_train = torch.tensor(newIndex, dtype=torch.long)


        return X_train, X_test, y_train, y_test   
    

    #Optimizer function. You can add as many as you want.
    def getOptimizer(i, parameters, lr):
        '''
        requieres torch
        '''
        if(i == 0):
            return torch.optim.Adam(parameters, lr)
        if(i == 1):
            return torch.optim.SGD(parameters, lr)
        if(i == 2):
            return torch.optim.Adamax(parameters, lr)

    #Loss functions. You can add as many as you want.
    def getLoss(i):
        '''
        requires torch
        inspired by: https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
        '''
        if(i == 0):
            return torch.nn.CrossEntropyLoss() 
        if(i == 1):
            return torch.nn.NLLLoss() 

    #In other versions I would use this function to call a confusion matrix, here it just returns the accuracy and loss.
    def getAccuracy(model, X_test, y_test, device, logger, loss_function, showCnf):
        correct = 0
        total = 0
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        with torch.no_grad():
            output = model(X_test)
            loss = loss_function(output, y_test)
            logger.fill_test(loss.item())
            for idx, i in enumerate(output):
                if torch.argmax(i) == y_test[idx]:
                    correct += 1
                total += 1

            #if(showCnf): printConfusionMatrix(output, y_test)#Uncommenting this line would generate an error due to lack of code.

        return round(correct/total, 3)*100, logger


    #This is the function with which the model is trained. It is prepared to work with different batch sizes and epochs that worked in other versions.
    def trainCNN(spacal_df, KFold, device, model, learning_rate, BATCH_SIZE_Factor, EPOCHS, loss_function, optimizer):
        '''
        Requires setData.
        Requires getLoss.
        Requires getOptimizer.
        '''

        HeightRefFront = len(spacal_df.iloc[0,2]) # Height reference value in the front cells.
        LongRefFront = len(spacal_df.iloc[0,2][0]) # Lognitude reference value in the front cells.

        start = timer()       
        
        originalLen = len(spacal_df["EnergyFrontRear"])

        maxEnergyFrontRear = max(spacal_df["MaxEnergyRear"].max(),spacal_df["MaxEnergyFront"].max()) #Get the max value for both channels

        X_train, X_test, y_train, y_test = train_test_split(spacal_df["EnergyFrontRear"], spacal_df["class"], test_size=(1/KFold))     
        X_train, X_test, y_train, y_test = setData(X_train, X_test, y_train, y_test, maxEnergyFrontRear, originalLen)

        #ToDo: Here you can modify the code to test with different batches and batch size.
        #ToDo: Here you can modify the code to test with different Cross-Validation techniques.
                

        logger = Logger()
        loss_function = getLoss(loss_function)
        optimizer = getOptimizer(optimizer, model.parameters(), learning_rate)
        
        for i_epoch in range(EPOCHS):
            model.zero_grad()  
            output = model(X_train.to(device)) # pass the reshaped batch
            loss = loss_function(output, y_train.to(device)) 
            logger.fill_train(loss.item())
            loss.backward()
            optimizer.step() 
            
            showCnf = True if i_epoch + 1 == EPOCHS else False
            accuracyPctg, logger = getAccuracy(model, X_test, y_test, device, logger, loss_function, showCnf) #It will show cnfMatrix just in the last epoch.
            logger.finish_epoch()


        time = round(timer() - start)

        #Return the model, the time, the accuracy and the loss of every epoch
        return time, accuracyPctg, logger.test_loss_epoch, logger.train_loss_epoch 


    def save2mlflow(time, accuracyPctg, enThreshold = enThreshold, KFold = KFold, model = model, variant = variant, learning_rate = learning_rate, BATCH_SIZE_Factor = BATCH_SIZE_Factor, EPOCHS = EPOCHS, loss_function = loss_function, optimizer = optimizer):
        mlflow.log_param("enThreshold", enThreshold)
        mlflow.log_param("KFold", KFold)
        mlflow.log_param("model", model)
        mlflow.log_param("variant", variant)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("BATCH_SIZE_Factor", BATCH_SIZE_Factor)
        mlflow.log_param("EPOCHS", EPOCHS)
        mlflow.log_param("loss_function", loss_function)
        mlflow.log_param("optimizer", optimizer)
        
        mlflow.log_metric("time", time)
        mlflow.log_metric("accuracyPctg", accuracyPctg)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="2ParticlesCNN")
        else:
            mlflow.sklearn.log_model(lr, "model")


    def combinationsCNN(spacal_df, KFold, device, model, variant, learning_rate, dropout_rate, BATCH_SIZE_Factor, EPOCHS, loss_function, optimizer):

        model = modelSelector(device, model, variant)
        
        return trainCNN(spacal_df, KFold, device, model, learning_rate, BATCH_SIZE_Factor, EPOCHS, loss_function, optimizer)



    with mlflow.start_run():
        combinationsResult = combinationsCNN(spacal_df, KFold, device, model, variant, learning_rate, 
                                             '', BATCH_SIZE_Factor, EPOCHS, loss_function, optimizer)
        
        #Save the results into mlflow.    
        save2mlflow(combinationsResult[0], combinationsResult[1], device)


