def modelSelector(device, modelNumber, variant = 0):
    import torch
    import torch.nn as nn # torch.nn gives us access to some helpful neural network things, such as fully-connected layers, convolutional layers (for imagery), recurrent layers, ...
    import torch.nn.functional as F # handy functions like RELu (activation functions).

    #This reshape class will be needed to reshape convolutions.
    class Reshape(torch.nn.Module):
        def __init__(self, *shape):
            super(Reshape, self).__init__()
            self.shape = shape

        def forward(self, x):
            return x.reshape(*self.shape)
    
    if(modelNumber == 0):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [6, 6],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.LeakyReLU(negative_slope=0.001), nn.LeakyReLU(negative_slope=0.1)],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1],
                    'out_channels4': [12, 12],
                    'kernel_size4': [2, 2],
                    'activation5': [nn.LeakyReLU(negative_slope=0.001), nn.LeakyReLU(negative_slope=0.1)],
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Conv2d(in_channels = variants["out_channels1"][variant], out_channels = variants["out_channels4"][variant], kernel_size = variants["kernel_size4"][variant]), 
                              variants["activation5"][variant], 
                              nn.Flatten(), nn.Linear(in_features=432, out_features=2), 
                              nn.Softmax(dim=1))
        return model.to(device)

    if(modelNumber == 0):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [6, 6],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.LeakyReLU(negative_slope=0.001), nn.LeakyReLU(negative_slope=0.1)],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1],
                    'out_channels4': [12, 12],
                    'kernel_size4': [2, 2],
                    'activation5': [nn.LeakyReLU(negative_slope=0.001), nn.LeakyReLU(negative_slope=0.1)],
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Conv2d(in_channels = variants["out_channels1"][variant], out_channels = variants["out_channels4"][variant], kernel_size = variants["kernel_size4"][variant]), 
                              variants["activation5"][variant], 
                              nn.Flatten(), nn.Linear(in_features=432, out_features=12), 
                              nn.Softmax(dim=1))
        return model.to(device)

    if(modelNumber == 1):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [6, 6],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.LeakyReLU(negative_slope=0.001), nn.LeakyReLU(negative_slope=0.1)],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1],
                    'out_channels4': [12, 12],
                    'kernel_size4': [2, 2],
                    'activation5': [nn.LeakyReLU(negative_slope=0.001), nn.LeakyReLU(negative_slope=0.1)],
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Flatten(), 
                              nn.Linear(in_features=294, out_features=128),
                              nn.ReLU(), 
                              nn.Linear(in_features=128, out_features=32), 
                              nn.ReLU(), 
                              nn.Linear(in_features=32, out_features=12), 
                              nn.Softmax(dim=1))
        return model.to(device)

    if(modelNumber == 2):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [6, 6],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.ReLU(), nn.LeakyReLU()],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1],
                    'out_channels4': [12, 12],
                    'kernel_size4': [2, 2],
                    'activation5': [nn.ReLU(), nn.LeakyReLU()],
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Flatten(), 
                              nn.Linear(in_features=294, out_features=512),
                              nn.ReLU(), 
                              nn.Linear(in_features=512, out_features=128), 
                              nn.ReLU(), 
                              nn.Linear(in_features=128, out_features=32), 
                              nn.ReLU(), 
                              nn.Linear(in_features=32, out_features=12), 
                              nn.Softmax(dim=1))
        return model.to(device)



    if(modelNumber == 3):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [6, 6],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.ReLU(), nn.LeakyReLU()],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1],
                    'out_channels4': [12, 12],
                    'kernel_size4': [2, 2],
                    'activation5': [nn.ReLU(), nn.LeakyReLU()],
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Flatten(), 
                              nn.Linear(in_features=294, out_features=512),
                              nn.ReLU(), 
                              nn.Linear(in_features=512, out_features=128), 
                              nn.ReLU(), 
                              nn.Linear(in_features=128, out_features=12), 
                              nn.Softmax(dim=1))
        return model.to(device)


    if(modelNumber == 4):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [16, 16, 16, 16, 32],
                    'kernel_size1': [2, 2, 2, 2, 2],
                    'activation2': [nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01)],
                    'kernel_size3': [2, 2, 2, 2, 2],
                    'stride3': [1, 1, 1, 1, 1],
                    'padding3': [1, 1, 1, 1, 1],
                    'out_channels4': [12, 12, 12, 12, 12],
                    'kernel_size4': [2, 2, 2, 2, 2],
                    'activation5': [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01), nn.ReLU(), nn.ReLU()],
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Conv2d(in_channels = variants["out_channels1"][variant], out_channels = variants["out_channels4"][variant], kernel_size = variants["kernel_size4"][variant]), 
                              variants["activation5"][variant], 
                              nn.Flatten(), nn.Linear(in_features=432, out_features=12), 
                              nn.Softmax(dim=1))
        return model.to(device)

    if(modelNumber == 5):
            
            #Number is just the row number, not the layer.
            variants = {'out_channels1': [6, 6],
                        'kernel_size1': [2, 2],
                        'activation2': [nn.ReLU(), nn.LeakyReLU()],
                        'kernel_size3': [2, 2],
                        'stride3': [1, 1],
                        'padding3': [1, 1]
                        }
                        
            #Reset model so previous trains dont affect
            try: del model
            except: pass 
            model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                                nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                                variants["activation2"][variant], 
                                nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                                nn.Flatten(), 
                                nn.Linear(in_features=294, out_features=64),
                                nn.ReLU(), 
                                nn.Linear(in_features=64, out_features=8), 
                                nn.ReLU(), 
                                nn.Linear(in_features=8, out_features=12), 
                                nn.Softmax(dim=1))
            return model.to(device)

    if(modelNumber == 6):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [4, 4],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.ReLU(), nn.LeakyReLU()],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1]
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                            nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                            variants["activation2"][variant], 
                            nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                            nn.Flatten(), 
                            nn.Linear(in_features=196, out_features=8),
                            nn.ReLU(), 
                            nn.Linear(in_features=8, out_features=12), 
                            nn.Softmax(dim=1))
        return model.to(device)


    if(modelNumber == 7):
        
        #Number is just the row number, not the layer.
        variants = {'out_channels1': [4],
                    'kernel_size1': [2],
                    'activation2': [nn.ReLU()],
                    'kernel_size3': [2],
                    'stride3': [1],
                    'padding3': [1]
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                            nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                            variants["activation2"][variant], 
                            nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                            nn.Flatten(), 
                            nn.Linear(in_features=196, out_features=8),
                            nn.ReLU(), 
                            nn.Linear(in_features=8, out_features=12), 
                            nn.Sigmoid())
        return model.to(device)

    if(modelNumber == 8):
        
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                            nn.Flatten(), 
                            nn.Linear(in_features=7*7*2, out_features=64),
                            nn.ReLU(), 
                            nn.Linear(in_features=64, out_features=64),
                            nn.ReLU(), 
                            nn.Linear(in_features=64, out_features=12), 
                            nn.LogSoftmax(dim=1))
        return model.to(device)


    if(modelNumber == 9):
        variants = {'out_channels1': [4],
                    'kernel_size1': [2],
                    'activation2': [nn.ReLU()],
                    'kernel_size3': [2],
                    'stride3': [1],
                    'padding3': [1]
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Flatten(), 
                              nn.Linear(in_features=196, out_features=32),
                              nn.ReLU(), 
                              nn.Linear(in_features=32, out_features=12),
                              nn.LogSoftmax(dim=1))
        return model.to(device)


    if(modelNumber == 10):
        variants = {'out_channels1': [4],
                    'kernel_size1': [2],
                    'activation2': [nn.ReLU()],
                    'kernel_size3': [2],
                    'stride3': [1],
                    'padding3': [1]
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Flatten(), 
                              nn.Linear(in_features=196, out_features=32),
                              nn.ReLU(), 
                              nn.Linear(in_features=32, out_features=12),
                              nn.LogSigmoid())
        return model.to(device)

    if(modelNumber == 11):
        variants = {'out_channels1': [4, 4, 4, 4, 4, 4],
                    'kernel_size1': [2, 2, 2, 2, 2, 2],
                    'activation2': [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.LogSoftmax(), nn.Softmax(), nn.ReLU()],
                    'kernel_size3': [2, 2, 2, 2, 2, 2],
                    'stride3': [1, 1, 1, 1, 1, 1],
                    'padding3': [1, 1, 1, 1, 1, 1],
                    'activationFC':[nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.LogSoftmax(), nn.Softmax(), nn.ReLU()],
                    'activationFC_out':[nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.LogSoftmax(), nn.Softmax(), nn.Softmax()],
                    }
                    
        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7), 
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Flatten(), 
                              nn.Linear(in_features=196, out_features=32),
                              variants["activationFC"][variant], 
                              nn.Linear(in_features=32, out_features=12),
                              variants["activationFC_out"][variant])
        return model.to(device)


    if(modelNumber == 12):

        #Number is just the row number, not the layer.
        variants = {'out_channels1': [16, 16],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01)],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1],
                    'out_channels4': [12, 12],
                    'kernel_size4': [2, 2],
                    'activation5': [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)],
                    }

        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7),
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Conv2d(in_channels = variants["out_channels1"][variant], out_channels = variants["out_channels4"][variant], kernel_size = variants["kernel_size4"][variant]), 
                              variants["activation5"][variant], 
                              nn.Flatten(), 
                              nn.Linear(in_features=432, out_features=256), 
                              nn.Linear(in_features=256, out_features=64),
                              nn.Linear(in_features=64, out_features=32),
                              nn.Linear(in_features=32, out_features=12),
                              nn.Softmax(dim=1))
        return model.to(device)

    if(modelNumber == 13):

        #Number is just the row number, not the layer.
        variants = {'out_channels1': [16, 16],
                    'kernel_size1': [2, 2],
                    'activation2': [nn.LeakyReLU(negative_slope=0.01), nn.LeakyReLU(negative_slope=0.01)],
                    'kernel_size3': [2, 2],
                    'stride3': [1, 1],
                    'padding3': [1, 1],
                    'out_channels4': [12, 12],
                    'kernel_size4': [2, 2],
                    'activation5': [nn.ReLU(), nn.LeakyReLU(negative_slope=0.01)],
                    }

        #Reset model so previous trains dont affect
        try: del model
        except: pass 
        model = nn.Sequential(Reshape(-1, 2, 7, 7),
                              nn.Conv2d(in_channels = 2, out_channels = variants["out_channels1"][variant], kernel_size = variants["kernel_size1"][variant]), 
                              variants["activation2"][variant], 
                              nn.MaxPool2d(kernel_size = variants["kernel_size3"][variant], stride = variants["stride3"][variant], padding = variants["padding3"][variant]), 
                              nn.Conv2d(in_channels = variants["out_channels1"][variant], out_channels = variants["out_channels4"][variant], kernel_size = variants["kernel_size4"][variant]), 
                              variants["activation5"][variant], 
                              nn.Flatten(), 
                              nn.Linear(in_features=432, out_features=256), 
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(in_features=256, out_features=64),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(in_features=64, out_features=32),
                              nn.LeakyReLU(negative_slope=0.01),
                              nn.Linear(in_features=32, out_features=12),
                              nn.Softmax(dim=1))
        return model.to(device)
