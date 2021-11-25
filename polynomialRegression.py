import numpy as np
import random
import matplotlib.pyplot as plt

# w1*x^1 + w2*x^2 + .... + xn*x^n
def prodottoPotenze(x,w):
    pred = w[0]
    for i in range(1,len(w)):
        pred = pred + w[i]*x**(i)
    return pred

def funzioneCosto(x,w,y):
    return np.sum((y - prodottoPotenze(x,w))**2)

def meanSquaredError(x,w,y):
    return funzioneCosto(x,w,y) / np.size(x,0)

def gradientDescent(x,w,y,learningRate,epsilon):
    costHistory = []
    gradient = np.zeros(len(w))
    while True:
        #Calcolo gradiente della funzione di costo
        for i in range(len(w)):
            gradient[i] = (-2) * np.sum((y - prodottoPotenze(x,w))*x**i)
        if(np.linalg.norm(gradient)<=epsilon):
            break
        else:
            costHistory.append(funzioneCosto(x,w,y))
            for i in range(len(w)):
                w[i] = w[i] - (learningRate * gradient[i])
            #print(funzioneCosto(x,w,y))
    plt.title("Andamento funzione di costo")
    plt.xlabel("Num iterazioni")
    plt.ylabel("Loss function value")
    plt.scatter(range(len(costHistory)),costHistory,  color='red')
    plt.show()
    return w


    

#Set del modello che si vuole adottare (lineare, quadratico,......)
gradoPolinomio = 7;

#Creazione dataset monodimensionale
X = np.sort(np.array(np.random.uniform(low=-1.0, high=1.0, size=(130,)),float).reshape(-1,1),axis=0)
Y = np.array([x+random.uniform(-3.0,3.0) for x in list(-12*(X**7)-10*(X**6)+13*(X**5)+6*(X**4)-20*(X**3)+4*(X**2) + 12*X + 1)],float).reshape(-1,1)
W = np.random.rand(gradoPolinomio+1).reshape(-1,1) #n pesi polinomio + w0 per intersezione asse y


#Suddivisione dataset in training set e test set
xTrain = np.empty(0,float).reshape(-1,1)
xTest  = np.empty(0,float).reshape(-1,1)

yTrain = np.empty(0,float).reshape(-1,1)
yTest =  np.empty(0,float).reshape(-1,1)

# 70% training, 30% test
for i in [0,10,20,30,40,50,60,70,80,90,100,110,120]:
    xTrain = np.append(xTrain,X[i:i+7,:],axis=0)
    yTrain = np.append(yTrain,Y[i:i+7,:],axis=0)

    xTest = np.append(xTest,X[i+7:i+10,:],axis=0)
    yTest = np.append(yTest,Y[i+7:i+10,:],axis=0)

#Addestramento modello
wOpt = gradientDescent(xTrain, W, yTrain, 0.001, 9.0)

#Plot del test set + modello addestrato
plt.scatter(xTest,yTest,color="blue",label='Test')
plt.plot(xTest,(prodottoPotenze(xTest,wOpt)),"green",label='Model')
plt.legend(scatterpoints=1, loc='lower left', ncol=1,fontsize=8)
plt.title("Trained model (TestSet Only) | "+str(np.size(wOpt,0))+" parameters")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Plot del dataset con il modello addestrato
plt.scatter(xTrain,yTrain,color="red",label='Training')
plt.scatter(xTest,yTest,color="blue",label='Test')
plt.plot(X,(prodottoPotenze(X,wOpt)),"green",label='Model')
plt.legend(scatterpoints=1, loc='lower left', ncol=1,fontsize=8)
plt.title("Trained model (Dataset) | "+str(np.size(wOpt,0))+" parameters")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Output dati ottenuti
print("Numero pattern training:\n"+str(np.size(xTrain,0))+"\n")
print("Numero pattern test:\n"+str(np.size(xTest,0))+"\n")
print("Pesi ottenuti:")

for i in range(np.size(wOpt,0)):
    print("w"+str(i)+"\t=\t"+ str(wOpt[i]))

mse = meanSquaredError(xTrain, wOpt, yTrain)
print("\nMean squared error training set:\n"+str(mse))
mse = meanSquaredError(xTest, wOpt, yTest)
print("\nMean squared error test set:\n"+str(mse))
