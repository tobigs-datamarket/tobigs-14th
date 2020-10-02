import numpy as np

#시그모이드 함수
def sigmoid(x):
        return 1 / (1+np.exp(-x))

#편미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index        
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = tmp_val - delta_x 
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val 
        it.iternext()   

    return grad

#========================================================================================#

class LogisticRegression_cls:
    def __init__(self, X_train, y_train, learning_rate = 1e-2): #learning_rate를 기본값으로 준다!
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.W = np.random.rand(1,1)  # W와 b는 class안에서만 사용되는 랜덤값이기때문에 이렇게 적어준다.
        self.b = np.random.rand(1)  
        
        
        
    #손실함수
    def loss_func(self):
        
        delta = 1e-7    # log 무한대 발산 방지
    
        z = np.dot(self.X_train,self.W) + self.b
        y = sigmoid(z)
    
        # cross-entropy 
        return  -np.sum(self.y_train*np.log(y + delta) + (1-self.y_train)*np.log((1 - y)+delta ) )
        
        
        
    #손실 값 계산 함수
    def error_val(self):
        return self.loss_func()  #손실함수인 loss_func에 계산까지 다 들어있기때문에 loss_func을 가져와 return해준다.
    
             
    #예측 함수
    def predict(self, X_test): #X_test는 class에 기본인수로 들어가는 것이 아니기때문에 self를 붙여주지않아도 된다.
        result=[]
        for x in X_test:
            z=np.dot(x, self.W) + self.b
            y=sigmoid(z)

            if y > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result
    
    #학습 함수
    def train(self):
        f = lambda x : self.loss_func()  # f(x) = loss_func(x_data, t_data)

        print("Initial error value = ", self.error_val())

        for step in  range(10001):  
            
            self.W -= self.learning_rate * numerical_derivative(f, self.W)
    
            self.b -= self.learning_rate * numerical_derivative(f, self.b)
    
            if (step % 400 == 0):
                print("step = ", step, "error value = ", self.error_val())