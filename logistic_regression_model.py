import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(x))

class LogisticRegression:
    def __init__(self, X_train, y_train, X_test, y_test, num_iterations=2000, learning_rate = 0.1):
        self.B =  np.zeros(1, X_train.shape[1])
        self.b = 0
        self.model(X_train, y_train, X_test, y_test, num_iterations, learning_rate)

    def propagate(self, B, b, X, y):
        m = X.shape[0]
        z = np.dot(X, B.T) + b
        y_hat = sigmoid(z)
        cost = (-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        dB = (1 / m) * np.dot((y_hat - y).T, X)
        assert (dB.shape[1] == B.shape[1])
        db = (1 / m) * np.sum(y_hat - y)
        return cost, dB, db

    def optimize(self,B, b, X, y, num_iterations, learning_rate):
        costs = []
        for i in range(num_iterations):
            cost, dB, db = self.propagate(B, b, X, y)
            B = B - learning_rate * dB
            b = b - learning_rate * db

            costs.append(cost)

        return B, b, dB, db, costs

    def predict(B, b, X):
        m = X.shape[0]
        y_prediction = np.zeros((m, 1))
        beta = B.reshape(1, X.shape[1])
        y_hat = sigmoid(np.dot(X, beta.T) + b)
        for i in range(y_hat.shape[0]):
            # Convert probabilities to actual predictions p
            if y_hat[i, 0] >= 0.5:
                y_prediction[i, 0] = 1
            else:
                y_prediction[i, 0] = 0

        assert (y_prediction.shape == (m, 1))
        return y_prediction

    def model(self, X_train, y_train, X_test, y_test, num_iterations, learning_rate):
        beta = self.B
        beta_zero = self.b
        B, b, dB, db, costs = self.optimize(beta, beta_zero, X_train, y_train, num_iterations, learning_rate)
        y_prediction_test = self.predict(B, b, X=X_test)
        y_prediction_train = self.predict(B, b, X=X_train)
        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

        model_result = {"costs": costs,
             "y_prediction_test": y_prediction_test,
             "y_prediction_train": y_prediction_train,
             "B": B,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return model_result
