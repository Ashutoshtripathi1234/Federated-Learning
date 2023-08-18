import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#accuracy=[]
loss=[]


def fit_round(server_round):
    return {"server_round": server_round}





def get_eval_fn(model):
    df = pd.read_csv(r'C:\Users\lenovo\Downloads\fl-main\fl-main\LogisticRegressionFL\dataset\textdatamy.csv')
    y = df["HeartDisease"].to_numpy(int, copy=False)
    x = df.iloc[:, 1:-1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=4)

    def eval(server_round, parameters, config):
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]
        #loss = log_loss(y_test, model.predict_proba(x_test))
        #accuracy .append(model.score(x_test, y_test))
        loss.append(model.score(y_test,model.predict(x_test)))

        # # Plot accuracy/model loss versus round
        # plt.scatter(server_round, accuracy, color="blue")
        # plt.scatter(server_round, loss, color="red")
        # plt.title("Test Accuracy/model loss vs Communication Rounds")
        # plt.xlabel("Communication Rounds")
        # plt.ylabel("Test Accuracy/model loss")
        # plt.legend(['accuracy', 'model loss'], loc='center right')
        # plt.savefig("accuracy_vs_rounds.png")

        # if server_round ==100:
        #
        #     y_pred = (model.predict_proba(x_test)[:, 1] >= 0.5).astype(int)
        #
        #     cm = confusion_matrix(y_test, y_pred)
        #     TP = cm[0][0]
        #
        #     TN = cm[1][1]
        #
        #     FP = cm[0][1]
        #     FN = cm[1][0]
        #     Accuracy = (TP + TN) / (TP + TN + FN + FP)
        #     Precision = TP / (TP + FP)
        #     Recall = TP / (TP + FN)
        #     F1_Score = 2 * Precision * Recall / (Precision + Recall)
        #     Eval_Metrics = [Accuracy, Precision, Recall, F1_Score]
        #     Metric_Names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        #     Metrics_pos = np.arange(len(Metric_Names))
        #     plt.bar(Metrics_pos, Eval_Metrics)
        #     plt.xticks(Metrics_pos, Metric_Names)
        #     plt.title('Accuracy v Precision v  Recall  v F1 Score of the FL model')
        #     plt.show()
        #     print(Accuracy, Precision, Recall, F1_Score)


        return loss, {"accuracy": accuracy}

    return eval


if __name__ == "__main__":
    # Create strategy and run server
    model = LogisticRegression()
    n_classes = 2
    n_features = 9
    model.classes_ = np.array([i for i in range(2)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )

    # Start Flower server for five rounds of federated learning
    fl.server.start_server(
        server_address="127.0.0.1:1023",
        config=fl.server.ServerConfig(num_rounds=100),
        strategy=strategy
    )
# Plot accuracy/model loss versus round:
server_round=list(range(0,101))
plt.plot(server_round, loss, color="blue")
#plt.scatter(server_round, loss, color="red")
plt.title("model loss vs Communication Rounds")
plt.xlabel("Communication Rounds")
plt.ylabel("model loss")
plt.legend(['accuracy', 'model loss'], loc='center right')
plt.savefig("accuracy_vs_rounds.png")

# deep neural network
# df = pd.read_csv(r'C:\Users\lenovo\Downloads\fl-main\fl-main\LogisticRegressionFL\dataset\textdatamy.csv')
# Y = df["HeartDisease"].to_numpy(int, copy=False)
# Y=Y.reshape(-1, 1)
# X = df.iloc[:, 1:-1].values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
# from keras.models import Sequential
# from keras.layers import Dense
# model=Sequential([
#     Dense(16,activation='relu',input_shape=(9,)),
#     Dense(16,activation='relu'),
#     Dense(16,activation='relu'),
#     Dense(1,activation='sigmoid'),
# ])
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# hist = model.fit(X_train, Y_train,
#           batch_size=32, epochs=100)
# plt.plot(hist.history['loss'],color="red")
# #plt.title('Model accuracy')
# plt.ylabel('model loss')
# plt.xlabel('Epoch')
# plt.show()