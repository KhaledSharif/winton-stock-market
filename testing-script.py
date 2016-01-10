import xgboost as xgb
import numpy as np
import pandas

# ------------------------------------------
percentage = 5
data = pandas.read_csv('./train.csv', index_col=0, nrows=int((percentage / 100) * 40000))

print("Working with ", len(data.index), " tuples.")

np.random.seed(4815)
data['Train_Or_Test'] = np.random.rand(len(data.index), 1) >= 0.9
train_data = data[data['Train_Or_Test'] == False]
test_data = data[data['Train_Or_Test'] == True]

train_targets = train_data['Ret_PlusOne'].values
test_targets = test_data['Ret_PlusOne'].values

train_weights = train_data['Weight_Daily'].values
test_weights = test_data['Weight_Daily'].values

data = data.drop(data.columns[range(146, 210)], axis=1)

train_data = train_data.drop('Train_Or_Test', axis=1).values
test_data = test_data.drop('Train_Or_Test', axis=1).values
# ------------------------------------------

data_train = xgb.DMatrix(train_data, label=train_targets, missing=np.NaN, weight=train_weights)
data_test = xgb.DMatrix(test_data, label=test_targets, missing=np.NaN, weight=test_weights)


model_parameters = {'max_depth': 10,
                    'eta': 0.1,
                    'silent': 1,
                    'gamma': 0,
                    'lambda': 500, 'alpha': 400
                    }

number_of_rounds = 10000

watchlist = [(data_test, 'eval'),
             (data_train, 'train')]

bst = xgb.train(model_parameters,
                data_train,
                number_of_rounds,
                watchlist, early_stopping_rounds=10)

preds = bst.predict(data_test)
labels = data_test.get_label()

X = np.mean(test_weights * np.abs(preds - labels))
print(round(X))
