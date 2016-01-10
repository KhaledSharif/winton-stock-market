import xgboost as xgb
import numpy as np
import pandas

major_list = []

training_csv = pandas.read_csv('./train.csv', index_col=0)
testing_csv = pandas.read_csv('./test_2.csv', index_col=0)

for Number in range(1,62+1): # From 1 to 62

    if Number == 61:
        name_of_column = 'Ret_PlusOne'
        name_of_weight = 'Weight_Daily'
    elif Number == 62:
        name_of_column = 'Ret_PlusTwo'
        name_of_weight = 'Weight_Daily'
    else:
        name_of_column = 'Ret_'+str(Number+120)
        name_of_weight = 'Weight_Intraday'


    train_targets = training_csv[name_of_column].values
    train_weights = training_csv[name_of_weight].values
    training_data = training_csv.drop(training_csv.columns[range(146, 210)], axis=1)
    training_data = training_data.values
    testing_data = testing_csv.values

    data_train = xgb.DMatrix(training_data, label=train_targets, missing=np.NaN, weight=train_weights)
    data_test = xgb.DMatrix(testing_data, missing=np.NaN)

    model_parameters = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'gamma': 0, 'lambda': 500, 'alpha': 400}
    number_of_rounds = 500

    watchlist = [(data_train, 'train')]
    bst = xgb.train(model_parameters, data_train, number_of_rounds, watchlist, early_stopping_rounds=10)

    predictions = bst.predict(data_test)
    for ID, P in enumerate(predictions):
        major_list.append({'Id': str(ID+1)+'_'+str(Number), 'Predicted': P})

output = pandas.DataFrame(data=major_list)
output.sort_values(by='Id', inplace=True)
print(output.head())
output.to_csv(path_or_buf="./output.csv",index=False)
