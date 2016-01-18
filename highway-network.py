import xgboost as xgb
import numpy as np
import pandas as pd
from lasagne.init import Orthogonal, Constant
from lasagne.layers import DenseLayer, MergeLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax, rectify, sigmoid
from lasagne.objectives import categorical_crossentropy, binary_crossentropy, squared_error
from lasagne.updates import nesterov_momentum, adadelta
from matplotlib import pyplot
from nolearn.lasagne import NeuralNet, TrainSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class MultiplicativeGatingLayer(MergeLayer):
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1 - inputs[0]) * inputs[2]


def highway_dense(incoming, Wh=Orthogonal(), bh=Constant(0.0),
                  Wt=Orthogonal(), bt=Constant(-4.0),
                  nonlinearity=rectify, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))

    l_h = DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    l_t = DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt, nonlinearity=sigmoid)

    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming)

# ------------------------------------------

percentage = 25
data = pd.read_csv('./train.csv', index_col=0, nrows=int((percentage / 100) * 40000))

print("Working with ", len(data.index), " tuples.")

np.random.seed(4815)

data['Train_Or_Test'] = np.random.rand(len(data.index), 1) >= 0.9
data = data.fillna(-1)

train_data = data[data['Train_Or_Test'] == False]
test_data = data[data['Train_Or_Test'] == True]

train_targets = train_data['Ret_PlusOne'].values
train_targets = np.array(train_targets).astype(np.float32)
test_targets = test_data['Ret_PlusOne'].values
test_targets = np.array(test_targets).astype(np.float32)

train_weights = train_data['Weight_Daily'].values
test_weights = test_data['Weight_Daily'].values

data = data.drop(data.columns[range(146, 210)], axis=1)

train_data = train_data.drop(train_data.columns[range(146, 210)], axis=1)
test_data = test_data.drop(test_data.columns[range(146, 210)], axis=1)

scaling_data = data.drop('Train_Or_Test', axis=1)
scaling_data = np.array(scaling_data.values).astype(np.float32)
scaler = StandardScaler()
scaler.fit(scaling_data)

train_data = train_data.drop('Train_Or_Test', axis=1).values
train_data = np.array(train_data).astype(np.float32)
train_data = scaler.transform(train_data)
test_data = test_data.drop('Train_Or_Test', axis=1).values
test_data = np.array(test_data).astype(np.float32)
test_data = scaler.transform(test_data)

# ------------------------------------------

num_features = train_data.shape[1]
epochs = 100

hidden_layers = 4
hidden_units = 1024
dropout_p = 0.5

val_auc = np.zeros(epochs)

# ==== Defining the neural network shape ====

l_in = InputLayer(shape=(None, num_features))
l_hidden1 = DenseLayer(l_in, num_units=hidden_units)
l_hidden2 = DropoutLayer(l_hidden1, p=dropout_p)
l_current = l_hidden2
for k in range(hidden_layers - 1):
    l_current = highway_dense(l_current)
    l_current = DropoutLayer(l_current, p=dropout_p)
l_dropout = DropoutLayer(l_current, p=dropout_p)
l_out = DenseLayer(l_dropout, num_units=1, nonlinearity=None)

# ==== Neural network definition ====

net1 = NeuralNet(layers=l_out,
                 update=adadelta, update_rho=0.95, update_learning_rate=1.0,
                 train_split=TrainSplit(eval_size=0), verbose=0, max_epochs=1, regression=True)

# ==== Print out input shape for diagnosis ====

print(train_data.shape)
print(train_targets.shape)

# ==== Train it for n iterations and validate on each iteration ====

for i in range(epochs):
    net1.fit(train_data, train_targets)
    pred = net1.predict(test_data)
    val_auc[i] = np.mean((test_targets - pred)**2)
    print(i + 1, "\t", val_auc[i], "\t", min(val_auc), "\t")


# ==== Make of the plot of the validation accuracy per iteration ====

pyplot.plot(val_auc, linewidth=2)
pyplot.grid()
pyplot.title("Minimum MSE is " + str(min(val_auc)))
pyplot.xlabel("Epoch")
pyplot.ylabel("Validation MSE")
pyplot.show()




