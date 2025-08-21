
# Neural Network
from keras import layers, optimizers, metrics, callbacks, losses, initializers, Sequential, regularizers


def build_nn_model(input_shape,
                   hidden_layer_1,
                   hidden_layer_2,
                   # hidden_layer_3,
                   dropout_1,
                   dropout_2,
                   # dropout_3,
                   l2,
                   learning_rate,
                   log_bias,
                   activation,
                   momentum,
                   loss_func,
                   seed):
    model = Sequential()
    # Input Layer
    model.add(layers.Input(shape=input_shape))
    # Hidden Layer 1
    model.add(layers.Dense(units=hidden_layer_1,
                           kernel_initializer=initializers.HeUniform(
                               seed=seed),  # type: ignore[arg-type]
                           bias_initializer='zeros',
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Dropout(dropout_1))
    # Hidden Layer 2
    model.add(layers.Dense(units=hidden_layer_2,
                           kernel_initializer=initializers.HeUniform(
                               seed=seed),  # type: ignore[arg-type]
                           bias_initializer='zeros',
                           kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.Dropout(dropout_2))

    # NOTE: Code for third layer is commented out since tuning with 2 layers was explored after 3-layer tuning
    # Ultimately, 2 layers out-performed 3 layers in average AUROC across CV folds
    ################################### Uncomment below for 3-layer architecture ###################################
    # Hidden Layer 3
    # model.add(layers.Dense(units = hidden_layer_3,
    #                                kernel_initializer= initializers.HeUniform(seed=seed),
    #                                bias_initializer='zeros'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation(activation))
    # model.add(layers.Dropout(dropout_3))
    #########################################################################################################

    # Output Layer
    model.add(layers.Dense(units=1, activation='sigmoid',
                           kernel_initializer=initializers.GlorotUniform(
                               seed=seed),  # type: ignore[arg-type]
                           bias_initializer=initializers.Constant(
                               log_bias)  # type: ignore[arg-type]
                           )
              )

    model.compile(optimizer=optimizers.SGD(learning_rate=learning_rate,
                                           momentum=momentum),  # type: ignore
                  loss=loss_func,
                  metrics=[metrics.AUC(curve='ROC', name='AUCROC')])
    return model
