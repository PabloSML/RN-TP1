import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  f1_score

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from datetime import datetime

# ------------------------------------------------------------------------------

def load_fmnist_data():
    """ Return the Fashion MNIST dataset, with the following structure:
        x_data: 60000x28x28 array with the images
        y_data: 60000x1 array with the labels
        class_names: 10x1 array with the class names
    """
    x_data = np.load('../AssignmentGoodies/train_images.npy')
    print(f"x data Shape: {x_data.shape}")

    y_data = pd.read_csv('../AssignmentGoodies/train_labels.csv').to_numpy()[:,0]
    print(f"y data Shape: {y_data.shape}")

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return x_data, y_data, class_names


def fmnist_eda_plots(x_data, y_data, class_names):
    """ Plot some EDA plots for the Fashion MNIST dataset.
        @param x_data Images to plot
        @param y_data Labels to plot
        @param class_names Names of the classes
    """   
    # Find indexes of each class instance in y_data
    class_examples = [np.where(y_data == i) for i in range(10)]

    # Plot 3 random examples of each class
    plt.figure(figsize=(20, 6))
    for i in range(10):
        for j in range(3):
            plt.subplot(3, 10, i+1 + j*10)
            plt.imshow(x_data[class_examples[i][0][np.random.randint(0, 6000)], ...])
            plt.axis('off')
        plt.title(f'{class_names[i]}', pad=245)
    plt.show()

    # Plot brightness mean of each class
    plt.figure(figsize=(20, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(np.mean(x_data[class_examples[i][0], ...], axis=0))
        plt.axis('off')
        plt.title(f'{class_names[i]}')    
    plt.show()

    # Plot brightness distribution of each class
    plt.figure(figsize=(20, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.hist(np.mean(x_data[class_examples[i][0], ...], axis=(1,2)), color=plt.cm.Accent(i%8), bins=50)
        plt.grid()
        plt.xlim(0, 255)
        plt.ylim(0, 650)
        plt.minorticks_on()
        plt.tight_layout()
        plt.title(f'{class_names[i]}')
    plt.show()


def create_fmnist_model(
        x_data, 
        y_data, 
        metrics=['accuracy','AUC','Precision','Recall'],
        loss='categorical_crossentropy',
        hidden_layers=1,
        hidden_units=[128],
        weight_initializer='glorot_normal',
        weight_initializer_stddev=0.001,
        optimizer='Adam',
        activation='relu',
        learning_rate=0.0001,
        batch_normalization=False,
        dropout_rate=0.0, 
        conv=False,
        train_valid_proportion=1/3,
        random_state_seed=10, 
        verbose=False
):
    """ Create a Softmax model for the Fashion MNIST dataset and return it, along with the train and valid datasets.
        @param x_data Images to use for training and validation
        @param y_data Labels to use for training and validation
        @param metrics Metrics to use for training and validation. Default is accuracy, AUC, Precision and Recall
        @param loss Loss function to use for training. Default is categorical crossentropy
        @param hidden_layers Number of hidden layers. Default is 1
        @param hidden_units List of number of hidden units to use in each layer. Default is [128]
        @param weight_initializer Weight initializer. Default is GlorotNormal
        @param weight_initializer_stddev Standard deviation to use for the weight initializer if Random Normal. Default is None
        @param optimizer Optimizer to use for training. Default is Adam
        @param activation Activation function to use for training. Default is ReLU
        @param learning_rate Learning rate. Default is 0.0001
        @param batch_normalization Whether to use batch normalization or not. Default is False
        @param dropout_rate Dropout rate. Default is 0.0
        @param conv Whether to use convolutional layers or not. Default is False
        @param train_valid_proportion Proportion of the dataset to use for training and validation. Default is 1/3
        @param random_state_seed Seed to use for random operations. Default is 10
        @param verbose Verbosity of the model. Default is False
    """

    if not batch_normalization:
        # Normalize the data
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data.reshape(-1, 784)).reshape(-1, 28, 28)

    # Split the train_valid sub-dataset into train and valid
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=train_valid_proportion, random_state=random_state_seed, shuffle=True)

    if verbose:
        print(f"x_train Shape: {x_train.shape}")
        print(f"x_valid Shape: {x_valid.shape}")
        print(f"y_train Shape: {y_train.shape}")
        print(f"y_valid Shape: {y_valid.shape}")
    
    if weight_initializer == 'random_normal':
        weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=weight_initializer_stddev)

    # Softmax model
    model = Sequential()

    if conv:
        model.add(Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=(28,28,1)))

    model.add(Flatten(input_shape=(28,28)))

    for i in range(hidden_layers):
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(Dense(hidden_units[i], activation=activation, kernel_initializer=weight_initializer))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))

    if batch_normalization:
        model.add(BatchNormalization())
    model.add(Dense(10, activation="softmax", kernel_initializer=weight_initializer))

    if verbose:
        model.summary()
    metrics = metrics
    if optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)

    return model, x_train, x_valid, y_train, y_valid

def run_model(
        model,
        x_train,
        x_valid,
        y_train,
        y_valid,
        batch_size=32,
        max_epochs=100,
        es_config = {
            'monitor': 'val_accuracy',
            'patience': 10,
            'mode': 'max',
            'restore_best_weights': True
        },
        show_metrics=False,
        verbose=False
):
    """ Run the given model, using the given datasets and parameters, metrics dataframe and id.
        @param model Model to run
        @param x_train Training dataset
        @param x_valid Validation dataset
        @param y_train Training labels
        @param y_valid Validation labels
        @param batch_size Batch size to use for training and validation. Default is 32
        @param max_epochs Maximum number of epochs to use for training and validation. Default is 100
        @param es_config Early stopping configuration. Default is monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True
        @param show_metrics Show metrics of the model. Default is False
        @param verbose Verbosity of the model. Default is False
    """
    train_size = x_train.shape[0]
    valid_size = x_valid.shape[0]

    # Make sparce label vectors
    y_sparse_train = np.zeros([train_size,10])
    y_sparse_valid = np.zeros([valid_size,10])
    for idx in range(train_size):
        y_sparse_train[idx,y_train[idx]] = 1
    for idx in range(valid_size):
        y_sparse_valid[idx,y_valid[idx]] = 1    

    date_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Configuring TensorBoard to log learning process
    log_dir = "logs/mlp/fit/" + date_id

    earlyStopping = EarlyStopping(monitor=es_config['monitor'], 
                                  patience=es_config['patience'], 
                                  mode=es_config['mode'], 
                                  restore_best_weights=es_config['restore_best_weights'], 
                                  verbose=verbose)
    
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x=x_train,
                        y=y_sparse_train,
                        validation_data=(x_valid, y_sparse_valid),
                        batch_size=batch_size,
                        epochs=max_epochs,
                        callbacks=[earlyStopping, tensorboard_callback],
                        use_multiprocessing=True,
                        verbose=verbose)    

    best_epoch_train_metrics = {metric: history.history[metric][earlyStopping.best_epoch] for metric in history.history.keys() if not metric.startswith('val_')}
    best_epoch_val_metrics = {metric.removeprefix('val_'): history.history[metric][earlyStopping.best_epoch] for metric in history.history.keys() if metric.startswith('val_')}

    f1_train = f1_score(y_train, clasify_maxprob(model.predict(x_train, verbose=False, use_multiprocessing=True)), average='macro')
    f1_val = f1_score(y_valid, clasify_maxprob(model.predict(x_valid, verbose=False, use_multiprocessing=True)), average='macro')

    best_epoch_train_metrics['f1'] = f1_train
    best_epoch_val_metrics['f1'] = f1_val

    best_epoch_train_metrics['iterations'] = earlyStopping.best_epoch*batch_size
    best_epoch_val_metrics['iterations'] = earlyStopping.best_epoch*batch_size

    metrics_df = pd.DataFrame({'train': best_epoch_train_metrics, 'val': best_epoch_val_metrics}).sort_index(key=lambda x: x.str.lower())

    if show_metrics:
        print(metrics_df)

    return history, metrics_df, date_id


def get_model_accuracies_iterations(
    x_data,
    y_data,
    metrics=['accuracy'],
    loss='categorical_crossentropy',
    hidden_layers=1,
    hidden_units=[512],
    weight_initializer='glorot_normal',
    optimizer='Adam',
    activation='relu',
    learning_rate=0.0001,
    batch_normalization=False,
    dropout_rate=0.3,
    train_valid_proportion=1/3,
    random_state_seed=10,
    batch_size=32,
    max_epochs=100,
    es_config = {
        'monitor': 'val_accuracy',
        'patience': 10,
        'mode': 'max',
        'restore_best_weights': True
    }        
):
    model, x_train, x_valid, y_train, y_valid = create_fmnist_model(
        x_data,
        y_data,
        metrics=metrics,
        loss=loss,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        weight_initializer=weight_initializer,
        optimizer=optimizer,
        activation=activation,
        learning_rate=learning_rate,
        batch_normalization=batch_normalization,
        dropout_rate=dropout_rate,
        train_valid_proportion=train_valid_proportion,
        random_state_seed=random_state_seed,
        verbose=False
    )

    history, metrics_df, id = run_model(
        model=model,
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
        batch_size=batch_size,
        max_epochs=max_epochs,
        es_config = es_config,
        show_metrics=False,
        verbose=False
    )

    return metrics_df['train']['accuracy'], metrics_df['val']['accuracy'], metrics_df['val']['iterations']

# LEARNING_RATE, BATCH_SIZE, OPTIMIZADORES, ACTIVACIONES, DROPOUT_RATE, BATCH_NORMALIZATION, INICIALIZACIONES DE PESOS

def get_plot_data_vs_param(
        x_data,
        y_data,
        param_name: str,
        param_data: list,
    ):
    accuracies_arr = []
    val_accuracies_arr = []
    iterations_arr = []

    for param in param_data:
        accuracy, val_accuracy, iterations = get_model_accuracies_iterations(x_data=x_data, y_data=y_data, **{param_name: param})

        accuracies_arr.append(accuracy)
        val_accuracies_arr.append(val_accuracy)
        iterations_arr.append(iterations)

    return accuracies_arr, val_accuracies_arr, iterations_arr

# def get_plot_data_vs_learning_rate(
#     x_data,
#     y_data,
#     learning_rates_arr,
# ):
#     accuracies_arr = []
#     val_accuracies_arr = []
#     iterations_arr = []

#     for learning_rate in learning_rates_arr:
#         accuracy, val_accuracy, iterations = get_model_accuracies_iterations(x_data=x_data, y_data=y_data, learning_rate=learning_rate)

#         accuracies_arr.append(accuracy)
#         val_accuracies_arr.append(val_accuracy)
#         iterations_arr.append(iterations)

#     return accuracies_arr, val_accuracies_arr, iterations_arr


def tensorboard_log(log_dir, tag, data):
    """ Log a scalar, a set of data or a time series in TensorBoard, by creating the proper log file
        in the logging directory, using the given tag and data.
        @param log_dir Logging directory where the TensorBoard file is created
        @param tag Tag used to group type of data or plots
        @param data Data to plot
    """
    # Create a file writer for TensorBoard logs
    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()

    # Send to TensorBoard both results
    for i in range(len(data)):
        tf.summary.scalar(tag, data=data[i], step=i)
        file_writer.flush()

def clasify_maxprob(vector):
    """ Return a vector with predicted labels, based on the maximum probability of each element.
    @param vector Vector with probabilities
    """
    rounded_vector = []
    for prediction in vector:
        rounded_vector.append(np.argmax(prediction))

    return np.array(rounded_vector)
