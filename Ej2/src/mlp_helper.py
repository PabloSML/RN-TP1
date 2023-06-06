import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  f1_score

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from datetime import datetime
from dataclasses import dataclass

# ------------------------------------------------------------------------------


def mnist_eda_plots(x_data, y_data, morpho_data):
    """ Plot some EDA plots for the MNIST dataset.
        @param x_data Images to plot
        @param y_data Labels to plot
    """   
    # Find indexes of each class instance in y_data
    class_examples = [np.where(y_data == i) for i in range(10)]
    morpho_data["digit"] = y_data
    morpho_data['slant_deg'] = np.rad2deg(np.arctan(-morpho_data['slant']))

    digit_attributes = []
    for i in range(10):
        digit_attributes.append(morpho_data[y_data == i])

    # Print number of instances of each class
    for i in range(10):
        print(f"Class {i}: {len(class_examples[i][0])} instances")

    # Plot 3 random examples of each class
    plt.figure(figsize=(20, 6))
    for i in range(10):
        for j in range(3):
            plt.subplot(3, 10, i+1 + j*10)
            plt.imshow(x_data[class_examples[i][0][np.random.randint(len(class_examples[i][0]))], ...])
            plt.axis('off')
        plt.title(f'Class {i}', pad=245)
    plt.show()

    print("Promedio de brillo de cada clase")
    # Plot brightness mean of each class
    plt.figure(figsize=(20, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(np.mean(x_data[class_examples[i][0], ...], axis=0))
        plt.axis('off')
        plt.title(f'Class {i}')    
    plt.show()

    print("Histogramas de brillo de cada clase")
    # Plot brightness distribution of each class
    plt.figure(figsize=(20, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.hist(np.mean(x_data[class_examples[i][0], ...], axis=(1,2)), color=plt.cm.Accent(i%8), bins=50)
        plt.grid()
        plt.xlim(0, 100)
        plt.ylim(0, 650)
        plt.minorticks_on()
        plt.tight_layout()
        plt.title(f'Class {i}') 
    plt.show()

    print("Histogramas de inclinación de cada clase")
    # Plot slant distribution of each class
    fig, ax = plt.subplots(2, 5, figsize=(20, 6))
    for i in range(10):
        ax[i//5, i%5].hist(morpho_data['slant_deg'][class_examples[i][0]], color=plt.cm.Accent(i%8), bins=50)
        ax[i//5, i%5].grid()
        ax[i//5, i%5].set_xlim(-45, 45)
        ax[i//5, i%5].set_ylim(0, 300)
        ax[i//5, i%5].minorticks_on()
        ax[i//5, i%5].set_title(f'Class {i}')
        ax[i//5, i%5].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}$\degree$"))
    plt.tight_layout()
    plt.show()

    print("Scatter entre slant y las otras variables morfológicas de cada clase")
    digit_attributes_labels = ['area', 'length', 'thickness', 'width', 'height']
    digit_attributes = []
    for i in range(10):
        digit_attributes.append(morpho_data[y_data == i])

    plt.figure(figsize=(20,16))
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.tight_layout()
        for k in range(len(digit_attributes_labels)):
            a = plt.scatter(x = digit_attributes[i]['slant'].values / np.max(digit_attributes[i]['slant'].values),
                            y = digit_attributes[i][digit_attributes_labels[k]].values / np.max(digit_attributes[i][digit_attributes_labels[k]].values),
                            label = digit_attributes_labels[k],
                            alpha = 0.1)
        plt.title("Class " + str(i))
        plt.grid(True)
        plt.legend()
    plt.show()


def create_fmnist_model(
        metrics=['mae'],
        loss='mae',
        hidden_layers=3,
        hidden_units=[512, 64, 64],
        weight_initializer='glorot_normal',
        weight_initializer_stddev=0.001,
        optimizer='Nadam',
        activation='sigmoid',
        learning_rate=0.0001,
        batch_normalization=True,
        dropout_rate=0.0,
        verbose=False
):
    """ Create a Softmax model for the Fashion MNIST dataset and return it
        @param metrics Metrics to use for training and validation. Default is mae
        @param loss Loss function to use for training. Default is mae
        @param hidden_layers Number of hidden layers. Default is 3
        @param hidden_units List of number of hidden units to use in each layer. Default is [512, 64, 64]
        @param weight_initializer Weight initializer. Default is GlorotNormal
        @param weight_initializer_stddev Standard deviation to use for the weight initializer if Random Normal. Default is None
        @param optimizer Optimizer to use for training. Default is Nadam
        @param activation Activation function to use for training. Default is sigmoid
        @param learning_rate Learning rate. Default is 0.0001
        @param batch_normalization Whether to use batch normalization or not. Default is True
        @param dropout_rate Dropout rate. Default is 0.0
        @param verbose Verbosity of the model. Default is False
    """

    if weight_initializer == 'random_normal':
        weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=weight_initializer_stddev)

    # Softmax model
    model = Sequential()

    model.add(Flatten(input_shape=(28,28)))

    for i in range(hidden_layers):
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(Dense(hidden_units[i], activation=activation, kernel_initializer=weight_initializer))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate/(i+1)))

    if batch_normalization:
        model.add(BatchNormalization())
    
    model.add(Dense(1, activation="linear", kernel_initializer=weight_initializer))

    if verbose:
        model.summary()
    metrics = metrics
    if optimizer == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == 'Adam':
        optimizer = Adam(learning_rate=learning_rate, beta_1=0.95, beta_2=0.999)
    elif optimizer == 'Nadam':
        optimizer = Nadam(learning_rate=learning_rate, beta_1=0.94, beta_2=0.999)
    elif optimizer == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)

    return model

def run_model(
        model,
        x_train,
        x_valid,
        y_train,
        y_valid,
        batch_size=32,
        max_epochs=100,
        batch_normalization=True,
        reduce_lr_config = {
            'monitor': 'val_mae',
            'factor': 0.2,
            'patience': 5,
            'min_lr': 0.00005
        },
        es_config = {
            'monitor': 'val_mae',
            'patience': 10,
            'mode': 'min',
            'restore_best_weights': True,
            'min_delta': 0.0
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

    if not batch_normalization:
        # Normalize the data
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data.reshape(-1, 784)).reshape(-1, 28, 28)
        x_valid = scaler.transform(x_valid.reshape(-1, 784)).reshape(-1, 28, 28)

    if verbose:
        print(f"x_train Shape: {x_train.shape}")
        print(f"x_test Shape: {x_valid.shape}")
        print(f"y_train Shape: {y_train.shape}")
        print(f"y_test Shape: {y_valid.shape}")

    date_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Configuring TensorBoard to log learning process
    log_dir = "../logs/mlp/fit/" + date_id

    # Callback to reduce LR if we find a plateau in validation accuracy
    reduceLr = tf.keras.callbacks.ReduceLROnPlateau(monitor=reduce_lr_config['monitor'], 
                                                    factor=reduce_lr_config['factor'], 
                                                    patience=reduce_lr_config['patience'], 
                                                    min_lr=reduce_lr_config['min_lr'])

    earlyStopping = EarlyStopping(monitor=es_config['monitor'], 
                                  patience=es_config['patience'], 
                                  mode=es_config['mode'], 
                                  restore_best_weights=es_config['restore_best_weights'],
                                  min_delta=es_config['min_delta'], 
                                  verbose=verbose)
    
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_valid, y_valid),
                        batch_size=batch_size,
                        epochs=max_epochs,
                        callbacks=[reduceLr, earlyStopping, tensorboard_callback],
                        use_multiprocessing=True,
                        verbose=verbose)    

    best_epoch_train_metrics = {metric: history.history[metric][earlyStopping.best_epoch] for metric in history.history.keys() if not metric.startswith('val_')}
    best_epoch_val_metrics = {metric.removeprefix('val_'): history.history[metric][earlyStopping.best_epoch] for metric in history.history.keys() if metric.startswith('val_')}

    best_epoch_train_metrics['iterations'] = earlyStopping.best_epoch*batch_size
    best_epoch_val_metrics['iterations'] = earlyStopping.best_epoch*batch_size

    metrics_df = pd.DataFrame({'train': best_epoch_train_metrics, 'val': best_epoch_val_metrics}).sort_index(key=lambda x: x.str.lower())

    if show_metrics:
        print(metrics_df)

    return history, metrics_df, date_id

def get_model_maes_iterations(
    x_train,
    x_valid,
    y_train,
    y_valid,
    metrics=['mae'],
    loss='mae',
    hidden_layers=3,
    hidden_units=[512, 64, 64],
    weight_initializer='glorot_normal',
    optimizer='Adam',
    activation='sigmoid',
    learning_rate=0.001,
    batch_normalization=True,
    dropout_rate=0.1,
    batch_size=256,
    max_epochs=100,
    reduce_lr_config = {
            'monitor': 'val_mae',
            'factor': 0.2,
            'patience': 5,
            'min_lr': 0.00005
    },
    es_config = {
        'monitor': 'val_mae',
        'patience': 10,
        'mode': 'min',
        'restore_best_weights': True,
        'min_delta': 0.0
    }       
):
    model = create_fmnist_model(
        metrics=metrics,
        loss=loss,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        weight_initializer=weight_initializer,
        weight_initializer_stddev=0.001,
        optimizer=optimizer,
        activation=activation,
        learning_rate=learning_rate,
        batch_normalization=batch_normalization,
        dropout_rate=dropout_rate,
        verbose=False
    )

    history, metrics_df, date_id = run_model(
        model=model,
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
        batch_size=batch_size,
        max_epochs=max_epochs,
        reduce_lr_config=reduce_lr_config,
        es_config = es_config,
        show_metrics=False,
        verbose=False
    )

    return metrics_df['train']['mae'], metrics_df['val']['mae'], metrics_df['val']['iterations']

# LEARNING_RATE, BATCH_SIZE, OPTIMIZADORES, ACTIVACIONES, DROPOUT_RATE, BATCH_NORMALIZATION, INICIALIZACIONES DE PESOS

def get_plot_data_vs_param(
        x_train,
        x_valid,
        y_train,
        y_valid,
        param_name: str,
        param_data: list,
    ):
    maes_arr = []
    val_maes_arr = []
    iterations_arr = []

    for param in param_data:
        mae, val_mae, iterations = get_model_maes_iterations(x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid, **{param_name: param})

        maes_arr.append(mae)
        val_maes_arr.append(val_mae)
        iterations_arr.append(iterations)

    return maes_arr, val_maes_arr, iterations_arr

@dataclass(frozen=True)
class PlotData:
    param_name: str
    param_data: np.ndarray
    maes_arr: np.ndarray
    val_maes_arr: np.ndarray
    iterations_arr: np.ndarray

def recover_plot_data(dir='plotting_data'):
    plot_data = {}
    plot_data['lr'] = PlotData('Learning Rate', *np.load(f'{dir}/lr.npy'))
    plot_data['bs'] = PlotData('Batch Size', *np.load(f'{dir}/bs.npy'))
    plot_data['opt'] = PlotData('Optimizer', *np.load(f'{dir}/opt.npy'))
    plot_data['act'] = PlotData('Activation', *np.load(f'{dir}/act.npy'))
    plot_data['dr'] = PlotData('Dropout Rate', *np.load(f'{dir}/dr.npy'))
    plot_data['bn'] = PlotData('Batch Normalization', *np.load(f'{dir}/bn.npy'))
    plot_data['wi'] = PlotData('Weight Initializer', *np.load(f'{dir}/wi.npy'))
    plot_data['loss'] = PlotData('Loss', *np.load(f'{dir}/loss.npy'))

    return plot_data

def plot_sweep_results(metric, dir='plotting_data'):
    # Recover Data from files
    plot_data = recover_plot_data(dir=dir)

    # Plot maes vs parameters in 2x4 grid with training mae and validation mae on each plot
    plt.figure(figsize=(20,10))

    # Learning Rate Plot
    plt.subplot(2, 4, 1)
    if metric == 'MAE':
        plt.plot(plot_data['lr'].param_data, plot_data['lr'].maes_arr, '-o', label='Training MAE', color='blue')
        plt.plot(plot_data['lr'].param_data, plot_data['lr'].val_maes_arr, '-o', label='Validation MAE', color='red')
    elif metric == 'Iterations':
        plt.plot(plot_data['lr'].param_data, plot_data['lr'].iterations_arr, '-o', label='Iterations', color='darkgreen')
    plt.xscale('log')
    plt.title(f"{metric} vs {plot_data['lr'].param_name}")
    plt.xlabel(plot_data['lr'].param_name)
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.minorticks_on()

    # Batch Size Plot
    plt.subplot(2, 4, 2)
    if metric == 'MAE':
        plt.plot(plot_data['bs'].param_data, plot_data['bs'].maes_arr, '-o', label='Training MAE', color='blue')
        plt.plot(plot_data['bs'].param_data, plot_data['bs'].val_maes_arr, '-o', label='Validation MAE', color='red')
    elif metric == 'Iterations':
        plt.plot(plot_data['bs'].param_data, plot_data['bs'].iterations_arr, '-o', label='Iterations', color='darkgreen')
    plt.xscale('log')
    plt.title(f"{metric} vs {plot_data['bs'].param_name}")
    plt.xlabel(plot_data['bs'].param_name)
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.minorticks_on()

    # Optimizer Plot as a grouped bar chart
    plt.subplot(2, 4, 3)
    bar_width = 0.35
    x = np.arange(len(plot_data['opt'].param_data))
    if metric == 'MAE':
        plt.bar(x - bar_width/2, np.array(plot_data['opt'].maes_arr, dtype=np.float32).round(4), bar_width, label='Training MAE', color='blue')
        plt.bar(x + bar_width/2, np.array(plot_data['opt'].val_maes_arr, dtype=np.float32).round(4), bar_width, label='Validation MAE', color='red')
        #plt.ylim(0.8, 0.98)
    elif metric == 'Iterations':
        plt.bar(x, np.array(plot_data['opt'].iterations_arr, dtype=np.float32).round(4), bar_width, label='Iterations', color='darkgreen')
    plt.xticks(x, plot_data['opt'].param_data)
    plt.title(f"{metric} vs {plot_data['opt'].param_name}")
    plt.xlabel(plot_data['opt'].param_name)
    plt.ylabel(f"{metric}")
    plt.minorticks_on()
    plt.grid(which='both', axis='y')
    plt.legend()

    # Activation Function Plot as a grouped bar chart
    plt.subplot(2, 4, 4)
    bar_width = 0.35
    x = np.arange(len(plot_data['act'].param_data))
    if metric == 'MAE':
        plt.bar(x - bar_width/2, np.array(plot_data['act'].maes_arr, dtype=np.float32).round(4), bar_width, label='Training MAE', color='blue')
        plt.bar(x + bar_width/2, np.array(plot_data['act'].val_maes_arr, dtype=np.float32).round(4), bar_width, label='Validation MAE', color='red')
        #plt.ylim(0.88, 0.98)
    elif metric == 'Iterations':
        plt.bar(x, np.array(plot_data['act'].iterations_arr, dtype=np.float32).round(4), bar_width, label='Iterations', color='darkgreen')
    plt.xticks(x, plot_data['act'].param_data)
    plt.title(f"{metric} vs {plot_data['act'].param_name}")
    plt.xlabel(plot_data['act'].param_name)
    plt.ylabel(f"{metric}")
    plt.minorticks_on()
    plt.grid(which='both', axis='y')
    plt.legend()

    # Dropout Rate Plot
    plt.subplot(2, 4, 5)
    if metric == 'MAE':
        plt.plot(plot_data['dr'].param_data, plot_data['dr'].maes_arr, '-o', label='Training MAE', color='blue')
        plt.plot(plot_data['dr'].param_data, plot_data['dr'].val_maes_arr, '-o', label='Validation MAE', color='red')
    elif metric == 'Iterations':
        plt.plot(plot_data['dr'].param_data, plot_data['dr'].iterations_arr, '-o', label='Iterations', color='darkgreen')
    plt.title(f"{metric} vs {plot_data['dr'].param_name}")
    plt.xlabel(plot_data['dr'].param_name)
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.minorticks_on()

    # Batch Normalization Plot as a grouped bar chart
    plt.subplot(2, 4, 6)
    bar_width = 0.35
    x = np.arange(len(plot_data['bn'].param_data))
    if metric == 'MAE':
        plt.bar(x - bar_width/2, np.array(plot_data['bn'].maes_arr, dtype=np.float32).round(4), bar_width, label='Training MAE', color='blue')
        plt.bar(x + bar_width/2, np.array(plot_data['bn'].val_maes_arr, dtype=np.float32).round(4), bar_width, label='Validation MAE', color='red')
        #plt.ylim(0.88, 0.98)
    elif metric == 'Iterations':
        plt.bar(x, np.array(plot_data['bn'].iterations_arr, dtype=np.float32).round(4), bar_width, label='Iterations', color='darkgreen')
    plt.xticks(x, ['Off', 'On'])
    plt.title(f"{metric} vs {plot_data['bn'].param_name}")
    plt.xlabel(plot_data['bn'].param_name)
    plt.ylabel(f"{metric}")
    plt.minorticks_on()
    plt.grid(which='both', axis='y')
    plt.legend()

    # Weight Initializer Plot as a grouped bar chart
    plt.subplot(2, 4, 7)
    bar_width = 0.35
    x = np.arange(len(plot_data['wi'].param_data))
    if metric == 'MAE':
        plt.bar(x - bar_width/2, np.array(plot_data['wi'].maes_arr, dtype=np.float32).round(4), bar_width, label='Training MAE', color='blue')
        plt.bar(x + bar_width/2, np.array(plot_data['wi'].val_maes_arr, dtype=np.float32).round(4), bar_width, label='Validation MAE', color='red')
        #plt.ylim(0.88, 1.0)
    elif metric == 'Iterations':
        plt.bar(x, np.array(plot_data['wi'].iterations_arr, dtype=np.float32).round(4), bar_width, label='Iterations', color='darkgreen')
    plt.xticks(x, ['glorot_n', 'glorot_u', 'rand_n', 'rand_u'])
    plt.title(f"{metric} vs {plot_data['wi'].param_name}")
    plt.xlabel(plot_data['wi'].param_name)
    plt.ylabel(f"{metric}")
    plt.minorticks_on()
    plt.grid(which='both', axis='y')
    plt.legend()

    # Loss Function Plot as a grouped bar chart
    plt.subplot(2, 4, 8)
    bar_width = 0.35
    x = np.arange(len(plot_data['loss'].param_data))
    if metric == 'MAE':
        plt.bar(x - bar_width/2, np.array(plot_data['loss'].maes_arr, dtype=np.float32).round(4), bar_width, label='Training MAE', color='blue')
        plt.bar(x + bar_width/2, np.array(plot_data['loss'].val_maes_arr, dtype=np.float32).round(4), bar_width, label='Validation MAE', color='red')
        #plt.ylim(0.88, 0.98)
    elif metric == 'Iterations':
        plt.bar(x, np.array(plot_data['loss'].iterations_arr, dtype=np.float32).round(4), bar_width, label='Iterations', color='darkgreen')
    plt.xticks(x, plot_data['loss'].param_data)
    plt.title(f"{metric} vs {plot_data['loss'].param_name}")
    plt.xlabel(plot_data['loss'].param_name)
    plt.ylabel(f"{metric}")
    plt.minorticks_on()
    plt.grid(which='both', axis='y')
    plt.legend()

    plt.show()
