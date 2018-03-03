import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['grid.linestyle'] = ':'

def plot_history(history):
    
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()