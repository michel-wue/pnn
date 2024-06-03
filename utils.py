import matplotlib.pyplot as plt

def get_plot(sgd_trainer, name, avg_time, prediction_acc):
    plt.plot(sgd_trainer.parameters['loss'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.text(max(sgd_trainer.parameters['epoche']), max(sgd_trainer.parameters['loss']), f'avg_training_time: {avg_time} \n accuracy: {prediction_acc}', fontsize=12, color='black', ha='right', va='top')
    plt.title(name) 
    plt.savefig(f'./figures/{name}')
    plt.show()