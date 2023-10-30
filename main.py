import neural_network
from data_analyze import *


path = './archive'

if __name__ == '__main__':
    df = get_df(path = path)
    # print(df['class'].unique())

    # region
    network_name = 'VGG19'
    epochs = 25
    batch_size = 128
    earlystop = False
    model_summary = True
    logging = True
    save = True
    graphs = True
    
    nn_model, nn_result = neural_network.NN(df, 
                                            network_name=network_name, 
                                            epochs=epochs, 
                                            batch_size=batch_size,
                                            earlystop=earlystop,
                                            model_summary=model_summary, 
                                            logging=logging,
                                            save=save,
                                            graphs=graphs
                                            )
    print(f'Test loss: {nn_result[0]:.4f}')
    print(f'Test accuracy: {nn_result[1]:.4f}')
    print(f'Training time: {nn_result[2]:.4f}')
    # endregion
