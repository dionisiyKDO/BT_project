import neural_network
from data_analyze import *

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

path = './archive'

if __name__ == '__main__':
    df = get_df(path = path)
    # print(df['class'].unique())


    # region
    # network_name = 'CNN'
    # epochs = 20
    # batch_size = 64
    # earlystop = False
    # model_summary = True
    # logging = True
    # save = True
    # graphs = True
    
    # nn_model, nn_result = neural_network.NN(df, 
    #                                         network_name=network_name, 
    #                                         epochs=epochs, 
    #                                         batch_size=batch_size,
    #                                         earlystop=earlystop,
    #                                         model_summary=model_summary, 
    #                                         logging=logging,
    #                                         save=save,
    #                                         graphs=graphs)
    # endregion
