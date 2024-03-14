import neural_network
from data_analyze import *
path = './data'
# path = './.archive/Training'


#         df
#                      image_path   class
# 0  ./data/glioma/Tr-gl_0010.jpg  glioma
# 1  ./data/glioma/Tr-gl_0011.jpg  glioma
# 2  ./data/glioma/Tr-gl_0012.jpg  glioma
# 3  ./data/glioma/Tr-gl_0013.jpg  glioma
# 4  ./data/glioma/Tr-gl_0014.jpg  glioma
if __name__ == '__main__':
    # df = get_df(path = path)
    # print(df.head())
    
    # CNN   VGG16   VGG19   AlexNet ResNet50
    # InceptionResNetV2 InceptionV3   
    # EfficientNetV2    EfficientNetB5
    # AlexNet_own
    network_name = 'AlexNet'
    epochs = 50
    batch_size = 32

    earlystop = False
    model_summary = True
    save = False

    logging = True
    graphs = True
    
    nn_model, nn_result = neural_network.NN(
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

