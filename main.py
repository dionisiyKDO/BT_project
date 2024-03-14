import neural_network
from data_analyze import *

if __name__ == '__main__':
    # VGG16     VGG19   AlexNet     ResNet50
    # InceptionResNetV2 InceptionV3   
    # EfficientNetV2    EfficientNetB5
    # AlexNet_own       test    OwnV1

    network_name = 'test'
    epochs = 50
    batch_size = 32

    earlystop = False
    model_summary = True
    save = False

    logging = True
    graphs = True
    
    nn_model, nn_result = neural_network.NN(network_name=network_name, 
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

