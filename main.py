import neural_network
from data_analyze import *
path = './data'


if __name__ == '__main__':
    df = get_df(path = path)

    #                   df
    #                                           image_path           class
    # 0         ./data/Astrocitoma T1/005_big_gallery.jpeg  Astrocitoma T1
    # 1         ./data/Astrocitoma T1/006_big_gallery.jpeg  Astrocitoma T1
    # 2  ./data/Astrocitoma T1/01809e58d2c1e7fff56cc5d8...  Astrocitoma T1
    # 3  ./data/Astrocitoma T1/02df132a56dfb89ece42ee8d...  Astrocitoma T1
    # 4  ./data/Astrocitoma T1/044d8d9984902ca03e652a6f...  Astrocitoma T1

    # CNN   VGG16   VGG19   AlexNet 
    # InceptionV3   EfficientNetV2
    # ResNet50  InceptionResNetV2
    network_name = 'InceptionResNetV2'
    epochs = 25
    batch_size = 16

    earlystop = False
    model_summary = True
    save = True

    logging = True
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

