from neural_network import *
from data_analyze import *

if __name__ == '__main__':
    # VGG16     VGG19   AlexNet     ResNet50
    # InceptionResNetV2 InceptionV3   
    # EfficientNetV2    AlexNet_own
    # test      OwnV1   OwnV2

    # network_name = 'OwnV2'
    # epochs = 35
    # batch_size = 32

    # earlystop = False
    # model_summary = True
    # save = False

    # logging = True
    # graphs = True
    
    # nn_model, nn_result = NN(network_name=network_name, 
    #                                         epochs=epochs, 
    #                                         batch_size=batch_size,
    #                                         earlystop=earlystop,
    #                                         model_summary=model_summary, 
    #                                         logging=logging,
    #                                         save=save,
    #                                         graphs=graphs
    #                                         )
    
    # print(f'Test loss: {nn_result[0]:.4f}')
    # print(f'Test accuracy: {nn_result[1]:.4f}')
    # print(f'Training time: {nn_result[2]:.4f}')



    # Load the model from a checkpoint
    checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
    loaded_model = load_model_from_checkpoint(checkpoint_path, 'OwnV2')

    # Classify an image
    # Te-no.jpg
    # Te-gl.jpg
    # Te-pi.jpg
    # Te-me.jpg
    image_path = './data/Te-me.jpg'  
    # image_path = './data/Testing/notumor/Te-no_0141.jpg'
    predicted_class = classify_image(image_path, loaded_model)
    print('Predicted class:', predicted_class)