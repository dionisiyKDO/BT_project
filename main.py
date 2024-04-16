from neural_network import *
from data_analyze import *

if __name__ == '__main__':
    # VGG16     VGG19   AlexNet     ResNet50
    # InceptionResNetV2 InceptionV3   
    # EfficientNetV2    AlexNet_own
    # test      OwnV1   OwnV2

    # Train different architectures + Results of training
    # region
    network_name = 'OwnV2'
    epochs = 60
    batch_size = 32

    earlystop = False
    model_summary = True
    logging = True

    classifier = MRIImageClassifier(network_name)
    classifier.train(epochs, batch_size, logging=logging, model_summary=model_summary, earlystop=earlystop)

    # endregion



    # Load the model from a checkpoint + Classify an 1 image
    # region
    # classifier = MRIImageClassifier()
    # checkpoint_path = './checkpoints/OwnV2.epoch36-val_acc0.9922.hdf5'
    # classifier.load_model_from_checkpoint(checkpoint_path=checkpoint_path)

    # image_path = './data/Te-pi.jpg' #   Te-no.jpg   Te-gl.jpg   Te-pi.jpg   Te-me.jpg
    # # image_path = './data/Testing/notumor/Te-no_0141.jpg'

    # predicted_class = classifier.classify_image(image_path)
    # print('Predicted class:', predicted_class)
    # endregion