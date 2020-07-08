'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2


class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.xml'
        self.model_structure = model_name + '.bin'
        self.device = device
        self.net = None
        return
        raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        model = core.read_network(self.model_weights, self.model_structure)
        self.net = core.load_network(network=model, device_name='CPU', num_requests=1)
        return self.net
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        input_name = next(iter(self.net.inputs))
        input_shape = self.net.inputs[input_name].shape
        output_name = next(iter(self.net.outputs))
        output_shape = self.net.outputs[output_name].shape
        return input_name, input_shape, output_name, output_shape
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        input_name, input_shape, output_name, output_shape = self.check_model()
        image = cv2.resize(image, (input_shape[3], input_shape[2]), interpolation=cv2.INTER_AREA)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
