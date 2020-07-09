'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import cv2


class FacialLandmarksDetection:
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
        network = self.load_model()
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_name, output_shape = self.check_model()
        input_dict = {input_name: processed_image}
        network.start_async(request_id=0, inputs=input_dict)
        if network.requests[0].wait(-1) == 0:
            results = network.requests[0].outputs[output_name]

        return results
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

    def preprocess_output(self, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outputs = self.predict(image)
        h, w, c = image.shape
        print('landmark output_2: ', outputs)
        x0, y0 = w*outputs[0][0][0][0], h*outputs[0][1][0][0]
        x1, y1 = w*outputs[0][2][0][0], h*outputs[0][3][0][0]
        x2, y2 = w*outputs[0][4][0][0], h*outputs[0][5][0][0]
        x3, y3 = w*outputs[0][6][0][0], h*outputs[0][7][0][0]
        x4, y4 = w*outputs[0][8][0][0], h*outputs[0][9][0][0]
        print(x0, x1, x2, x3, x3, x4)
        print(y0, y1, y2, y3, y3, y4)
        image = cv2.circle(image, (int(x0), int(y0)), 10, (255, 0, 100), 2)
        image = cv2.circle(image, (int(x1), int(y1)), 10, (255, 0, 100), 2)
        image = cv2.circle(image, (int(x2), int(y2)), 10, (255, 0, 100), 2)
        image = cv2.circle(image, (int(x3), int(y3)), 10, (255, 0, 100), 2)
        image = cv2.circle(image, (int(x4), int(y4)), 10, (255, 0, 100), 2)
        cv2.imshow('Landmarks', image)
        return
        raise NotImplementedError
