'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
from src.gaze_estimation import GazeEstimation
import cv2

ge = GazeEstimation('models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002', device='CPU')


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
        print('Loading the Facial Landmarks Detection Model...')
        self.net = core.load_network(network=model, device_name='CPU', num_requests=1)
        return self.net
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image)
        input_name, input_shape, output_name, output_shape = self.check_model()
        input_dict = {input_name: processed_image}
        self.net.start_async(request_id=0, inputs=input_dict)
        if self.net.requests[0].wait(-1) == 0:
            results = self.net.requests[0].outputs[output_name]

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
        if image.any():
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
        x0, y0 = int(w*outputs[0][0][0][0]), int(h*outputs[0][1][0][0])
        x1, y1 = int(w*outputs[0][2][0][0]), int(h*outputs[0][3][0][0])
        right_eye = image[y0-30: y0+30, x0-30:x0+30]
        left_eye = image[y1 - 30: y1 + 30, x1 - 30:x1 + 30]

        return right_eye, left_eye

        raise NotImplementedError
