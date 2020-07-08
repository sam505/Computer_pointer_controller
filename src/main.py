from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPoseEstimation
from src.facial_landmarks_detection import FacialLandmarksDetection
from src.gaze_estimation import GazeEstimation
import numpy as np


def main():
    fd = FaceDetection('models/intel/face-detection-adas-binary-0001/FP32-'
                       'INT1/face-detection-adas-binary-0001', device='CPU')
    HeadPoseEstimation('models/intel/head-pose-estimation'
                       '-adas-0001/FP32/head-pose-estimation-adas-0001', device='CPU')
    FacialLandmarksDetection('models/intel/landmarks-regression-retail-0009/FP32/'
                             'landmarks-regression-retail-0009', device='CPU')
    GazeEstimation('models/intel/landmarks-regression-retail-0009/FP32/'
                   'landmarks-regression-retail-0009', device='CPU')
    feed = InputFeeder(input_type='cam', input_file='bin/demo.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        image = batch
        print(image.shape)
        img = np.reshape(image, (1, 3, -1))
        print(img.shape)
        fd.predict(image=image)

    feed.close()


if __name__ == '__main__':
    main()
