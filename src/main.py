from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPoseEstimation
from src.facial_landmarks_detection import FacialLandmarksDetection
from src.gaze_estimation import GazeEstimation


def main():
    fd = FaceDetection('models/intel/face-detection-adas-binary-0001/FP32-'
                       'INT1/face-detection-adas-binary-0001', device='CPU')
    hpe = HeadPoseEstimation('models/intel/head-pose-estimation'
                       '-adas-0001/FP32/head-pose-estimation-adas-0001', device='CPU')
    fld = FacialLandmarksDetection('models/intel/landmarks-regression-retail-0009/FP32/'
                             'landmarks-regression-retail-0009', device='CPU')
    ge = GazeEstimation('models/intel/landmarks-regression-retail-0009/FP32/'
                   'landmarks-regression-retail-0009', device='CPU')
    feed = InputFeeder(input_type='video', input_file='bin/demo.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        image = batch
        cropped = fd.preprocess_output(image=image)
        landmarks_results = fld.preprocess_output(cropped)
        pose_results = hpe.preprocess_output(cropped)


    feed.close()


if __name__ == '__main__':
    main()
