from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPoseEstimation
from src.facial_landmarks_detection import FacialLandmarksDetection
from src.gaze_estimation import GazeEstimation
from src.mouse_controller import MouseController


def main():
    fd = FaceDetection('models/intel/face-detection-adas-binary-0001/FP32-'
                       'INT1/face-detection-adas-binary-0001', device='CPU')
    hpe = HeadPoseEstimation('models/intel/head-pose-estimation'
                       '-adas-0001/FP32/head-pose-estimation-adas-0001', device='CPU')
    fld = FacialLandmarksDetection('models/intel/landmarks-regression-retail-0009/FP32/'
                             'landmarks-regression-retail-0009', device='CPU')
    ge = GazeEstimation('models/intel/gaze-estimation-adas-0002/FP32/'
                        'gaze-estimation-adas-0002', device='CPU')
    mc = MouseController('medium', 'fast')

    feed = InputFeeder(input_type='video', input_file='bin/demo.mp4')

    feed.load_data()

    for batch in feed.next_batch():
        if batch.any():
            cropped = fd.preprocess_output(image=batch)
            right_eye, left_eye = fld.preprocess_output(cropped)
            head_angles = hpe.preprocess_output(cropped)
            coordinates = ge.predict(right_eye, left_eye, head_angles)
            mc.move(coordinates[0][0], coordinates[0][1])

    feed.close()


if __name__ == '__main__':
    main()
