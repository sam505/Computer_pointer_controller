from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPoseEstimation
from src.facial_landmarks_detection import FacialLandmarksDetection
from src.gaze_estimation import GazeEstimation
from src.mouse_controller import MouseController
import argparse
import time
import cv2


def main(args):
    fd = FaceDetection('models/intel/face-detection-adas-binary-0001/FP32-'
                       'INT1/face-detection-adas-binary-0001', device='CPU')
    hpe = HeadPoseEstimation('models/intel/head-pose-estimation'
                       '-adas-0001/FP16/head-pose-estimation-adas-0001', device='CPU')
    fld = FacialLandmarksDetection('models/intel/landmarks-regression-retail-0009/FP16/'
                             'landmarks-regression-retail-0009', device='CPU')
    ge = GazeEstimation('models/intel/gaze-estimation-adas-0002/FP16/'
                        'gaze-estimation-adas-0002', device='CPU')
    mc = MouseController('medium', 'fast')

    feed = InputFeeder(input_type=args.input_type, input_file=args.input_file)

    feed.load_data()

    fd.load_model()
    fld.load_model()
    hpe.load_model()
    ge.load_model()

    for batch in feed.next_batch():
        if batch.any():
            start = time.time()
            cropped = fd.preprocess_output(image=batch)
            key = cv2.waitKey(1)
            stream = cv2.waitKey(1)
            not_stream = cv2.waitKey(1)
            raw = cv2.resize(batch, (720, 480), interpolation=cv2.INTER_AREA)
            right_eye, left_eye = fld.preprocess_output(cropped)
            head_angles = hpe.preprocess_output(cropped)
            coordinates = ge.predict(right_eye, left_eye, head_angles)
            mc.move(coordinates[0][0], coordinates[0][1])
            print('The total time taken to obtain results is: {:.4f} seconds'.format(time.time()-start))
            print("")
            if key == ord('q'):
                break
        if not_stream != ord('t') and (stream == ord('w') or not_stream != ord('t')):
            cv2.imshow('Raw', raw)
        if stream != ord('w') and not_stream == ord('t'):
            cv2.destroyAllWindows()
    feed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_type', required=True, help='Enter the type of input either video, cam or image')
    parser.add_argument('--input_file', default='bin/demo.mp4', help='Enter the directory path for the input file')

    args = parser.parse_args()

    main(args)
