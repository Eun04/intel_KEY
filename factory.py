import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import sys
import cv2
import numpy as np
import openvino as ov

from iotdemo import FactoryController
from iotdemo.motion.motion_detector import MotionDetector
from iotdemo.color.color_detector import ColorDetector

FORCE_STOP = False

path = ['resources/conveyor.mp4']


def thread_cam1(queue):
    # MotionDetector
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')

    # Load and initialize OpenVINO
    core = ov.Core()
    model_path = 'openvino.xml'
    model = core.read_model(model_path)
    ppp = ov.preprocess.PrePostProcessor(model)

    # HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    # pylint: disable=E1101
    cap = cv2.VideoCapture(path[0])
    flag = True

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam1 live", frame info
        queue.put(('1', frame))

        # Motion detect
        detected1 = det.detect(frame)
        if detected1 is None:
            continue

        # Enqueue "VIDEO:Cam1 detected", detected info.
        queue.put(('VIDEO: Cam1 Detected', detected1))
        input_tensor = np.expand_dims(detected1, 0)

        if flag is True:
            ppp.input().tensor() \
                .set_shape(input_tensor.shape) \
                .set_element_type(ov.Type.u8) \
                .set_layout(ov.Layout('NHWC'))
            ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
            ppp.input().model().set_layout(ov.Layout('NCHW'))
            ppp.output().tensor().set_element_type(ov.Type.f32)
            model = ppp.build()
            device_name = 'CPU'
            compiled_model = core.compile_model(model, device_name)
            flag = False
        results = compiled_model.infer_new_request({0: input_tensor})
        predictions = next(iter(results.values()))
        probs = predictions.reshape(-1)
        x_ratio = probs[0]*100
        o_ratio = probs[1]*100

        print(f"X = {x_ratio:.2f}%, O = {o_ratio:.2f}%")
        # in queue for moving the actuator 1
        if x_ratio > 80:
            queue.put(('PUSH', 1))
    cap.release()
    queue.put(('DONE', None))
    sys.exit()

def thread_cam2(queue):
    # MotionDetector
    det = MotionDetector()
    det.load_preset('motion.cfg', 'default')

    # ColorDetector
    colorDet = ColorDetector()
    colorDet.load_preset('color.cfg', 'default')

    # HW2 Open "resources/conveyor.mp4" video clip
    # pylint: disable=E1101
    cap = cv2.VideoCapture(path[0])

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # HW2 Enqueue "VIDEO:Cam2 live", frame info
        queue.put(('2', frame))

        # Detect motion
        detected2 = det.detect(frame)
        if detected2 is None:
            continue

        # Detect Color
        predict = colorDet.detect(detected2)

        # print(predict)
        name, ratio = predict[0]
        ratio = ratio * 100

        # Compute ratio
        print(f"{name}: {ratio:.2f}%")

        # Enqueue "VIDEO:Cam2 detected", detected info.
        queue.put(('VIDEO: Cam2 Detected', detected2))

        # Enqueue to handle actuator 2
        if name == 'blue':
            queue.put(('PUSH', 2))

    cap.release()
    queue.put(('DONE', None))
    sys.exit()


def imshow(title, frame, pos=None):
    # pylint: disable=E1101
    cv2.namedWindow(title)
    if pos:
        # pylint: disable=E1101
        cv2.moveWindow(title, pos[0], pos[1])
    # pylint: disable=E1101
    cv2.imshow(title, frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
                            description="Factory tool")

    parser.add_argument("-d",
                        "--device",
                        default=None,
                        type=str,
                        help="Arduino port")
    args = parser.parse_args()

    # HW2 Create a Queue
    queue = Queue()

    # HW2 Create thread_cam1 and thread_cam2 threads and start them.
    th1 = threading.Thread(target=thread_cam1, args=(queue,))
    th2 = threading.Thread(target=thread_cam2, args=(queue,))
    th1.start()
    th2.start()


    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            # pylint: disable=E1101
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            try:
            # HW2 get an item from the queue. You might need to properly handle exceptions.

            # de-queue name and data
                name, frame = queue.get(timeout=1)

                if name == 'PUSH':
                    # HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
                    ctrl.push_actuator(frame)
                elif name:
                    imshow(name, frame)
                else:
                    queue.task_done()

                if name == 'DONE':
                    FORCE_STOP = True
            except Empty:
                pass
            except Exception as error:
                print(f"Error: {error}")

#Control actuator, name == 'PUSH's

    th1.join()
    th2.join()
    # pylint: disable=E1101
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()
