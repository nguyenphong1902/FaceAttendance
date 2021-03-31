import cv2
import device


class MyVideoCapture:
    def __init__(self, video_source=0, width=2000, height=2000):
        self.cap = cv2.VideoCapture(video_source)
        self.cam_id = video_source
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            # raise ValueError('Unable to open this camera', video_source)
            print('Unable to open this camera', video_source)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.cap.isOpened():
            isTrue, frame = self.cap.read()
            if isTrue:
                return (isTrue, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (isTrue, None)
        else:
            return (False, None)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
