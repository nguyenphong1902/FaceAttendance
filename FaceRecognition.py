import cv2
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
import numpy as np
import threading
import os
import glob


class FaceRecognition:
    def __init__(self, capture, min_face=300, accuracy_th=0.7):
        self.min_face = min_face
        self.mtcnn_pt = MTCNN(image_size=160, margin=0, min_face_size=self.min_face)  # initializing mtcnn for face detection
        self.resnet = InceptionResnetV1(
            pretrained='vggface2').eval()  # initializing resnet for face img to embeding conversion
        self.model_path = 'classify_model.pkl'
        self.accuracy_th = accuracy_th
        self.new_boxes = False
        self.lock_boxes = threading.Lock()
        self.lock_cap = threading.Lock()
        self.lock_flag = threading.Lock()
        self.cap = capture
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        with open(self.model_path, 'rb') as infile:
            (self.model, self.class_names) = pickle.load(infile)

        # cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
        self.box_draw = [[]]
        self.text_draw = [[]]
        self.mark_draw = [[]]
        self.stop_flag = [False]
        self.mask = cv2.imread('images/fm2.png')

    def set_params(self, min_face=None, accuracy_th=None):
        if min_face is not None and 0 < min_face <= 1000:
            self.min_face = min_face
            self.mtcnn_pt = MTCNN(image_size=160, margin=0, min_face_size=self.min_face)
        if accuracy_th is not None and 0 < accuracy_th < 1:
            self.accuracy_th = accuracy_th

    def load_model(self, path=''):
        if path != '' and os.path.isfile(path):
            with open(path, 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)
        else:
            with open(self.model_path, 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)

    # Draw bounding box and text on image
    def draw_frame(self, image, bounding_boxes, label_texts=[],
                   landmarks=[], face_mask_anchor=False, color=[], thick=2, text_scale=0.5, skip_list=[]):
        if bounding_boxes is None:
            return
        if not color:
            color = [(255, 255, 0)] * len(bounding_boxes)
        for i, box in enumerate(bounding_boxes):
            if i in skip_list:
                continue
            cv2.rectangle(image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color[i],
                          thick)
            if label_texts:
                cv2.putText(image, label_texts[i],
                            (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color[i], thick)
            if landmarks:
                for point in landmarks[i]:
                    cv2.circle(image, (int(point[0]), int(point[1])), 2, color, thick)
                if face_mask_anchor:
                    center_eye = (landmarks[i][0] + landmarks[i][1]) / 2
                    center_lip = (landmarks[i][3] + landmarks[i][4]) / 2
                    slope_ver = (center_eye[1] - center_lip[1]) / (center_eye[0] - center_lip[0])
                    slope_hor = -1 / slope_ver
                    chin = ((box[3] - center_eye[1]) / slope_ver + center_eye[0], box[3])
                    center = (center_eye + landmarks[i][2]) / 2
                    left_ear = (box[2], slope_hor * (box[2] - center[0]) + center[1])
                    right_ear = (box[0], slope_hor * (box[0] - center[0]) + center[1])
                    cv2.circle(image, (int(chin[0]), int(chin[1])), 2, (255, 255, 255), 2)
                    cv2.circle(image, (int(center[0]), int(center[1])), 2, (255, 255, 255), 2)
                    cv2.circle(image, (int(left_ear[0]), int(left_ear[1])), 2, (255, 255, 255), 2)
                    cv2.circle(image, (int(right_ear[0]), int(right_ear[1])), 2, (255, 255, 255), 2)

    # Detect face on image and match with classify model, update result to bounding boxes and texts
    def face_match(self, image, classify_model, person_names):
        box_dr = []
        text_dr = []
        mark_dr = []
        try:
            bboxes, prob, landmarks = self.mtcnn_pt.detect(image, landmarks=True)
        except Exception as ex:
            with self.lock_boxes:
                self.box_draw[0] = box_dr
                self.text_draw[0] = text_dr
            return box_dr, text_dr, mark_dr
        if bboxes is None:
            with self.lock_boxes:
                self.box_draw[0] = box_dr
                self.text_draw[0] = text_dr
            return box_dr, text_dr, mark_dr
        for idx, box in enumerate(bboxes):
            if prob[idx] > 0.90:  # if face detected and probability > 90%
                box_dr.append(box)
                mark_dr.append(landmarks[idx])
                face = extract_face(image, box, image_size=self.mtcnn_pt.image_size, margin=self.mtcnn_pt.margin)
                face = fixed_image_standardization(face)
                emb = self.resnet(face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
                emb_array = emb.detach().numpy()
                predictions = classify_model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                if best_class_probabilities[0] > self.accuracy_th:
                    text = '{0}: {1:.0%}'.format(person_names[best_class_indices[0]], best_class_probabilities[0])
                else:
                    text = '{0}'.format('Unknown')
                text_dr.append(text)

            elif prob[idx] > 0.10:
                continue
            else:
                continue
        with self.lock_boxes:
            self.box_draw[0] = box_dr
            self.text_draw[0] = text_dr
            self.mark_draw[0] = mark_dr
            self.new_boxes = True
        return box_dr, text_dr, mark_dr

    # A thread to apply function face_match
    def thread_face_recog(self):
        while True:
            if self.cap is None:
                break
            with self.lock_flag:
                if self.stop_flag[0]:
                    break
            with self.lock_cap:
                ret_copy, frame_copy = self.cap.read()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            self.face_match(frame_copy, self.model, self.class_names)
        print('thread_face_recog stoped')

    # Stop above thread
    def stop_thread_face_recog(self):
        with self.lock_flag:
            self.stop_flag[0] = True

        with self.lock_boxes:
            self.box_draw[0] = []
            self.text_draw[0] = []
            self.mark_draw[0] = []

    # Sample to implement with camera
    def face_recog_cam(self):
        thread = threading.Thread(target=self.thread_face_recog, args=(), daemon=True)
        thread.start()
        while True:
            # Capture frame-by-frame
            with self.lock_cap:
                ret, frame = self.cap.read()

            with self.lock_boxes:
                boxes = self.box_draw[0]
                texts = self.text_draw[0]
                marks = self.mark_draw[0]

            self.draw_frame(frame, boxes, texts)

            # frame = add_face_mask(frame, mask)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Sample with image folder or file
    def face_recog_image(self, path):
        if not os.path.exists(path):
            return
        if os.path.isfile(path):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, texts, marks = self.face_match(image, self.model, self.class_names)
            self.draw_frame(image, bboxes, texts)
            cv2.imshow('', image)
            cv2.waitKey()
            cv2.destroyWindow('')

        if os.path.isdir(path):
            filenames = glob.glob(path + '/*.jpg')
            images = [cv2.imread(img) for img in filenames]
            for idx, img in enumerate(images):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bboxes, texts, marks = self.face_match(img, self.model, self.class_names)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.draw_frame(img, bboxes, texts)
                cv2.imshow(str(idx), img)
                cv2.waitKey()
                cv2.destroyWindow(str(idx))

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

# fr = FaceRecognition()
# fr.face_recog_cam()
# del fr
