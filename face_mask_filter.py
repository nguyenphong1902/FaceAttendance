import cv2
from facenet_pytorch import MTCNN
import numpy as np


# Draw box and landmarks to image
def draw_frame(image, bounding_boxes, label_texts=[], landmarks=[]):
    if bounding_boxes is None:
        return
    for i, box in enumerate(bounding_boxes):
        cv2.rectangle(image,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 155, 255),
                      2)
        if label_texts:
            cv2.putText(image, label_texts[i],
                        (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 255), 2)
        if landmarks is not None:
            for point in landmarks[i]:
                cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 155, 255), 2)
            face_mask_pos = get_face_mask_pos(box, landmarks[i])
            for p in face_mask_pos:
                cv2.circle(image, (int(p[0]), int(p[1])), 2, (255, 255, 255), 2)


# Get more face landmarks as anchor for adding face mask
def get_face_mask_pos(box, lms):
    center_eye = (lms[0] + lms[1]) / 2
    center_lip = (lms[3] + lms[4]) / 2
    slope_ver = (center_eye[1] - center_lip[1]) / (center_eye[0] - center_lip[0])
    slope_hor = -1 / slope_ver
    chin = [(box[3] - center_eye[1]) / slope_ver + center_eye[0], box[3] + 1 / 20 * (box[3] - box[1])]
    center = ((center_eye + lms[2]) / 2)
    left_ear = [box[2], slope_hor * (box[2] - center[0]) + center[1]]
    right_ear = [box[0], slope_hor * (box[0] - center[0]) + center[1]]
    x_left = (slope_ver * right_ear[0] - slope_hor * chin[0] + chin[1] - right_ear[1]) / (slope_ver - slope_hor)
    y_left = slope_ver * (x_left - right_ear[0]) + right_ear[1]
    x_right = (slope_ver * left_ear[0] - slope_hor * chin[0] + chin[1] - left_ear[1]) / (slope_ver - slope_hor)
    y_right = slope_ver * (x_right - left_ear[0]) + left_ear[1]
    left_conner = [x_left, y_left]
    right_conner = [x_right, y_right]
    return [np.array(right_ear), center, np.array(left_ear), np.array(left_conner), np.array(chin), np.array(right_conner)]


# Split image in 2 half left, right
def split_image_in_half(image):
    if image is None:
        return
    crop_img_left = image[0:image.shape[0] - 1, 0:int(image.shape[1] / 2)]
    crop_img_right = image[0:image.shape[0] - 1, int(image.shape[1] / 2):image.shape[1] - 1]
    return crop_img_left, crop_img_right


# Add face mask to image
def add_face_mask(srs_image, mask_image):
    mtcnn_pt = MTCNN(image_size=160, margin=0, min_face_size=20)
    srs_image = cv2.cvtColor(srs_image, cv2.COLOR_BGR2RGB)
    bboxes = []
    landmarks = []
    try:
        bboxes, prob, landmarks = mtcnn_pt.detect(srs_image, landmarks=True)
    except:
        exit()
    srs_image = cv2.cvtColor(srs_image, cv2.COLOR_RGB2BGR)
    face_mask_pos = get_face_mask_pos(bboxes[0], landmarks[0])
    fmask_left, fmask_right = split_image_in_half(mask_image)

    # Add left half of face mask
    rows, cols, ch = fmask_left.shape
    x = int(min(face_mask_pos[0][0], face_mask_pos[1][0], face_mask_pos[3][0], face_mask_pos[4][0]))
    y = int(min(face_mask_pos[0][1], face_mask_pos[1][1], face_mask_pos[3][1], face_mask_pos[4][1]))
    xmax = int(max(face_mask_pos[0][0], face_mask_pos[1][0], face_mask_pos[3][0], face_mask_pos[4][0]))
    ymax = int(max(face_mask_pos[0][1], face_mask_pos[1][1], face_mask_pos[3][1], face_mask_pos[4][1]))

    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32(
        [face_mask_pos[0] - (x, y), face_mask_pos[1] - (x, y), face_mask_pos[3] - (x, y), face_mask_pos[4] - (x, y)])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    fmask_warp = cv2.warpPerspective(fmask_left, M, (xmax - x, ymax - y), borderValue=(255, 255, 255))
    fmask_gray = cv2.cvtColor(fmask_warp, cv2.COLOR_BGR2GRAY)
    _, fmask_mask = cv2.threshold(fmask_gray, 242, 255, cv2.THRESH_BINARY)
    fmask_warp[fmask_mask == 255] = 0
    srs_image_crop = srs_image[y:ymax, x:xmax]
    fmask_mask = fmask_mask[0:srs_image_crop.shape[0], 0:srs_image_crop.shape[1]]
    fmask_warp = fmask_warp[0:srs_image_crop.shape[0], 0:srs_image_crop.shape[1]]
    applied_mask = cv2.bitwise_and(srs_image_crop, srs_image_crop, mask=fmask_mask)
    srs_image_crop = cv2.add(applied_mask, fmask_warp)
    srs_image[y:ymax, x:xmax] = srs_image_crop

    # Add right half of face mask
    rows, cols, ch = fmask_right.shape

    x = int(min(face_mask_pos[1][0], face_mask_pos[2][0], face_mask_pos[4][0], face_mask_pos[5][0]))
    y = int(min(face_mask_pos[1][1], face_mask_pos[2][1], face_mask_pos[4][1], face_mask_pos[5][1]))
    xmax = int(max(face_mask_pos[1][0], face_mask_pos[2][0], face_mask_pos[4][0], face_mask_pos[5][0]))
    ymax = int(max(face_mask_pos[1][1], face_mask_pos[2][1], face_mask_pos[4][1], face_mask_pos[5][1]))

    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32([face_mask_pos[1]-(x, y), face_mask_pos[2]-(x, y), face_mask_pos[4]-(x, y), face_mask_pos[5]-(x, y)])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    fmask_warp = cv2.warpPerspective(fmask_right, M, (xmax - x, ymax - y), borderValue=(255, 255, 255))
    fmask_gray = cv2.cvtColor(fmask_warp, cv2.COLOR_BGR2GRAY)
    _, fmask_mask = cv2.threshold(fmask_gray, 242, 255, cv2.THRESH_BINARY)
    fmask_warp[fmask_mask == 255] = 0
    srs_image_crop = srs_image[y:ymax, x:xmax]
    fmask_mask = fmask_mask[0:srs_image_crop.shape[0], 0:srs_image_crop.shape[1]]
    fmask_warp = fmask_warp[0:srs_image_crop.shape[0], 0:srs_image_crop.shape[1]]
    applied_mask = cv2.bitwise_and(srs_image_crop, srs_image_crop, mask=fmask_mask)
    srs_image_crop = cv2.add(applied_mask, fmask_warp)
    srs_image[y:ymax, x:xmax] = srs_image_crop
    return srs_image
