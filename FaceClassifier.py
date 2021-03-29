from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import pickle
from sklearn.svm import SVC
from collections import OrderedDict
import os


class FaceClassifier:
    def __init__(self, mtcnn_init=MTCNN(image_size=160, margin=0, min_face_size=20),
                 resnet_init=InceptionResnetV1(pretrained='vggface2').eval()):
        self.mtcnn = mtcnn_init
        self.resnet = resnet_init
        self.embedding_list = []
        self.label_list = []
        self.class_list = []
        self.pretrained_path = 'data.pt'

    def load_pretrained_data(self, model_path):
        if not os.path.exists(model_path):
            return
        data = torch.load(model_path)
        self.embedding_list = data[0]
        self.label_list = data[1]
        self.class_list = list(OrderedDict.fromkeys(self.label_list))

    @staticmethod
    def collate_fn(x):
        return x[0]

    def train_SVM(self, data_folder, outtext=print, out_model='classify_model.pkl', retrain_all=False,
                  retrain_folder=None):
        if retrain_folder is None:
            retrain_folder = []
        if not os.path.exists(data_folder):
            outtext('ERROR: Data not found')
            return
        self.load_pretrained_data(self.pretrained_path)
        outtext('Start training SVM model')
        outtext('Input data: {}'.format(data_folder))
        outtext('Output: {}'.format(out_model))
        outtext('Retrain: {}'.format(str(retrain_all)))

        try:
            dataset = datasets.ImageFolder(data_folder)  # photos folder path
        except RuntimeError as rte:
            outtext('ERROR: ' + str(rte))
            return
        img_total = len(dataset.imgs)
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}  # accessing names of peoples from folder names
        loader = DataLoader(dataset, collate_fn=self.collate_fn)

        rmv_idx = []
        for idx, label in enumerate(self.label_list):
            if label not in idx_to_class.values():
                rmv_idx.append(idx)
        self.embedding_list = [i for j, i in enumerate(self.embedding_list) if j not in rmv_idx]
        self.label_list = [i for j, i in enumerate(self.label_list) if j not in rmv_idx]

        if retrain_all:  # Clear previous data if retrain all
            self.embedding_list = []
            self.label_list = []
            self.class_list = []
        elif retrain_folder:    # Remove previous data from a folder if retrain that folder
            for folder in retrain_folder:
                if not os.path.exists(os.path.join(data_folder, folder)):
                    outtext('ERROR: {} folder not exist'.format(folder))
                    return
            rmv_idx.clear()
            for idx, label in enumerate(self.label_list):
                if label in retrain_folder:
                    rmv_idx.append(idx)
            self.embedding_list = [i for j, i in enumerate(self.embedding_list) if j not in rmv_idx]
            self.label_list = [i for j, i in enumerate(self.label_list) if j not in rmv_idx]
            for folder in retrain_folder:
                if folder in self.class_list:
                    self.class_list.remove(folder)

        outtext('Create embedding from image:')
        img_count = 0
        if retrain_all:
            for img, idx in loader:
                img_count += 1
                outtext('\t{}/{}\t{}'.format(img_count, img_total, idx_to_class[idx]))
                try:
                    face, prob = self.mtcnn(img, return_prob=True)
                except Exception:
                    continue
                if face is not None and prob > 0.90:  # if face detected and porbability > 90%
                    emb = self.resnet(
                        face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
                    self.embedding_list.append(emb.detach())  # resulten embedding matrix is stored in a list
                    self.label_list.append(idx_to_class[idx])  # names are stored in a list
        else:
            for img, idx in loader:
                img_count += 1
                outtext('\t{}/{}\t{}'.format(img_count, img_total, idx_to_class[idx]))
                if idx_to_class[idx] in self.class_list:  # If not retrain all, skip image that already in previous data
                    continue
                try:
                    face, prob = self.mtcnn(img, return_prob=True)
                except Exception:
                    continue
                if face is not None and prob > 0.90:  # if face detected and porbability > 90%
                    emb = self.resnet(
                        face.unsqueeze(0))  # passing cropped face into resnet model to get embedding matrix
                    self.embedding_list.append(emb.detach())  # resulten embedding matrix is stored in a list
                    self.label_list.append(idx_to_class[idx])  # names are stored in a list

        # self.class_list = idx_to_class
        data = [self.embedding_list, self.label_list]
        outtext('Save data to: {}'.format('data.pt'))
        torch.save(data, 'data.pt')  # saving data.pt file
        cvt_embedding_list = []
        for tensor in self.embedding_list:
            cvt_embedding_list.append(tensor.numpy()[0])

        outtext('Train SVM')
        model = SVC(kernel='linear', probability=True)
        try:
            model.fit(cvt_embedding_list, self.label_list)
        except ValueError as ve:
            outtext('ERROR: ' + str(ve))
            return

        # Saving classifier model
        with open(out_model, 'wb') as outfile:
            pickle.dump((model, idx_to_class), outfile)
        outtext('Saved classifier model to file "%s"' % out_model)
