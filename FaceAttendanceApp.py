import os
import time
import glob
import threading
from datetime import datetime, timedelta

import cv2
from PIL import ImageTk
from PIL import Image as ImagePIL
from tkinter import *
from tkinter import messagebox
import tkinter.ttk as ttk
from sqlalchemy import Column, Integer, String, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from face_mask_filter import add_face_mask
from FaceClassifier import FaceClassifier
from VideoCapture import MyVideoCapture
from FaceRecognition import FaceRecognition

Base = declarative_base()


class Attendance(Base):
    __tablename__ = 'Attendance'
    id = Column(Integer, primary_key=True)
    name = Column(String(250))
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)


# Link to database
engine = create_engine('sqlite:///Attendance.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


class FaceAttendanceApp:
    def __init__(self):
        # GUI
        self.appName = 'Face Attendance'
        self.root = Tk()
        self.root.title(self.appName)
        self.root.geometry("1080x720")
        self.root.minsize(1080, 720)
        self.root.resizable(True, True)
        self.tabControl = ttk.Notebook(self.root)
        self.tabFaceRecog = ttk.Frame(self.tabControl)
        self.tabAddFace = ttk.Frame(self.tabControl)
        self.tabAttendance = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tabFaceRecog, text='FaceRecog')
        self.tabControl.add(self.tabAddFace, text='AddFace')
        self.tabControl.add(self.tabAttendance, text='Attendance')
        self.tabControl.pack(expand=1, fill="both")
        self.tabControl.bind("<<NotebookTabChanged>>", self.on_tab_selected)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Params
        self.is_record = False
        self.record_count = 0
        self.record_total = 10
        self.image_buffer = []
        self.data_folder = 'images/data/train'
        self.person_name = ''
        self.modified_folder = []
        self.name_cache = []

        self.is_running_fr = False
        self.stop_update_frame_facerecog = False
        self.stop_update_frame_addface = False
        self.name_last_frame = []
        self.last_boxes = []
        self.last_texts = []
        self.last_colors = []
        self.skip_list = []

        # Video Capture
        self.vid = MyVideoCapture()
        # Init face classifier
        self.fc = FaceClassifier()
        self.fc.load_pretrained_data('data.pt')
        # Init face recognition engine
        self.fr = FaceRecognition(self.vid.cap)
        for key, value in self.fr.class_names.items():
            self.name_cache.append(value)
        self.name_counter = [0] * len(self.fr.class_names)
        self.name_time_pause = [datetime.min] * len(self.fr.class_names)
        self.green_box_counter = [0] * len(self.fr.class_names)

        # Tab FaceRecog
        self.top_frame_facerecog = Frame(self.tabFaceRecog)
        self.bottom_frame_facerecog = Frame(self.tabFaceRecog)
        self.top_frame_facerecog.grid(row=0, column=0, sticky=W + E)
        self.bottom_frame_facerecog.grid(row=1, column=0, sticky=N + W + E + S)
        self.label = Label(self.top_frame_facerecog, text=self.appName, font=15, bg='blue', fg='white')
        self.label.grid(row=0, column=0, sticky=W)
        self.frame_canvas = Frame(self.bottom_frame_facerecog, bg='red')
        self.frame_canvas.grid(row=0, column=0, sticky=N + W + E + S)
        self.canvas_facerecog = Canvas(self.frame_canvas,
                                       bg='black')
        self.canvas_facerecog.pack(fill=BOTH, expand=YES)
        self.frame_notify = LabelFrame(self.bottom_frame_facerecog, text='Notify', bg='green')
        self.frame_notify.grid(row=0, column=1, sticky=N + W + E + S)
        self.scroll_notify = Scrollbar(self.frame_notify, orient=VERTICAL)
        self.scroll_notify.pack(side=RIGHT, fill=Y)
        self.txt_notify = Text(self.frame_notify, bg='black', fg='white', width=20, yscroll=self.scroll_notify.set)
        self.txt_notify.pack(fill=Y, expand=YES)
        self.scroll_notify.config(command=self.txt_notify.yview)

        self.bottom_frame_facerecog.rowconfigure(0, weight=1)
        self.bottom_frame_facerecog.columnconfigure(0, weight=1)

        # self.btnFaceRecog = Button(self.tabFaceRecog, text='Start', command=self.btn_FaceRecog_pressed)
        # self.btnFaceRecog.pack_forget()
        self.btnRefreshFR = Button(self.top_frame_facerecog, text='Refresh', command=self.btnRefreshFRPressed)
        self.btnRefreshFR.grid(row=0, column=1, sticky=W, padx=10)
        self.lblMinFace = Label(self.top_frame_facerecog, text='Minimum face', font=15)
        self.lblMinFace.grid(row=0, column=2, sticky=E)
        self.entrMinFace = Entry(self.top_frame_facerecog, width=10)
        self.entrMinFace.grid(row=0, column=3, sticky=E)
        self.entrMinFace.insert(END, str(self.fr.min_face))
        self.lblMinFace = Label(self.top_frame_facerecog, text='Accuracy threshold', font=15)
        self.lblMinFace.grid(row=0, column=4, sticky=E)
        self.entrAccuracyTH = Entry(self.top_frame_facerecog, width=10)
        self.entrAccuracyTH.grid(row=0, column=5, sticky=E)
        self.entrAccuracyTH.insert(END, str(self.fr.accuracy_th))

        self.top_frame_facerecog.rowconfigure(0, weight=1)
        self.top_frame_facerecog.columnconfigure(1, weight=1)

        self.tabFaceRecog.rowconfigure(1, weight=1)
        self.tabFaceRecog.columnconfigure(0, weight=1)

        # Tab AddFace
        self.left_frame_addface = LabelFrame(self.tabAddFace, text='Cam')
        self.right_frame_addface = LabelFrame(self.tabAddFace, text='Button', width=150)
        self.bottom_frame_addface = LabelFrame(self.tabAddFace, text='Console')
        self.left_frame_addface.grid(row=0, column=0, sticky=N + W + E + S)
        self.right_frame_addface.grid(row=0, column=1, sticky=N + W + E + S)
        self.bottom_frame_addface.grid(row=1, column=0, columnspan=2, sticky=W + E)
        self.tabAddFace.rowconfigure(0, weight=1)
        self.tabAddFace.columnconfigure(0, weight=1)
        self.canvas_addface = Canvas(self.left_frame_addface, width=300, height=300, bg='black')
        self.canvas_addface.pack(fill=BOTH, expand=YES)

        self.lblRecord = Label(self.right_frame_addface, text='Num of image')
        self.lblRecord.grid(row=0, column=0, sticky=E)
        self.entrRecord = Entry(self.right_frame_addface, width=4)
        self.entrRecord.insert(END, str(self.record_total))
        self.entrRecord.grid(row=0, column=1, sticky=N, padx=10, pady=10)
        self.btnRecord = Button(self.right_frame_addface, text='Record', command=self.btnRecordPressed)
        self.btnRecord.grid(row=1, column=0, sticky=N, padx=10, pady=10, columnspan=2)
        self.btnTrain = Button(self.right_frame_addface, text='Train', command=self.btnTrainPressed)
        self.btnTrain.grid(row=2, column=0, sticky=S, padx=10, pady=10, columnspan=2)
        self.btnReTrainAll = Button(self.right_frame_addface, text='Retrain all', command=self.btnReTrainAllPressed)
        self.btnReTrainAll.grid(row=3, column=0, sticky=S, padx=10, pady=10, columnspan=2)

        self.scroll_console = Scrollbar(self.bottom_frame_addface, orient=VERTICAL)
        self.scroll_console.pack(side=RIGHT, fill=Y)
        self.txt_console = Text(self.bottom_frame_addface, bg='black', fg='white', height=7,
                                yscroll=self.scroll_console.set)
        self.txt_console.pack(fill=BOTH, expand=YES)
        self.scroll_console.config(command=self.txt_console.yview)

        # Tab Attendance
        self.top_frame_attend = LabelFrame(self.tabAttendance)
        self.bottom_frame_attend = LabelFrame(self.tabAttendance, text='Table')
        self.top_frame_attend.grid(row=0, column=0, sticky=W + E)
        self.bottom_frame_attend.grid(row=1, column=0, sticky=N + W + E + S)
        self.btnShowAttend = Button(self.top_frame_attend, text='Show', command=self.btnShowAttendPressed)
        self.btnShowAttend.grid(row=0, column=0, sticky=W, padx=2)
        self.btnCleanAttend = Button(self.top_frame_attend, text='Clean', command=self.btnCleanAttendPressed)
        self.btnCleanAttend.grid(row=0, column=1, sticky=W)
        self.scroll_table = Scrollbar(self.bottom_frame_attend, orient=VERTICAL)
        self.scroll_table.pack(side=RIGHT, fill=Y)
        self.tblAttend = ttk.Treeview(self.bottom_frame_attend, yscroll=self.scroll_table.set)
        self.tblAttend['columns'] = ('ID', 'Name', 'Date', 'Time')
        self.tblAttend['show'] = 'headings'
        self.tblAttend.heading("#1", text="ID")
        self.tblAttend.heading("#2", text="Name")
        self.tblAttend.heading("#3", text="Date")
        self.tblAttend.heading("#4", text="Time")
        self.tblAttend.pack(fill=BOTH, expand=YES)
        self.scroll_table.config(command=self.tblAttend.yview)

        self.tabAttendance.rowconfigure(1, weight=1)
        self.tabAttendance.columnconfigure(0, weight=1)

    # Popup
    def popup_input_frame(self):
        input_name = Toplevel()
        input_name.wm_title("Input name")
        lab = Label(input_name, text="Enter name", borderwidth=2)
        lab.grid(row=0, column=0, padx=10, pady=10)
        txtBox = Text(input_name, height=1, width=50, borderwidth=2)
        txtBox.grid(row=0, column=1, padx=10, pady=10, columnspan=4)
        btnOkay = ttk.Button(input_name, text="Okay",
                             command=lambda: [self.get_person_name(txtBox.get("1.0", 'end-1c')),
                                              input_name.destroy(),
                                              self.save_image()])
        btnOkay.grid(row=1, column=3, padx=10, pady=10, sticky=E)
        btnCancel = ttk.Button(input_name, text="Cancel", command=input_name.destroy)
        btnCancel.grid(row=1, column=4, padx=10, pady=10, sticky=E)
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        wf = input_name.winfo_width()
        hf = input_name.winfo_height()
        x = (w - wf) / 2
        y = (h - hf) / 2
        input_name.geometry('+%d+%d' % (x - 25, y))
        self.root.wait_window(input_name)

    # Control
    def btnCleanAttendPressed(self):
        clean_list = []
        with engine.connect() as con:
            rs = con.execute('''SELECT DISTINCT name, date From Attendance''')
            for row in rs:
                clean_list.append(row)
        for item in clean_list:
            self.clean_up_attendant(item[0], datetime.strptime(item[1], '%Y-%m-%d'))
        self.btnShowAttendPressed()

    def btnShowAttendPressed(self):
        self.tblAttend.delete(*self.tblAttend.get_children())
        with engine.connect() as con:
            rs = con.execute('''SELECT * FROM Attendance''')
            for row in rs:
                self.tblAttend.insert("", END, values=list(row))

    def btnRefreshFRPressed(self):
        self.stop_face_recog()
        # self.stop_update_frame_facerecog = True
        self.fr.load_model()
        self.name_cache = []
        for key, value in self.fr.class_names.items():
            self.name_cache.append(value)
        minFace = accuracyTH = None
        try:
            minFace = int(self.entrMinFace.get())
        except ValueError:
            self.entrMinFace.delete(0, 'end')
        try:
            accuracyTH = float(self.entrAccuracyTH.get())
        except ValueError:
            self.entrAccuracyTH.delete(0, 'end')
        self.fr.set_params(minFace, accuracyTH)
        time.sleep(0.5)
        # self.stop_update_frame_facerecog = False
        # self.update_frame_facerecog()
        self.start_face_recog()

    def btnTrainPressed(self):
        thread = threading.Thread(target=self.thread_train, args=(), daemon=True)
        thread.start()

    def btnReTrainAllPressed(self):
        thread = threading.Thread(target=self.thread_train, args=(),
                                  kwargs={'retrain': True}, daemon=True)
        thread.start()

    def btnRecordPressed(self):
        try:
            self.record_total = int(self.entrRecord.get())
        except ValueError:
            self.entrMinFace.delete(0, 'end')
            return
        if not self.is_record:
            self.disable_buttons()
            self.image_buffer.clear()
            # self.printProgressBar(0, self.record_total, prefix='Progress:', suffix='Complete', length=50)
            # time.sleep(0.1)
            self.is_record = True

    def resize_frame(self, event):
        pass

    # Create thread to train
    def thread_train(self, retrain=False):
        self.disable_buttons()
        self.fc.train_SVM(self.data_folder, outtext=self.write, retrain_all=retrain,
                          retrain_folder=self.modified_folder)
        self.enable_buttons()
        self.modified_folder.clear()

    def disable_buttons(self):
        self.tabControl.tab(0, state='disable')
        self.tabControl.tab(2, state='disable')
        self.btnTrain['state'] = 'disable'
        self.btnRecord['state'] = 'disable'
        self.btnReTrainAll['state'] = 'disable'

    def enable_buttons(self):
        self.btnTrain['state'] = 'normal'
        self.btnRecord['state'] = 'normal'
        self.btnReTrainAll['state'] = 'normal'
        self.tabControl.tab(0, state='normal')
        self.tabControl.tab(2, state='normal')

    # Update frame face recognition
    def update_frame_facerecog(self):
        if self.stop_update_frame_facerecog:
            return
        with self.fr.lock_cap:
            isTrue, frame = self.vid.get_frame()
        if isTrue:
            with self.fr.lock_boxes:
                if self.fr.new_boxes:
                    is_new_boxes = True
                    self.last_boxes = self.fr.box_draw[0]
                    self.last_texts = self.fr.text_draw[0]
                    self.last_colors = [(255, 255, 0)] * len(self.last_boxes)
                    self.fr.new_boxes = False
                else:
                    is_new_boxes = False

            if is_new_boxes:
                list_index = []
                self.skip_list.clear()
                for idx, text in enumerate(self.last_texts):
                    name = text.split(':')[0]
                    if name != 'Unknown':
                        try:
                            index = self.name_cache.index(name)
                        except ValueError:
                            continue
                        list_index.append(index)
                        if self.name_time_pause[index] == datetime.min:
                            if self.name_counter[index] == 0:
                                self.name_counter[index] += 1
                            else:
                                if index not in self.name_last_frame:
                                    self.name_counter[index] = 0
                                self.name_counter[index] += 1
                            if self.name_counter[index] >= 6:
                                self.write(
                                    '\n' + name + '\n' + datetime.now().strftime("%Y-%m-%d, %H:%M") + '\n' + '_' * 20,
                                    out='notify')
                                self.update_attendant(name, datetime.now())
                                self.last_colors[idx] = (0, 255, 0)
                                self.name_time_pause[index] = datetime.now() + timedelta(seconds=6)
                                self.name_counter[index] = 0
                        else:
                            if self.name_time_pause[index] > datetime.now():
                                self.green_box_counter[index] += 1
                                if self.green_box_counter[index] < 3:
                                    self.last_colors[idx] = (0, 255, 0)
                                else:
                                    self.skip_list.append(idx)
                                continue
                            else:
                                self.green_box_counter[index] = 0
                                self.name_time_pause[index] = datetime.min
                self.name_last_frame = list_index

            self.fr.draw_frame(frame, self.last_boxes, self.last_texts, color=self.last_colors, thick=2, text_scale=1,
                               skip_list=self.skip_list)  # RGB
            w = self.canvas_facerecog.winfo_width()
            h = self.canvas_facerecog.winfo_height()
            # self.photo = ImageTk.PhotoImage(image=ImagePIL.fromarray(frame).resize((w, h)))
            self.photo = ImageTk.PhotoImage(image=self.resize_with_pad(ImagePIL.fromarray(frame), (w, h)))
            self.canvas_facerecog.create_image(0, 0, image=self.photo, anchor='nw')
        self.root.after(20, self.update_frame_facerecog)

    # Update frame at Add face
    def update_frame_addface(self):
        if self.stop_update_frame_addface:
            return
        isTrue, frame = self.vid.get_frame()
        if isTrue:
            if self.is_record:
                self.image_buffer.append(frame.copy())
                self.record_count += 1
                text = 'record {}/{}'.format(self.record_count, self.record_total)
                self.write(text)
                cv2.putText(frame, text,
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                time.sleep(0.5)
            w = self.canvas_addface.winfo_width()
            h = self.canvas_addface.winfo_height()
            # self.photo = ImageTk.PhotoImage(image=ImagePIL.fromarray(frame).resize((w, h)))
            self.photo = ImageTk.PhotoImage(image=self.resize_with_pad(ImagePIL.fromarray(frame), (w, h)))
            self.canvas_addface.create_image(0, 0, image=self.photo, anchor='nw')
            if self.record_count >= self.record_total:
                self.is_record = False
                self.record_count = 0
                self.popup_input_frame()
                self.enable_buttons()
        self.root.after(20, self.update_frame_addface)

    # Event when changing tab
    def on_tab_selected(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == 'FaceRecog':
            self.stop_update_frame_addface = True
            self.stop_update_frame_facerecog = False
            self.update_frame_facerecog()
            self.start_face_recog()
        if tab_text == 'AddFace':
            self.stop_face_recog()
            self.stop_update_frame_facerecog = True
            self.stop_update_frame_addface = False
            self.update_frame_addface()
        if tab_text == 'Attendance':
            self.stop_face_recog()
            self.stop_update_frame_facerecog = True
            self.stop_update_frame_addface = True

    # def btn_FaceRecog_pressed(self):
    #     if self.is_running_fr:
    #         self.stop_face_recog()
    #         self.btnFaceRecog.configure(text='Start')
    #     else:
    #         self.start_face_recog()
    #         self.btnFaceRecog.configure(text='Stop')

    # Print to console text box
    def write(self, *message, end="\n", sep=" ", out='console'):
        text = ""
        for item in message:
            text += "{}".format(item)
            text += sep
        text += end
        if out == 'console':
            self.txt_console.insert(INSERT, text)
            self.txt_console.see("end")
        if out == 'notify':
            self.txt_notify.insert(INSERT, text)
            self.txt_notify.see("end")

    @staticmethod
    def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    # Save image to folder
    def save_image(self):
        if self.person_name == '':
            self.write('Person name required')
            return
        image_folder = os.path.join(self.data_folder, self.person_name)
        self.write('Saving image to {}'.format(image_folder))
        self.modified_folder.append(self.person_name)
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)

        for i, image in enumerate(self.image_buffer):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                image_folder + '/{}_{}_{}.jpg'.format(self.person_name, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                                                      i), image)

        thread = threading.Thread(target=self.create_images_with_masks, args=(image_folder,), daemon=True)
        thread.start()

    # Create image with masks
    def create_images_with_masks(self, image_folder):
        filenames = glob.glob(image_folder + '/*.jpg')
        images = [cv2.imread(img) for img in filenames]
        masks = [cv2.imread('images/masks/fm1.png'), cv2.imread('images/masks/fm2.png')]
        self.write('Generating face with mask')
        for idx, img in enumerate(images):
            out_path = '{}/face_with_mask_{}.png'.format(image_folder, idx)
            if os.path.exists(out_path):
                continue
            out_img = add_face_mask(img, masks[idx % len(masks)])
            cv2.imwrite(out_path, out_img)
            self.write('\t{}/{}'.format(idx + 1, len(images)))
        self.write('Done')

    def get_person_name(self, name):
        self.person_name = name

    # Stop face recognition engine
    def stop_face_recog(self):
        if self.is_running_fr:
            self.fr.stop_thread_face_recog()
            self.is_running_fr = False

    # Start face recognition engine
    def start_face_recog(self):
        if not self.is_running_fr:
            self.fr.stop_flag[0] = False
            thread = threading.Thread(target=self.fr.thread_face_recog, args=(), daemon=True)
            thread.start()
            self.is_running_fr = True

    # Resize keep aspect ratio
    @staticmethod
    def resize_with_pad(image, desired_size):
        old_size = image.size
        ratiow = float(desired_size[0]) / old_size[0]
        ratioh = float(desired_size[1]) / old_size[1]
        ratio = min(ratiow, ratioh)
        new_size = tuple([int(max(x * ratio, 1)) for x in old_size])
        image = image.resize(new_size, ImagePIL.ANTIALIAS)
        new_im = ImagePIL.new("RGB", desired_size)
        new_im.paste(image, ((desired_size[0] - new_size[0]) // 2,
                             (desired_size[1] - new_size[1]) // 2))
        return new_im

    # Update attendant to database
    @staticmethod
    def update_attendant(name, dt):
        record = Attendance(name=name, date=dt.date(), time=dt.time())
        session.add(record)
        session.commit()

    # Delete records except first and last
    @staticmethod
    def clean_up_attendant(name, date):
        date_str = date.strftime('%Y-%m-%d')
        with engine.connect() as con:
            con.execute('''
            DELETE FROM Attendance
             WHERE name = '{0}'
             AND (time !=
                 (
                 SELECT MAX(time)
                 FROM Attendance WHERE name = '{0}'
                 AND DATE = '{1}'
                 )
                 AND
                 TIME != (
                 SELECT MIN(TIME)
                 FROM Attendance WHERE name = '{0}'
                 AND DATE = '{1}'
                 )
             )
             AND DATE = '{1}'      
            '''.format(name, date_str))

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()


if __name__ == '__main__':
    app = FaceAttendanceApp()
    app.run()
