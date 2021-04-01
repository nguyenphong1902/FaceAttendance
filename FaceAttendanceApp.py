# pyinstaller --onedir --icon=icon.ico --clean --noconfirm FaceAttendanceApp.py
import os
import shutil
import time
import glob
import threading
from sys import exit
from datetime import datetime, timedelta
import cv2
import csv
from PIL import ImageTk
from PIL import Image as ImagePIL
from tkinter import *
from tkinter import messagebox, filedialog
from tkcalendar import *
import tkinter.ttk as ttk
# os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from configparser import ConfigParser
import pyttsx3 as pyttsx
import numpy as np
import sklearn.utils._weight_vector  # use when build .exe with pyinstaller
import babel.numbers  # use when build .exe with pyinstaller
import pyttsx3.drivers  # use when build .exe with pyinstaller
import pyttsx3.drivers.sapi5  # use when build .exe with pyinstaller
from face_mask_filter import add_face_mask, device_list
from FaceClassifier import FaceClassifier
from VideoCapture import MyVideoCapture
from FaceRecognition import FaceRecognition
from language import languages
from model import Attendance, Timekeeping, engine, session, Employee

pygame.mixer.init()
config_file = 'config.ini'
config = ConfigParser()
tts_engine = pyttsx.init()
voices = tts_engine.getProperty('voices')


class FaceAttendanceApp:
    def __init__(self):
        # GUI
        config.read(config_file)
        self.lang = config['settings']['language']
        self.lang_options = languages.keys()
        self.appName = languages[self.lang]['app_name']
        self.root = Tk()
        self.root.configure(bg='silver')
        self.root.iconbitmap("icon.ico")
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.root.title(self.appName)
        self.root.geometry("1080x720")
        self.root.minsize(1080, 720)
        self.root.resizable(True, True)
        self.lang_clicked = StringVar()
        self.lang_clicked.set(self.lang)
        self.fr_status = Frame(self.root)
        self.fr_status.grid(row=1, column=0, sticky=E)
        self.drd_language = OptionMenu(self.fr_status, self.lang_clicked, *self.lang_options,
                                       command=self.language_selected)
        self.drd_language.grid(row=0, column=1, sticky=E)
        self.drd_language.config(bg='gray')
        self.lbl_language = Label(self.fr_status, text=languages[self.lang]['lbl_language'])
        self.lbl_language.grid(row=0, column=0, sticky=E)
        self.drd_language['menu'].config(bg='light gray')
        self.tabControl = ttk.Notebook(self.root)
        self.tabFaceRecog = ttk.Frame(self.tabControl)
        self.tabAddFace = ttk.Frame(self.tabControl)
        self.tabAttendance = ttk.Frame(self.tabControl)
        self.tabControl.add(self.tabFaceRecog, text=languages[self.lang]['tab_facerecog'])
        self.tabControl.add(self.tabAddFace, text=languages[self.lang]['tab_addface'])
        self.tabControl.add(self.tabAttendance, text=languages[self.lang]['tab_data'])
        self.tabControl.grid(row=0, column=0, sticky=N + W + S + E)
        self.tabControl.bind("<<NotebookTabChanged>>", self.on_tab_selected)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        pygame.mixer.music.load('audio/confirm_sound.mp3')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.settings = None

        # Params
        self.start_hr_1 = datetime.strptime(config['settings']['start_hr_1'], '%H:%M').time()
        self.end_hr_1 = datetime.strptime(config['settings']['end_hr_1'], '%H:%M').time()
        self.start_hr_2 = datetime.strptime(config['settings']['start_hr_2'], '%H:%M').time()
        self.end_hr_2 = datetime.strptime(config['settings']['end_hr_2'], '%H:%M').time()
        self.cam_num = 2
        self.is_record = False
        self.record_count = 0
        self.record_total = 10
        self.image_buffer = []
        self.data_folder = 'images/data/train'
        self.image_folder_name = ''
        self.modified_folder = []
        self.name_cache = []
        self.photo_in = None
        self.photo_out = None
        self.frame_lock = threading.Lock()
        if self.cam_num == 2:
            self.cam_resolution = (640, 480)
        else:
            self.cam_resolution = (1280, 720)

        self.is_running_fr = False
        self.stop_update_frame_facerecog = False
        self.stop_update_frame_addface = False
        self.name_last_frame = [[], []]
        self.last_boxes = [[], []]
        self.last_texts = [[], []]
        self.last_colors = [[], []]
        self.skip_list = [[], []]
        self.table = []
        self.emp_name_list = []
        self.emp_id_list = []
        self.get_employee_from_db()
        self.mute_tts = False
        self.speak_lock = threading.Lock()
        self.emp_notify_lang = ''

        # Video Capture
        self.vids = [MyVideoCapture(video_source=0, width=self.cam_resolution[0], height=self.cam_resolution[1]),
                     MyVideoCapture(video_source=1, width=self.cam_resolution[0], height=self.cam_resolution[1])]
        self.cam_in_idx = int(config['settings']['cam_in_idx'])
        self.cam_out_idx = int(config['settings']['cam_out_idx'])
        if self.cam_in_idx == 0:
            self.cam_pos = ['in', 'out']
        else:
            self.cam_pos = ['out', 'in']
        self.cam_capture_idx = 0
        # Init face classifier
        self.fc = FaceClassifier()
        self.fc.load_pretrained_data('data.pt')
        # Init face recognition engine
        self.min_face = [int(config['settings']['min_face_1']), int(config['settings']['min_face_1'])]
        self.accuracy_th = float(config['settings']['accuracy_th'])
        self.fr = FaceRecognition(min_face=self.min_face, accuracy_th=self.accuracy_th)
        if self.fr.class_names is not None:
            for key, value in self.fr.class_names.items():
                self.name_cache.append(value)

        self.name_counter = [[0] * len(self.fr.class_names), [0] * len(self.fr.class_names)]
        self.name_time_pause = [[datetime.min] * len(self.fr.class_names), [datetime.min] * len(self.fr.class_names)]
        self.green_box_counter = [[0] * len(self.fr.class_names), [0] * len(self.fr.class_names)]

        # Tab FaceRecog
        self.top_frame_facerecog = Frame(self.tabFaceRecog)
        self.bottom_frame_facerecog = Frame(self.tabFaceRecog)
        self.top_frame_facerecog.grid(row=0, column=0, sticky=W + E)
        self.bottom_frame_facerecog.grid(row=1, column=0, sticky=N + W + E + S)
        self.label = Label(self.top_frame_facerecog, text=self.appName, font=15, bg='light steel blue', fg='black')
        self.label.grid(row=0, column=0, sticky=W)
        self.frame_canvas = LabelFrame(self.bottom_frame_facerecog, bg='light steel blue')
        self.frame_canvas.grid(row=0, column=0, sticky=N + W + E + S)
        self.canvas_facerecog_in = Canvas(self.frame_canvas, bg='black')
        self.canvas_facerecog_in.grid(row=0, column=0, sticky=N + W + E + S)

        self.frame_canvas.rowconfigure(0, weight=1)
        self.frame_canvas.columnconfigure(0, weight=1)
        if self.cam_num == 2:
            self.canvas_facerecog_out = Canvas(self.frame_canvas, bg='black')
            self.canvas_facerecog_out.grid(row=0, column=1, sticky=N + W + E + S)
            self.frame_canvas.columnconfigure(1, weight=1)
            self.lbl_facerecog_in = Label(self.frame_canvas, text=languages[self.lang]['lbl_enter'], font=25,
                                          bg='green',
                                          fg='white')
            self.lbl_facerecog_in.grid(row=0, column=0, sticky=N, pady=10)
            self.lbl_facerecog_out = Label(self.frame_canvas, text=languages[self.lang]['lbl_exit'], font=25, bg='red',
                                           fg='white')
            self.lbl_facerecog_out.grid(row=0, column=1, sticky=N, pady=10)

        self.frame_notify = LabelFrame(self.bottom_frame_facerecog, text=languages[self.lang]['fr_notify'],
                                       bg='light steel blue')
        self.frame_notify.grid(row=0, column=1, sticky=N + W + E + S)
        self.scroll_notify = Scrollbar(self.frame_notify, orient=VERTICAL)
        self.scroll_notify.pack(side=RIGHT, fill=Y)
        self.txt_notify = Text(self.frame_notify, bg='black', fg='white', width=18, yscroll=self.scroll_notify.set)
        self.txt_notify.pack(fill=Y, expand=YES)
        self.scroll_notify.config(command=self.txt_notify.yview)

        self.bottom_frame_facerecog.rowconfigure(0, weight=1)
        self.bottom_frame_facerecog.columnconfigure(0, weight=1)

        self.btnRefreshFR = ttk.Button(self.top_frame_facerecog, text=languages[self.lang]['btn_refresh'],
                                       command=self.btnRefreshFRPressed)
        self.btnRefreshFR.grid(row=0, column=2, sticky=W, padx=10)
        self.btnCamSettings = ttk.Button(self.top_frame_facerecog, text=languages[self.lang]['btn_cam_settings'],
                                         command=self.btnCamSettingsPressed)
        self.btnCamSettings.grid(row=0, column=1, sticky=W, padx=10)
        self.chkbtn_var = IntVar()
        self.chkbtn_mute_tts = Checkbutton(self.top_frame_facerecog, text=languages[self.lang]['chkbtn_mute_tts'],
                                           command=self.on_chkbtn_mute_tts_clicked, variable=self.chkbtn_var)
        self.chkbtn_var.set(int(config['settings']['mute_tts']))
        self.chkbtn_mute_tts.grid(row=0, column=3, sticky=W, padx=(10, 0))
        self.on_chkbtn_mute_tts_clicked()

        self.top_frame_facerecog.rowconfigure(0, weight=1)
        self.top_frame_facerecog.columnconfigure(1, weight=1)

        self.tabFaceRecog.rowconfigure(1, weight=1)
        self.tabFaceRecog.columnconfigure(0, weight=1)

        # Tab AddFace
        self.left_frame_addface = LabelFrame(self.tabAddFace, text=languages[self.lang]['lblfr_camera'],
                                             bg='light steel blue')
        self.right_frame_addface = LabelFrame(self.tabAddFace, text=languages[self.lang]['lblfr_button'], width=150)
        self.bottom_frame_addface = LabelFrame(self.tabAddFace, text=languages[self.lang]['lblfr_console'],
                                               bg='light steel blue')
        self.left_frame_addface.grid(row=0, column=0, sticky=N + W + E + S)
        self.right_frame_addface.grid(row=0, column=1, sticky=N + W + E + S)
        self.bottom_frame_addface.grid(row=1, column=0, columnspan=2, sticky=W + E)
        self.tabAddFace.rowconfigure(0, weight=1)
        self.tabAddFace.columnconfigure(0, weight=1)
        self.canvas_addface = Canvas(self.left_frame_addface, width=300, height=300, bg='black')
        self.canvas_addface.pack(fill=BOTH, expand=YES)

        self.fr_training = LabelFrame(self.right_frame_addface, text=languages[self.lang]['lblfr_training'])
        self.fr_training.grid(row=0, column=0, sticky=N+S+W+E)
        self.cam_capture_id = StringVar()
        self.cbb_chose_cam_capture = ttk.Combobox(self.fr_training, width=12, textvariable=self.cam_capture_id)
        if len(device_list) >= 2:
            self.cam_capture_id.set('0-' + device_list[0])
            self.cbb_chose_cam_capture['values'] = ('0-' + device_list[0], '1-' + device_list[1])
        self.cbb_chose_cam_capture.bind('<<ComboboxSelected>>', self.cam_capture_id_changed)
        self.cbb_chose_cam_capture.grid(row=0, column=0)
        self.lblRecord = Label(self.fr_training, text=languages[self.lang]['lbl_record'])
        self.lblRecord.grid(row=1, column=0)
        self.entrRecord = Entry(self.fr_training, width=4, borderwidth=2)
        self.entrRecord.insert(END, str(self.record_total))
        self.entrRecord.grid(row=1, column=1, pady=10 ,sticky=W)
        self.btnRecord = ttk.Button(self.fr_training, text=languages[self.lang]['btn_record'],
                                    command=self.btnRecordPressed)
        self.btnRecord.grid(row=2, column=0, pady=10, columnspan=2)
        self.btnTrain = ttk.Button(self.fr_training, text=languages[self.lang]['btn_train'],
                                   command=self.btnTrainPressed)
        self.btnTrain.grid(row=3, column=0, pady=10, columnspan=2)
        self.btnReTrainAll = ttk.Button(self.fr_training, text=languages[self.lang]['btn_retrain_all'],
                                        command=self.btnReTrainAllPressed)
        self.btnReTrainAll.grid(row=4, column=0, pady=10, columnspan=2)
        self.fr_training.columnconfigure(0, weight=1)
        self.fr_training.columnconfigure(1, weight=1)

        self.fr_employee_edit = LabelFrame(self.right_frame_addface, text=languages[self.lang]['fr_employee_edit'])
        self.fr_employee_edit.grid(row=1, column=0, sticky=N+S+W+E)
        self.lbl_employee = Label(self.fr_employee_edit, text=languages[self.lang]['lbl_employee'])
        self.lbl_employee.grid(row=0, column=0, pady=10)
        self.btnDelete = ttk.Button(self.fr_employee_edit, text=languages[self.lang]['btn_delete'],
                                    command=self.btnDeletePressed)
        self.btnDelete.grid(row=1, column=0, pady=10)
        self.btnChangeName = ttk.Button(self.fr_employee_edit, text=languages[self.lang]['btn_change_name'],
                                    command=self.btnChangeNamePressed)
        self.btnChangeName.grid(row=1, column=1, pady=10)
        self.name_to_del = StringVar()
        self.cbb_chose_delete = ttk.Combobox(self.fr_employee_edit, width=18, textvariable=self.name_to_del)
        self.cbb_chose_delete['values'] = [name for name in os.listdir(self.data_folder) if
                                           os.path.isdir(os.path.join(self.data_folder, name))]
        self.cbb_chose_delete.bind('<Button-1>', self.on_cbb_del_clicked)
        self.cbb_chose_delete.grid(row=0, column=1, pady=10)
        self.right_frame_addface.rowconfigure(0, weight=1)
        self.right_frame_addface.rowconfigure(1, weight=1)
        self.right_frame_addface.columnconfigure(0, weight=1)
        self.scroll_console = Scrollbar(self.bottom_frame_addface, orient=VERTICAL)
        self.scroll_console.pack(side=RIGHT, fill=Y)
        self.txt_console = Text(self.bottom_frame_addface, bg='black', fg='white', height=7,
                                yscroll=self.scroll_console.set)
        self.txt_console.pack(fill=BOTH, expand=YES)
        self.scroll_console.config(command=self.txt_console.yview)

        # Tab Attendance
        self.top_frame_attend = LabelFrame(self.tabAttendance)
        self.bottom_frame_attend = LabelFrame(self.tabAttendance, text=languages[self.lang]['lblfr_data_table'],
                                              bg='light steel blue')
        self.top_frame_attend.grid(row=0, column=0, sticky=W + E, pady=5)
        self.bottom_frame_attend.grid(row=1, column=0, sticky=N + W + E + S)
        self.btnShowAttend = ttk.Button(self.top_frame_attend, text=languages[self.lang]['btn_show_data'],
                                        command=self.btnShowAttendPressed)
        self.btnShowAttend.grid(row=0, column=0, sticky=W, padx=5)
        self.btnCleanAttend = ttk.Button(self.top_frame_attend, text=languages[self.lang]['btn_clean_data'],
                                         command=self.btnCleanAttendPressed)
        self.btnCleanAttend.grid(row=0, column=1, sticky=W, padx=5)
        self.btnCleanAttend.grid_forget()
        self.btnSaveExcel = ttk.Button(self.top_frame_attend, text=languages[self.lang]['btn_save_table'],
                                       command=self.btnSaveExcelPressed)
        self.btnSaveExcel.grid(row=0, column=2, sticky=W, padx=5)
        self.lbl_chose_date_from = Label(self.top_frame_attend, text=languages[self.lang]['lbl_from'])
        self.lbl_chose_date_from.grid(row=0, column=3, sticky=E)
        self.dtentry_from = DateEntry(self.top_frame_attend, width=12, background='steel blue', foreground='white',
                                      borderwidth=2, date_pattern='y-mm-dd')
        self.dtentry_from.grid(row=0, column=4, sticky=W)
        self.lbl_chose_date_to = Label(self.top_frame_attend, text=languages[self.lang]['lbl_to'])
        self.lbl_chose_date_to.grid(row=0, column=5, sticky=E, padx=(10, 0))
        self.dtentry_to = DateEntry(self.top_frame_attend, width=12, background='steel blue', foreground='white',
                                    borderwidth=2, date_pattern='y-mm-dd')
        self.dtentry_to.grid(row=0, column=6, sticky=W)
        self.lbl_chose_name = Label(self.top_frame_attend, text=languages[self.lang]['lbl_name'])
        self.lbl_chose_name.grid(row=0, column=7, sticky=E, padx=(20, 0))
        self.name_filter = StringVar()
        self.cbb_chose_name = ttk.Combobox(self.top_frame_attend, width=18, textvariable=self.name_filter)

        self.cbb_chose_name.bind('<Button-1>', self.cbb_chose_name_clicked)
        self.name_filter.set('All')
        self.cbb_chose_name.grid(row=0, column=8)
        self.top_frame_attend.columnconfigure(2, weight=1)
        self.scroll_table = Scrollbar(self.bottom_frame_attend, orient=VERTICAL)
        self.scroll_table.pack(side=RIGHT, fill=Y)
        self.tblAttend = ttk.Treeview(self.bottom_frame_attend, yscroll=self.scroll_table.set)
        self.scroll_table.config(command=self.tblAttend.yview)

        self.tabAttendance.rowconfigure(1, weight=1)
        self.tabAttendance.columnconfigure(0, weight=1)

    # Popup
    def popup_input_frame(self, mode='add_new', emp_id=''):
        input_name = Toplevel()
        if mode == 'add_new':
            input_name.wm_title(languages[self.lang]['popup_input_name_title'])
        if mode == 'change_name':
            input_name.wm_title(languages[self.lang]['btn_change_name'])
        lab = Label(input_name, text=languages[self.lang]['popup_lbl_input_name'], borderwidth=2)
        lab.grid(row=0, column=0, padx=10, pady=10)
        txtBox = Entry(input_name, width=20, borderwidth=2)
        txtBox.grid(row=0, column=1, padx=10, pady=10)
        if mode == 'add_new':
            lab_id = Label(input_name, text='ID', borderwidth=2)
            lab_id.grid(row=0, column=2, padx=10, pady=10)
            txtBox_id = Entry(input_name, width=20, borderwidth=2)
            txtBox_id.grid(row=0, column=3, padx=10, pady=10)
        lbl_notify_lang = Label(input_name, text=languages[self.lang]['lbl_language'])
        lbl_notify_lang.grid(row=0, column=4, sticky=E)
        notify_lang = StringVar()
        cbb_notify_lang = ttk.Combobox(input_name, width=5, textvariable=notify_lang)
        cbb_notify_lang['values'] = list(self.lang_options)
        notify_lang.set('VN')
        cbb_notify_lang.grid(row=0, column=5, pady=10)
        if mode == 'add_new':
            btnOkay = ttk.Button(input_name, text=languages[self.lang]['popup_btn_okay'],
                                 command=lambda: [self.get_person_name(txtBox.get(), txtBox_id.get(), notify_lang.get()),
                                                  input_name.destroy(), self.save_image()])
            btnOkay.grid(row=1, column=3, padx=10, pady=10, sticky=E)
        if mode == 'change_name':
            btnOkay = ttk.Button(input_name, text=languages[self.lang]['btn_change_name'],
                                 command=lambda: [self.update_employee(txtBox.get(), emp_id, notify_lang.get()),
                                                  input_name.destroy()])
            btnOkay.grid(row=1, column=1, padx=10, pady=10, sticky=E)
        btnCancel = ttk.Button(input_name, text=languages[self.lang]['popup_btn_cancel'], command=input_name.destroy)
        btnCancel.grid(row=1, column=5, padx=10, pady=10, sticky=E)
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        wf = input_name.winfo_width()
        hf = input_name.winfo_height()
        x = (w - wf) / 2
        y = (h - hf) / 2
        input_name.geometry('+%d+%d' % (x - 25, y))
        self.root.wait_window(input_name)

    def setting_window(self):
        self.settings = Toplevel()
        self.settings.resizable(0, 0)
        self.settings.wm_title(languages[self.lang]['setting_window_title'])
        fr_shift_setting = LabelFrame(self.settings, text=languages[self.lang]['lblfr_shift_setting'])
        fr_cam_setting = LabelFrame(self.settings, text=languages[self.lang]['lblfr_cam_setting'])
        lbl_chose_cam = Label(fr_cam_setting, text=languages[self.lang]['lbl_chose_cam'], font=12)
        self.cam_id = StringVar()
        self.cbb_chose_cam = ttk.Combobox(fr_cam_setting, width=12, textvariable=self.cam_id)
        if len(device_list) >= 2:
            self.cam_id.set('0-' + device_list[0])
            self.cbb_chose_cam['values'] = ('0-' + device_list[0], '1-' + device_list[1])
        self.cbb_chose_cam.bind('<<ComboboxSelected>>', self.cam_id_changed)
        self.r = StringVar()
        self.r.set(self.cam_pos[0])
        self.rad_cam_in = Radiobutton(fr_cam_setting, text=languages[self.lang]['lbl_enter'], variable=self.r,
                                      value='in', command=lambda: self.rad_cam_changed(self.rad_cam_in))
        self.rad_cam_out = Radiobutton(fr_cam_setting, text=languages[self.lang]['lbl_exit'], variable=self.r,
                                       value='out', command=lambda: self.rad_cam_changed(self.rad_cam_out))
        lblMinFace = Label(fr_cam_setting, text=languages[self.lang]['lbl_minface'])
        self.sv_min_face = StringVar()
        self.sv_min_face.trace_add("write", self.entry_minface_changed)
        self.entrMinFace = Entry(fr_cam_setting, width=4, textvariable=self.sv_min_face, borderwidth=2)
        self.entrMinFace.insert(END, str(self.min_face[0]))
        lblAccuracyTH = Label(fr_cam_setting, text=languages[self.lang]['lbl_accuracyTH'], font=12)
        self.entrAccuracyTH = Entry(fr_cam_setting, width=4, borderwidth=2)
        self.entrAccuracyTH.insert(END, str(self.accuracy_th))
        lblShift = Label(fr_shift_setting, text=languages[self.lang]['lbl_shift'])
        lblto = Label(fr_shift_setting, text=languages[self.lang]['lbl_to'])
        sv_start_hr_1 = StringVar()
        sv_start_hr_2 = StringVar()
        sv_end_hr_1 = StringVar()
        sv_end_hr_2 = StringVar()
        entr_start_hr_1 = Entry(fr_shift_setting, width=10, textvariable=sv_start_hr_1, borderwidth=2)
        entr_start_hr_1.insert(END, self.start_hr_1.strftime('%H:%M'))
        entr_end_hr_1 = Entry(fr_shift_setting, width=10, textvariable=sv_end_hr_1, borderwidth=2)
        entr_end_hr_1.insert(END, self.end_hr_1.strftime('%H:%M'))
        lblto2 = Label(fr_shift_setting, text=languages[self.lang]['lbl_to'])
        entr_start_hr_2 = Entry(fr_shift_setting, width=10, textvariable=sv_start_hr_2, borderwidth=2)
        entr_start_hr_2.insert(END, self.start_hr_2.strftime('%H:%M'))
        entr_end_hr_2 = Entry(fr_shift_setting, width=10, textvariable=sv_end_hr_2, borderwidth=2)
        entr_end_hr_2.insert(END, self.end_hr_2.strftime('%H:%M'))
        btnSave = ttk.Button(self.settings, text=languages[self.lang]['popup_btn_save'],
                             command=lambda: [
                                 self.save_setting(sv_start_hr_1.get(), sv_end_hr_1.get(), sv_start_hr_2.get(),
                                                   sv_end_hr_2.get()),
                                 self.settings.destroy()]
                             )
        btnCancel = ttk.Button(self.settings, text=languages[self.lang]['popup_btn_cancel'],
                               command=self.settings.destroy)

        fr_cam_setting.grid(row=0, column=0, columnspan=4, sticky=N + W + E + S)
        fr_shift_setting.grid(row=1, column=0, columnspan=4, sticky=N + W + E + S)
        btnSave.grid(row=2, column=2, sticky=E)
        btnCancel.grid(row=2, column=3, sticky=E)

        self.settings.rowconfigure(0, weight=1)
        self.settings.rowconfigure(1, weight=1)
        self.settings.columnconfigure(0, weight=1)

        lbl_chose_cam.grid(row=0, column=0, sticky=W, padx=10, pady=10)
        self.cbb_chose_cam.grid(row=0, column=1, sticky=W)
        self.rad_cam_in.grid(row=0, column=2, sticky=W, padx=10)
        self.rad_cam_out.grid(row=0, column=3, sticky=W, padx=10)
        lblMinFace.grid(row=1, column=2, padx=10)
        self.entrMinFace.grid(row=1, column=3)
        lblAccuracyTH.grid(row=2, column=0, sticky=E, padx=10, pady=10)
        self.entrAccuracyTH.grid(row=2, column=1, sticky=W)

        lblShift.grid(row=0, column=0, padx=5, pady=10)
        entr_start_hr_1.grid(row=0, column=1, padx=5, pady=10)
        lblto.grid(row=0, column=2, padx=5, pady=10)
        entr_end_hr_1.grid(row=0, column=3, padx=5, pady=10)
        entr_start_hr_2.grid(row=1, column=1, padx=5, pady=10)
        lblto2.grid(row=1, column=2, padx=5, pady=10)
        entr_end_hr_2.grid(row=1, column=3, padx=5, pady=10)
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        wf = self.settings.winfo_width()
        hf = self.settings.winfo_height()
        x = (w - wf) / 2
        y = (h - hf) / 2
        self.settings.geometry('+%d+%d' % (x - 25, y))
        self.root.wait_window(self.settings)

    def entry_minface_changed(self, *args):
        cam_id = int(self.cam_id.get().split('-')[0])
        if self.sv_min_face.get() != '':
            try:
                self.min_face[cam_id] = int(self.sv_min_face.get())
            except ValueError:
                pass

    def rad_cam_changed(self, widget):
        cam_id = int(self.cam_id.get().split('-')[0])
        if self.r.get() == 'in' and cam_id == 0:
            self.cam_pos = ['in', 'out']
        elif self.r.get() == 'out' and cam_id == 1:
            self.cam_pos = ['in', 'out']
        else:
            self.cam_pos = ['out', 'in']

    def cam_id_changed(self, event):
        cam_id = int(event.widget.get().split('-')[0])
        min_face = self.min_face[cam_id]
        self.entrMinFace.delete(0, 'end')
        self.entrMinFace.insert(END, str(min_face))
        self.r.set(self.cam_pos[cam_id])

    def cam_capture_id_changed(self, event):
        self.cam_capture_idx = int(event.widget.get().split('-')[0])

    def save_setting(self, *args):
        try:
            start1 = datetime.strptime(args[0], '%H:%M').time()
            end1 = datetime.strptime(args[1], '%H:%M').time()
            start2 = datetime.strptime(args[2], '%H:%M').time()
            end2 = datetime.strptime(args[3], '%H:%M').time()
            if start1 < end1 < start2 < end2:
                self.start_hr_1 = start1
                self.end_hr_1 = end1
                self.start_hr_2 = start2
                self.end_hr_2 = end2
        except ValueError:
            pass

        with self.fr.lock_cap:
            if self.cam_pos[0] == 'in':
                self.cam_in_idx = 0
                self.cam_out_idx = 1
            else:
                self.cam_in_idx = 1
                self.cam_out_idx = 0
        try:
            self.accuracy_th = float(self.entrAccuracyTH.get())
        except ValueError:
            self.entrAccuracyTH.delete(0, 'end')
        self.btnRefreshFRPressed()
        self.update_config()

    def update_config(self):
        config.set('settings', 'start_hr_1', self.start_hr_1.strftime('%H:%M'))
        config.set('settings', 'end_hr_1', self.end_hr_1.strftime('%H:%M'))
        config.set('settings', 'start_hr_2', self.start_hr_2.strftime('%H:%M'))
        config.set('settings', 'end_hr_2', self.end_hr_2.strftime('%H:%M'))
        config.set('settings', 'cam_in_idx', str(self.cam_in_idx))
        config.set('settings', 'cam_out_idx', str(self.cam_out_idx))
        config.set('settings', 'min_face_1', str(self.min_face[0]))
        config.set('settings', 'min_face_2', str(self.min_face[1]))
        config.set('settings', 'accuracy_th', str(self.accuracy_th))
        config.set('settings', 'language', str(self.lang))
        config.set('settings', 'mute_tts', str(self.chkbtn_var.get()))
        with open(config_file, 'w') as config_write:
            config.write(config_write)

    # Control
    def language_selected(self, lang):
        self.change_language(lang)

    def btnSaveExcelPressed(self):
        if not self.table:
            return
        file = filedialog.asksaveasfilename(initialdir=os.getcwd(),
                                            defaultextension='.csv',
                                            filetypes=[('CSV file', '.csv')]
                                            )
        if file == '':
            return
        with open(file, mode='w', newline='') as csv_file:
            fieldnames = self.table[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.table:
                writer.writerow(row)

    def btnDeletePressed(self):
        folder = self.name_to_del.get()
        if folder == '':
            return
        try:
            emp_id = self.name_to_del.get().split('_ID_')[1]
            with engine.connect() as con:
                con.execute('''DELETE FROM Employee WHERE id = '{}' '''.format(emp_id))
                con.execute('''DELETE FROM Attendance WHERE employeeID = '{}' '''.format(emp_id))
                con.execute('''DELETE FROM Timekeeping WHERE employeeID = '{}' '''.format(emp_id))
            self.write('Delete successful: {}'.format(os.path.join(self.data_folder, folder)))
            shutil.rmtree(os.path.join(self.data_folder, folder))
            if folder in self.modified_folder:
                self.modified_folder.remove(folder)
            self.btnTrainPressed()
        except OSError as e:
            self.write("Error: %s - %s." % (e.filename, e.strerror))

    def btnChangeNamePressed(self):
        try:
            emp_id = self.cbb_chose_delete.get().split('_ID_')[1]
        except Exception:
            self.write('Wrong format, no employee found')
            return
        self.popup_input_frame(mode='change_name', emp_id=emp_id)

    def btnCamSettingsPressed(self):
        if self.settings is None or not self.settings.winfo_exists():
            self.setting_window()
        else:
            self.settings.lift()

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
        date_from = self.dtentry_from.get()
        date_to = self.dtentry_to.get()
        namenid = self.name_filter.get()
        self.tblAttend.delete(*self.tblAttend.get_children())
        frame_width = self.bottom_frame_attend.winfo_width()
        with engine.connect() as con:
            if namenid == 'All' or namenid == '':
                rs = con.execute('''SELECT * FROM Timekeeping
                                 WHERE date BETWEEN '{}' AND '{}'
                                 '''.format(date_from, date_to))
            else:
                name = namenid.split('_ID_')[0]
                emp_id = namenid.split('_ID_')[1]
                rs = con.execute('''SELECT * FROM Timekeeping
                             WHERE date BETWEEN '{}' AND '{}'
                             AND name = '{}'
                             AND employeeID = '{}'
                             '''.format(date_from, date_to, name, emp_id))
            createCols = True
            self.table.clear()
            for row in rs:
                di = dict(row)
                if createCols:
                    keys = di.keys()
                    colnum = len(keys)
                    self.tblAttend['columns'] = list(keys)
                    self.tblAttend['show'] = 'headings'
                    for key in keys:
                        self.tblAttend.heading(str(key), text=key)
                        self.tblAttend.column(str(key), minwidth=0, width=int(frame_width / colnum), stretch=NO)
                    self.tblAttend.pack()
                    createCols = False
                if di.get('OT') < 60:
                    di['OT'] = 0
                self.tblAttend.insert("", END, values=list(di.values()))
                self.table.append(di)

    def btnRefreshFRPressed(self):
        self.stop_face_recog()
        time.sleep(0.5)
        self.fr.load_model()
        self.name_cache = []
        for key, value in self.fr.class_names.items():
            self.name_cache.append(value)
        self.name_counter = [[0] * len(self.fr.class_names), [0] * len(self.fr.class_names)]
        self.name_time_pause = [[datetime.min] * len(self.fr.class_names), [datetime.min] * len(self.fr.class_names)]
        self.green_box_counter = [[0] * len(self.fr.class_names), [0] * len(self.fr.class_names)]
        self.fr.set_params(self.min_face, self.accuracy_th)
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
        self.btnDelete['state'] = 'disable'
        self.btnChangeName['state'] = 'disable'

    def enable_buttons(self):
        self.btnTrain['state'] = 'normal'
        self.btnRecord['state'] = 'normal'
        self.btnDelete['state'] = 'normal'
        self.btnChangeName['state'] = 'normal'
        self.btnReTrainAll['state'] = 'normal'
        self.tabControl.tab(0, state='normal')
        self.tabControl.tab(2, state='normal')

    # Event when changing tab
    def on_tab_selected(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == languages[self.lang]['tab_facerecog']:
            self.stop_update_frame_addface = True
            self.stop_update_frame_facerecog = False
            self.fr.load_model()
            self.name_cache = []
            for key, value in self.fr.class_names.items():
                self.name_cache.append(value)
            self.name_counter = [[0] * len(self.fr.class_names), [0] * len(self.fr.class_names)]
            self.name_time_pause = [[datetime.min] * len(self.fr.class_names),
                                    [datetime.min] * len(self.fr.class_names)]
            self.green_box_counter = [[0] * len(self.fr.class_names), [0] * len(self.fr.class_names)]
            self.update_frame_facerecog()
            self.start_face_recog()
        if tab_text == languages[self.lang]['tab_addface']:
            self.stop_face_recog()
            self.stop_update_frame_facerecog = True
            self.stop_update_frame_addface = False
            self.update_frame_addface()
        if tab_text == languages[self.lang]['tab_data']:
            self.stop_face_recog()
            self.stop_update_frame_facerecog = True
            self.stop_update_frame_addface = True

    def on_cbb_del_clicked(self, event):
        self.get_employee_from_db()
        list_emp = []
        for i, emp_id in enumerate(self.emp_id_list):
            list_emp.append(self.emp_name_list[i] + '_ID_' + emp_id)
        # self.cbb_chose_delete['values'] = [name for name in os.listdir(self.data_folder) if
        #                                    os.path.isdir(os.path.join(self.data_folder, name))]
        self.cbb_chose_delete['value'] = list_emp

    def cbb_chose_name_clicked(self, event):
        name_list = []
        with engine.connect() as con:
            rs = con.execute('''SELECT DISTINCT name, employeeID FROM Timekeeping''')
            for row in rs:
                name_list.append(list(row)[0] + '_ID_' + list(row)[1])
        self.cbb_chose_name['values'] = name_list
        self.cbb_chose_name['values'] = (*self.cbb_chose_name['values'], 'All')

    def on_chkbtn_mute_tts_clicked(self):
        if self.chkbtn_var.get() == 1:
            self.mute_tts = True
        elif self.chkbtn_var.get() == 0:
            self.mute_tts = False

    def change_language(self, lang):
        self.lang = lang
        self.appName = languages[self.lang]['app_name']
        self.root.title(self.appName)
        self.label['text'] = languages[self.lang]['app_name']
        self.tabControl.tab(0, text=languages[self.lang]['tab_facerecog'])
        self.tabControl.tab(1, text=languages[self.lang]['tab_addface'])
        self.tabControl.tab(2, text=languages[self.lang]['tab_data'])
        self.lbl_facerecog_in['text'] = languages[self.lang]['lbl_enter']
        self.lbl_facerecog_out['text'] = languages[self.lang]['lbl_exit']
        self.frame_notify['text'] = languages[self.lang]['fr_notify']
        self.btnRefreshFR['text'] = languages[self.lang]['btn_refresh']
        self.left_frame_addface['text'] = languages[self.lang]['lblfr_camera']
        self.right_frame_addface['text'] = languages[self.lang]['lblfr_button']
        self.bottom_frame_addface['text'] = languages[self.lang]['lblfr_console']
        self.lblRecord['text'] = languages[self.lang]['lbl_record']
        self.btnRecord['text'] = languages[self.lang]['btn_record']
        self.btnTrain['text'] = languages[self.lang]['btn_train']
        self.btnReTrainAll['text'] = languages[self.lang]['btn_retrain_all']
        self.bottom_frame_attend['text'] = languages[self.lang]['lblfr_data_table']
        self.btnShowAttend['text'] = languages[self.lang]['btn_show_data']
        self.btnCleanAttend['text'] = languages[self.lang]['btn_clean_data']
        self.lbl_language['text'] = languages[self.lang]['lbl_language']
        self.btnCamSettings['text'] = languages[self.lang]['btn_cam_settings']
        self.lbl_chose_date_to['text'] = languages[self.lang]['lbl_to']
        self.lbl_chose_date_from['text'] = languages[self.lang]['lbl_from']
        self.btnSaveExcel['text'] = languages[self.lang]['btn_save_table']
        self.lbl_chose_name['text'] = languages[self.lang]['lbl_name']
        self.btnDelete['text'] = languages[self.lang]['btn_delete']
        self.chkbtn_mute_tts['text'] = languages[self.lang]['chkbtn_mute_tts']
        self.fr_employee_edit['text'] = languages[self.lang]['fr_employee_edit']
        self.fr_training['text'] = languages[self.lang]['lblfr_training']
        self.lbl_employee['text'] = languages[self.lang]['lbl_employee']
        self.btnChangeName['text'] = languages[self.lang]['btn_change_name']

    # Update frame face recognition
    def update_frame_facerecog(self):
        if self.stop_update_frame_facerecog:
            return
        with self.fr.lock_cap:
            isTrue2, frame_out = self.vids[self.cam_out_idx].get_frame()
            isTrue, frame_in = self.vids[self.cam_in_idx].get_frame()
        if isTrue and isTrue2:
            with self.fr.lock_boxes:
                if self.fr.new_boxes:
                    is_new_boxes = True
                    self.last_boxes = self.fr.box_draw
                    self.last_texts = self.fr.text_draw
                    self.fr.new_boxes = False
                else:
                    is_new_boxes = False

            if is_new_boxes:
                self.skip_list = [[] for x in range(0, len(self.last_boxes))]
                self.last_colors = [[] for x in range(0, len(self.last_boxes))]
                for idx, color in enumerate(self.last_colors):
                    self.last_colors[idx] = [(255, 255, 0)] * len(self.last_boxes[idx])

                for j, texts in enumerate(self.last_texts):
                    list_index = []
                    for idx, text in enumerate(texts):
                        name = text.split(':')[0]
                        if name != 'Unknown':
                            try:
                                index = self.name_cache.index(name)
                            except ValueError:
                                continue
                            list_index.append(index)
                            if self.name_time_pause[j][index] == datetime.min:
                                if self.name_counter[j][index] == 0:
                                    self.name_counter[j][index] += 1
                                else:
                                    if index not in self.name_last_frame[j]:
                                        self.name_counter[j][index] = 0
                                    self.name_counter[j][index] += 1
                                if self.name_counter[j][index] >= 3:
                                    if j == self.cam_in_idx:
                                        status = self.lbl_facerecog_in['text']
                                        stt = 'in'
                                        thumbnail = self.get_thumbnail(frame_in, self.last_boxes[self.cam_in_idx][idx],
                                                                       (160, 160))
                                    else:
                                        status = self.lbl_facerecog_out['text']
                                        stt = 'out'
                                        thumbnail = self.get_thumbnail(frame_out,
                                                                       self.last_boxes[self.cam_out_idx][idx],
                                                                       (160, 160))
                                    self.write('\n' + name.replace('_ID_', ' ') + '\n' + status + '\n' + datetime.now().strftime(
                                        "%Y-%m-%d, %H:%M")
                                               + '\n' + '_' * self.txt_notify['width'], out='notify')
                                    if thumbnail is not None:
                                        thumbnail_path = 'images/data/log/' + name + '_' + stt + '_' + datetime.now().strftime(
                                            "%Y-%m-%d_%H-%M") + '.jpg'
                                        is_success, im_buf_arr = cv2.imencode(".jpg", thumbnail)
                                        if is_success:
                                            im_buf_arr.tofile(thumbnail_path)
                                    if self.mute_tts:
                                        pygame.mixer.music.play()
                                    else:
                                        thread_speak = threading.Thread(target=self.speak, args=(name, stt,),
                                                                        daemon=True)
                                        thread_speak.start()
                                    self.update_attendant(name, datetime.now(), stt)
                                    self.update_timekeeping(name, datetime.now(), stt)
                                    self.last_colors[j][idx] = (0, 255, 0)
                                    self.name_time_pause[j][index] = datetime.now() + timedelta(seconds=6)
                                    self.name_counter[j][index] = 0
                            else:
                                if self.name_time_pause[j][index] > datetime.now():
                                    self.green_box_counter[j][index] += 1
                                    if self.green_box_counter[j][index] < 3:
                                        self.last_colors[j][idx] = (0, 255, 0)
                                    else:
                                        self.skip_list[j].append(idx)
                                    continue
                                else:
                                    self.green_box_counter[j][index] = 0
                                    self.name_time_pause[j][index] = datetime.min
                    self.name_last_frame[j] = list_index
            w = self.canvas_facerecog_in.winfo_width()
            h = self.canvas_facerecog_in.winfo_height()
            frame_in_pil = self.fr.draw_frame_pil(frame_in, self.last_boxes[self.cam_in_idx],
                                                  self.last_texts[self.cam_in_idx],
                                                  color=self.last_colors[self.cam_in_idx], thick=4,
                                                  skip_list=self.skip_list[self.cam_in_idx])
            self.photo_in = ImageTk.PhotoImage(image=self.resize_with_pad(frame_in_pil, (w, h)))
            self.canvas_facerecog_in.create_image(0, 0, image=self.photo_in, anchor='nw')
            frame_out_pil = self.fr.draw_frame_pil(frame_out, self.last_boxes[self.cam_out_idx],
                                                   self.last_texts[self.cam_out_idx],
                                                   color=self.last_colors[self.cam_out_idx], thick=4,
                                                   skip_list=self.skip_list[self.cam_out_idx])
            self.photo_out = ImageTk.PhotoImage(image=self.resize_with_pad(frame_out_pil, (w, h)))
            self.canvas_facerecog_out.create_image(0, 0, image=self.photo_out, anchor='nw')
        self.root.after(20, self.update_frame_facerecog)

    # Update frame at Add face
    def update_frame_addface(self):
        if self.stop_update_frame_addface:
            return
        isTrue, frame = self.vids[self.cam_capture_idx].get_frame()
        if isTrue:
            if self.is_record:
                self.image_buffer.append(frame.copy())
                self.record_count += 1
                text = '{}/{}'.format(self.record_count, self.record_total)
                self.write(text)
                cv2.putText(frame, text,
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                time.sleep(0.5)
            w = self.canvas_addface.winfo_width()
            h = self.canvas_addface.winfo_height()
            self.photo_in = ImageTk.PhotoImage(image=self.resize_with_pad(ImagePIL.fromarray(frame), (w, h)))
            self.canvas_addface.create_image(0, 0, image=self.photo_in, anchor='nw')
            if self.record_count >= self.record_total:
                self.is_record = False
                self.record_count = 0
                self.popup_input_frame(mode='add_new')
                self.enable_buttons()
        else:
            if self.is_record:
                self.is_record = False
                self.record_count = 0
                self.popup_input_frame(mode='add_new')
                self.enable_buttons()
        self.root.after(20, self.update_frame_addface)

    # Print to console text box
    def write(self, *message, end="\n", sep=" ", out='console'):
        text = ""
        for item in message:
            text += "{}".format(item)
            text += sep
        text += end
        if out == 'console':
            self.txt_console.configure(state='normal')
            self.txt_console.insert(END, text)
            self.txt_console.see("end")
            self.txt_console.configure(state='disable')
        if out == 'notify':
            self.txt_notify.configure(state='normal')
            self.txt_notify.insert(END, text)
            self.txt_notify.see("end")
            self.txt_notify.configure(state='disable')

    # Save image to folder
    def save_image(self):
        if self.image_folder_name == '':
            return
        self.get_employee_from_db()
        emp_id = self.image_folder_name.split('_ID_')[1]
        emp_name = self.image_folder_name.split('_ID_')[0]
        if emp_id not in self.emp_id_list:
            self.add_employee(emp_name, emp_id, self.emp_notify_lang)
        image_folder = os.path.join(self.data_folder, self.image_folder_name)
        self.write(languages[self.lang]['msg_save_image'] + image_folder)
        if self.image_folder_name not in self.modified_folder:
            self.modified_folder.append(self.image_folder_name)
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        for i, image in enumerate(self.image_buffer):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_path = image_folder + '/{}_{}_{}.jpg'.format(emp_id, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), i)
            is_success, im_buf_arr = cv2.imencode(".jpg", image)
            im_buf_arr.tofile(image_path)
        thread = threading.Thread(target=self.create_images_with_masks, args=(image_folder,), daemon=True)
        thread.start()

    # Create image with masks
    def create_images_with_masks(self, image_folder):
        try:
            filenames = glob.glob(image_folder + '/*.jpg')
            images = [cv2.imdecode(np.fromfile(img, np.uint8), cv2.IMREAD_UNCHANGED) for img in filenames]
            masks_filenames = glob.glob('images/masks' + '/*.png')
            masks = [cv2.imread(mask_img) for mask_img in masks_filenames]
            self.write(languages[self.lang]['msg_gen_face_mask'])
            for idx, img in enumerate(images):
                try:
                    out_path = '{}/face_with_mask_{}.png'.format(image_folder, idx)
                    if os.path.exists(out_path):
                        continue
                    out_img = add_face_mask(img, masks[idx % len(masks)])
                    is_success, im_buf_arr = cv2.imencode(".png", out_img)
                    if is_success:
                        im_buf_arr.tofile(out_path)
                    self.write('\t{}/{}'.format(idx + 1, len(images)))
                except Exception as ex:
                    self.write('\t{}/{} ERROR face out side of image'.format(idx + 1, len(images)))
                    os.remove(filenames[idx])
                    pass
            self.write(languages[self.lang]['msg_done'])
        except Exception as ex:
            self.write('ERROR ' + str(ex))
            return

    def get_person_name(self, name, name_id, notify_lang):
        if name == '' or name_id == '':
            self.image_folder_name = ''
            self.emp_notify_lang = ''
            self.write(languages[self.lang]['err_name_empty'])
            return
        self.get_employee_from_db()
        try:
            index = self.emp_id_list.index(name_id)
            if name != self.emp_name_list[index]:
                self.image_folder_name = ''
                self.write('ERROR: ID {} already exist with name: {}'.format(name_id, self.emp_name_list[index]))
                return
        except ValueError:
            pass
        self.image_folder_name = name + '_ID_' + name_id
        self.emp_notify_lang = notify_lang

    # Stop face recognition engine
    def stop_face_recog(self):
        if self.is_running_fr:
            self.fr.stop_thread_face_recog()
            self.is_running_fr = False

    # Start face recognition engine
    def start_face_recog(self):
        if not self.is_running_fr:
            self.fr.stop_flag[0] = False
            self.is_running_fr = True
            thread1 = threading.Thread(target=self.fr.thread_face_recog, args=(self.vids[0].cap, 0), daemon=True)
            thread1.start()
            if self.cam_num == 2:
                thread2 = threading.Thread(target=self.fr.thread_face_recog, args=(self.vids[1].cap, 1), daemon=True)
                thread2.start()

    @staticmethod
    def get_thumbnail(image, box, size=None):
        try:
            if image is None:
                return
            x1 = int(box[0])
            x2 = int(box[2])
            y1 = int(box[1])
            y2 = int(box[3])
            crop = image[y1:y2, x1:x2]
            if size is not None:
                crop = cv2.resize(crop, size)
            return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        except:
            return

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
    def update_attendant(name, dt, stt):
        n = name.split('_ID_')[0]
        employee_id = name.split('_ID_')[1]
        record = Attendance(name=n, employeeID=employee_id, date=dt.date(), time=dt.time().strftime('%H:%M:%S'),
                            status=stt)
        session.add(record)
        session.commit()

    # Calculate and Update Timekeeping to database
    def update_timekeeping(self, name, dt, status):
        date_str = dt.date().strftime('%Y-%m-%d')
        time_str = dt.time().strftime('%H:%M')
        emp_id = name.split('_ID_')[1]
        emp_name = name.split('_ID_')[0]
        lst = []
        with engine.connect() as con:
            rs = con.execute('''SELECT checkOut, last, work, rest, OT FROM Timekeeping
                            WHERE employeeID = '{0}'
                            AND date = '{1}' '''.format(emp_id, date_str))
            for row in rs:
                lst.append(row)
        if len(lst) == 0:
            if status == 'in':
                record = Timekeeping(name=emp_name, employeeID=emp_id, date=dt.date(),
                                     checkIn=dt.time().strftime('%H:%M:%S'),
                                     last='{}_{}'.format(time_str, status), work=0, rest=0, OT=0)
                session.add(record)
                session.commit()
            if status == 'out':
                record = Timekeeping(name=emp_name, employeeID=emp_id, date=dt.date(),
                                     checkIn=dt.time().strftime('%H:%M:%S'),
                                     checkOut=dt.time().strftime('%H:%M:%S'),
                                     last='{}_{}'.format(time_str, status), work=0, rest=0, OT=0)
                session.add(record)
                session.commit()
        else:
            result = lst[0]
            checkOut = result[0]
            last = result[1]
            last_str = result[1].split('_')[0]
            last_stt = result[1].split('_')[1]
            work = result[2]
            rest = result[3]
            OT = result[4]
            last_time = datetime.strptime(last_str, '%H:%M')
            last_delta = timedelta(hours=last_time.hour, minutes=last_time.minute)
            now_delta = timedelta(hours=dt.time().hour, minutes=dt.time().minute)
            break_start_delta = timedelta(hours=self.end_hr_1.hour, minutes=self.end_hr_1.minute)
            break_end_delta = timedelta(hours=self.start_hr_2.hour, minutes=self.start_hr_2.minute)
            end_delta = timedelta(hours=self.end_hr_2.hour, minutes=self.end_hr_2.minute)
            if last_delta < break_start_delta < now_delta <= break_end_delta:
                diff_delta = break_start_delta - last_delta
            elif last_delta < break_start_delta < break_end_delta < now_delta:
                diff_delta = now_delta - last_delta - (break_end_delta - break_start_delta)
            elif break_start_delta <= last_delta < break_end_delta < now_delta:
                diff_delta = now_delta - break_end_delta
            elif break_start_delta <= last_delta < now_delta <= break_end_delta:
                diff_delta = timedelta(minutes=0)
            else:
                diff_delta = now_delta - last_delta

            if last_delta <= end_delta < now_delta:
                ot_delta = now_delta - end_delta
                diff_delta -= ot_delta
            elif end_delta < last_delta <= now_delta:
                ot_delta = now_delta - last_delta
                diff_delta -= ot_delta
            else:
                ot_delta = timedelta(minutes=0)
            if status == 'in':
                if last_stt == 'out':
                    rest += int(diff_delta.total_seconds() / 60)
                if last_stt == 'in':
                    OT += int(ot_delta.total_seconds() / 60)
                    work += int(diff_delta.total_seconds() / 60)
                last = '{}_{}'.format(time_str, status)

            elif status == 'out':
                work += diff_delta.total_seconds() / 60
                OT += int(ot_delta.total_seconds() / 60)
                last = '{}_{}'.format(time_str, status)
                checkOut = dt.time().strftime('%H:%M:%S')

            with engine.connect() as con:
                rs = con.execute('''UPDATE Timekeeping
                                SET checkOut='{0}', last='{1}', work='{2}', rest='{3}', OT='{4}'
                                WHERE employeeID = '{5}' AND date = '{6}' 
                                '''.format(checkOut, last, work, rest, OT, emp_id, date_str))

    @staticmethod
    def add_employee(emp_name, emp_id, emp_notify_lang):
        if emp_notify_lang == '':
            record = Employee(name=emp_name, id=emp_id)
        else:
            record = Employee(name=emp_name, id=emp_id, language=emp_notify_lang)
        session.add(record)
        session.commit()

    def update_employee(self, emp_name, emp_id, emp_notify_lang):
        if emp_name == '':
            return
        try:
            with engine.connect() as con:
                con.execute('''UPDATE Employee SET language = '{0}', name = '{1}' WHERE id = '{2}' '''.format(emp_notify_lang, emp_name, emp_id))
                con.execute('''UPDATE Timekeeping SET name = '{0}' WHERE employeeID = '{1}' '''.format(emp_name, emp_id))
                con.execute('''UPDATE Attendance SET name = '{0}' WHERE employeeID = '{1}' '''.format(emp_name, emp_id))
            self.fr.change_name(emp_id=emp_id, new_name=emp_name)
            self.fc.change_name(emp_id=emp_id, new_name=emp_name)
            basedir = self.data_folder
            for fn in os.listdir(basedir):
                if not os.path.isdir(os.path.join(basedir, fn)):
                    continue  # Not a directory
                if fn.split('_ID_')[1] == emp_id:
                    os.rename(os.path.join(basedir, fn),
                              os.path.join(basedir, emp_name + '_ID_' + emp_id))

        except Exception:
            self.write('ERROR: Change name failed')
            return
        self.write('Change name successful')

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

    def get_employee_from_db(self):
        self.emp_name_list.clear()
        self.emp_id_list.clear()
        with engine.connect() as con:
            rs = con.execute('''SELECT DISTINCT id, name From Employee''')
            for row in rs:
                self.emp_name_list.append(list(row)[1])
                self.emp_id_list.append(list(row)[0])

    def speak(self, name, stt):
        emp_id = name.split('_ID_')[1]
        notify_lang = 'VN'
        text_to_speak = ''
        with self.speak_lock:
            with engine.connect() as con:
                rs = con.execute('''SELECT language From Employee WHERE id = '{}' '''.format(emp_id))
                for row in rs:
                    if len(list(row)) > 0:
                        notify_lang = list(row)[0]
            if notify_lang == 'EN':
                tts_engine.setProperty('voice', voices[1].id)
                tts_engine.setProperty('volume', 1)
            if notify_lang == 'VN':
                tts_engine.setProperty('voice', voices[2].id)
                tts_engine.setProperty('volume', 0.5)
            tts_engine.setProperty('rate', 200)
            if stt == 'out':
                text_to_speak = '{} {}'.format(languages[notify_lang]['tts_goodbye'], name.split('_ID_')[0])
            elif stt == 'in':
                text_to_speak = '{} {}'.format(languages[notify_lang]['tts_hello'], name.split('_ID_')[0])
            tts_engine.say(text_to_speak)
            tts_engine.runAndWait()

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        if messagebox.askokcancel(languages[self.lang]['popup_quit'], languages[self.lang]['popup_quit_confirm']):
            self.root.destroy()


if __name__ == '__main__':
    if len(device_list) < 2:
        print('Error: Not enough camera')
        time.sleep(5)
        exit()
    app = FaceAttendanceApp()
    app.run()
