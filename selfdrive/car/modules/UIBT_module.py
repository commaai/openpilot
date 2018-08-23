#library to work with buttons and ui.c via buttons.msg file
import struct
from ctypes import create_string_buffer
import os
import pickle
from datetime import datetime


class UIButton:
    def __init__(self, btn_name, btn_label, btn_status, btn_label2):
        self.btn_name = btn_name
        self.btn_label = btn_label
        self.btn_label2 = btn_label2
        self.btn_status = btn_status


class UIButtons:
    def write_buttons_out_file(self):
        if self.hasChanges:
            try:
                with open(self.buttons_status_out_path, "wb") as fo:
                    pickle.dump(self.btns, fo)
            except Exception as e:
                print "Failed to write button file %s" % self.buttons_status_out_path
                print str(e)
        self.hasChanges = False

    def read_buttons_out_file(self):
        if os.path.exists(self.buttons_status_out_path):
            try:
                with open(self.buttons_status_out_path, "rb") as fo:
                    self.btns = pickle.load(fo)
                return True
            except Exception as e:
                print "Failed to read button file %s" % self.buttons_status_out_path
                print str(e)
        return False

    def send_button_info(self):
        if self.isLive:
            for i in range(0,6):
                self.CS.UE.uiButtonInfoEvent(i,
                                             self.btns[i].btn_name,
                                             self.btns[i].btn_label,
                                             self.btns[i].btn_status,
                                             self.btns[i].btn_label2)

    def __init__(self, carstate, car, folder):
        self.isLive = False
        self.CS = carstate
        self.car_folder = folder
        self.car_name = car
        self.buttons_status_out_path = "/data/openpilot/selfdrive/car/"+self.car_folder+"/buttons.pickle"
        self.btns = []
        self.hasChanges = True
        self.last_in_read_time = datetime.min 
        if not self.read_buttons_out_file():
            #there is no file, create it
            self.btns = self.CS.init_ui_buttons()
            self.hasChanges = True
            self.write_buttons_out_file()
        #send events to initiate UI
        self.isLive = True
        self.send_button_info()
        self.CS.UE.uiSetCarEvent(self.car_folder, self.car_name)

    def get_button(self, btn_name):
        for button in self.btns:
            if button.btn_name.strip() == btn_name:
                return button
        return None

    def get_button_status(self, btn_name):
        btn = self.get_button(btn_name)
        if btn:
            return btn.btn_status
        else:
            return -1


    def set_button_status(self, btn_name, btn_status):
        btn = self.get_button(btn_name)
        if btn:
            btn.btn_status = btn_status
            self.hasChanges = True
            self.CS.UE.uiButtonInfoEvent(self.btns.index(btn),
                                         btn.btn_name,
                                         btn.btn_label,
                                         btn.btn_status,
                                         btn.btn_label2)
        if self.hasChanges:
            self.write_buttons_out_file()
            self.hasChanges = False

    def set_button_status_from_ui(self, id,btn_status):
        self.CS.update_ui_buttons(id, btn_status)
        self.CS.UE.uiButtonInfoEvent(id,
                                     self.btns[id].btn_name,
                                     self.btns[id].btn_label,
                                     self.btns[id].btn_status,
                                     self.btns[id].btn_label2)
        self.hasChanges = True
        self.write_buttons_out_file()
        

    def get_button_label2(self, btn_name):
        btn = self.get_button(btn_name)
        if btn:
            return btn.btn_label2
        else:
            return -1    
