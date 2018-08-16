#library to work with buttons and ui.c via buttons.msg file
import struct
from ctypes import create_string_buffer
import os
from datetime import datetime


buttons_file_rw = "wb"
buttons_file_r = "rb"
btn_msg_len = 23
btn_msg_struct = "6s6s11s" #name=5 char string, label = 5 char string, satus = 1 char, label2 = 10 char string

class UIButton:
    def __init__(self,btn_name,btn_label,btn_status,btn_label2):
        self.btn_name = btn_name
        self.btn_label = btn_label
        self.btn_label2 = btn_label2
        self.btn_status = btn_status


class UIButtons:
    def write_buttons_labels_to_file(self):
        fo = open(self.buttons_labels_path, buttons_file_rw)
        for btn in self.btns:
            fo.write(struct.pack(btn_msg_struct,btn.btn_name,btn.btn_label,btn.btn_label2))
        fo.close()

    def read_buttons_labels_from_file(self):
        fi =  open(self.buttons_labels_path, buttons_file_r)
        indata = fi.read()
        fi.close()
        if len(indata) == btn_msg_len * 6 :
            #we have all the data
            self.btns = []
            for i in range(0, len(indata), btn_msg_len):
                name,label,label2 = struct.unpack(btn_msg_struct, indata[i:i+btn_msg_len])  
                self.btns.append(UIButton(name.rstrip("\0"),label.rstrip("\0"),0,label2.rstrip("\0")))
            #now read the last saved statuses
        else:
            #we don't have all the data, ignore
            print "labels file is bad"


    def write_buttons_out_file(self):
        if self.hasChanges:
            fo = open(self.buttons_status_out_path, buttons_file_rw)
            for btn in self.btns:
                fo.write(struct.pack("B",btn.btn_status + 48))
            fo.close()
        self.hasChanges = False

    def read_buttons_out_file(self):
        fi =  open(self.buttons_status_out_path, buttons_file_r)
        indata = fi.read()
        fi.close()
        if len(indata) == 6:
            for i in range(0,len(indata)):
                self.btns[i].btn_status = ord(indata[i]) - 48
        else:
            #something wrong with the file
            print "status file is bad"

    def send_button_info(self):
        if self.isLive:
            for i in range(0,6):
                self.CS.UE.uiButtonInfoEvent(i,self.btns[i].btn_name, \
                    self.btns[i].btn_label,self.btns[i].btn_status,self.btns[i].btn_label2)

    def __init__(self, carstate,car,folder):
        self.isLive = False
        self.CS = carstate
        self.car_folder = folder
        self.car_name = car
        self.buttons_labels_path = "/data/openpilot/selfdrive/car/"+self.car_folder+"/buttons.msg"
        self.buttons_status_out_path = "/data/openpilot/selfdrive/car/"+self.car_folder+"/buttons.cc.msg"
        self.btns = []
        self.hasChanges = True
        self.last_in_read_time = datetime.min 
        if os.path.exists(self.buttons_labels_path):
            #there is a file, load it
            self.read_buttons_labels_from_file()
            self.read_buttons_out_file()
        else:
            #there is no file, create it
            self.CS.init_ui_buttons()
            self.write_buttons_labels_to_file()
            self.write_buttons_out_file()
        #send events to initiate UI
        self.isLive = True
        self.send_button_info()
        self.CS.UE.uiSetCarEvent(self.car_folder,self.car_name)


    def get_button_status(self,btn_name):
        ret_val =-1 
        for i in range(0,6):
            if self.btns[i].btn_name.strip() == btn_name:
                ret_val = self.btns[i].btn_status
        return ret_val

    def set_button_status(self,btn_name,btn_status):
        for i in range(0,6):
            if self.btns[i].btn_name.strip() == btn_name:
                self.btns[i].btn_status = btn_status
                self.hasChanges = True
                self.CS.UE.uiButtonInfoEvent(i,self.btns[i].btn_name, \
                    self.btns[i].btn_label,self.btns[i].btn_status,self.btns[i].btn_label2)
        self.write_buttons_out_file()

    def set_button_status_from_ui(self,id,btn_status):
        self.CS.update_ui_buttons(id,btn_status)
        self.CS.UE.uiButtonInfoEvent(id,self.btns[id].btn_name, \
                    self.btns[id].btn_label,self.btns[id].btn_status,self.btns[id].btn_label2)
        self.hasChanges = True
        self.write_buttons_out_file()
        

    
