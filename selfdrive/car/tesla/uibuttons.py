#library to work with buttons and ui.c via buttons.msg file
import struct
from ctypes import create_string_buffer
import os
from datetime import datetime

buttons_labels_path = "/data/openpilot/selfdrive/car/tesla/buttons.msg"
buttons_status_in_path = "/data/openpilot/selfdrive/car/tesla/buttons.ui.msg"
buttons_status_out_path = "/data/openpilot/selfdrive/car/tesla/buttons.cc.msg"
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
        fo = open(buttons_labels_path, buttons_file_rw)
        for btn in self.btns:
            fo.write(struct.pack(btn_msg_struct,btn.btn_name,btn.btn_label,btn.btn_label2))
        fo.close()

    def read_buttons_labels_from_file(self):
        fi =  open(buttons_labels_path, buttons_file_r)
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

    def read_buttons_in_file(self):
        modification_date =  datetime.fromtimestamp(os.path.getmtime(buttons_status_in_path))
        if (modification_date > self.last_in_read_time):
            fi =  open(buttons_status_in_path, buttons_file_r)
            indata = fi.read()
            fi.close()
            if len(indata) == 6:
                for i in range(0,len(indata)):
                    if self.btns[i].btn_status > 0:
                        if (i == 1) and (ord(indata[1])==48):
                            #don't change status, just model
                            if self.btns[i].btn_label2 == "Mod OP":
                                self.btns[i].btn_label2 = "Mod JJ"
                            #elif self.btns[i].btn_label2 == "Mod JJ":
                            #    self.btns[i].btn_label2 = "AutoRes"
                            else:
                                self.btns[i].btn_label2 = "Mod OP"
                            self.write_buttons_labels_to_file()
                        else:
                            self.btns[i].btn_status = (ord(indata[i]) - 48) * self.btns[i].btn_status
                    else:
                        self.btns[i].btn_status = ord(indata[i]) - 48
                self.last_in_read_time = modification_date
                self.hasChanges = True
            else:
                #something wrong with the file
                print "status file is bad"    

    def write_buttons_in_file(self):
        fo = open(buttons_status_in_path, buttons_file_rw)
        for btn in self.btns:
            btn_val = 1 if btn.btn_status > 0 else 0
            fo.write(struct.pack("B",btn_val + 48))
        fo.close()

    def write_buttons_out_file(self):
        if self.hasChanges:
            fo = open(buttons_status_out_path, buttons_file_rw)
            for btn in self.btns:
                fo.write(struct.pack("B",btn.btn_status + 48))
            fo.close()
        self.hasChanges = False

    def read_buttons_out_file(self):
        fi =  open(buttons_status_out_path, buttons_file_r)
        indata = fi.read()
        fi.close()
        if len(indata) == 6:
            for i in range(0,len(indata)):
                self.btns[i].btn_status = ord(indata[i]) - 48
        else:
            #something wrong with the file
            print "status file is bad"


    def __init__(self):
        self.btns = []
        self.hasChanges = True
        self.last_in_read_time = datetime.min 
        if os.path.exists(buttons_labels_path):
            #there is a file, load it
            self.read_buttons_labels_from_file()
            self.read_buttons_out_file()
            self.read_buttons_in_file()
        else:
            #there is no file, create it
            self.btns.append(UIButton("alca","ALC",0,""))
            self.btns.append(UIButton("acc","ACC",0,"Mod OP"))
            self.btns.append(UIButton("pedal","PDL",0,""))
            self.btns.append(UIButton("steer","STR",0,""))
            self.btns.append(UIButton("brake","BRK",1,""))
            self.btns.append(UIButton("sound","SND",1,""))
            self.write_buttons_labels_to_file()
            self.write_buttons_in_file()
            self.write_buttons_out_file()
            
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
        
    def get_button_label2(self, btn_name):
        btn = self.get_button(btn_name)
        if btn:
            return btn.btn_label2
        else:
            return -1

    def set_button_status(self, btn_name, btn_status):
        btn = self.get_button(btn_name)
        if btn:
            btn.btn_status = btn_status
            self.hasChanges = True
        

    
