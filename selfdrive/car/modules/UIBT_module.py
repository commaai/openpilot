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
    def __init__(self,btn_name,btn_label,btn_status,btn_label2,btn_index):
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
        file_matches = False
        if len(indata) == btn_msg_len * 6 :
            for i in range(0, len(indata), btn_msg_len):
                j = int(i/btn_msg_len)
                name,label,label2 = struct.unpack(btn_msg_struct, indata[i:i+btn_msg_len]) 
                if self.btns[j].btn_name == name.rstrip("\0"):
                    file_matches = True
                    self.btns[j].btn_label = label.rstrip("\0")
                    #check if label is actually a valid option
                    if label2.rstrip("\0") in self.CS.btns_init[j][2]:
                        self.btns[j].btn_label2 = label2.rstrip("\0")
                    else:
                        self.btns[j].btn_label2 = self.CS.btns_init[j][2][0]
            return file_matches
        else:
            #we don't have all the data, ignore
            print "labels file is bad"
            return False


    def write_buttons_out_file(self):
        if self.hasChanges:
            fo = open(self.buttons_status_out_path, buttons_file_rw)
            for btn in self.btns:
                btn_status = 1 if btn.btn_status > 0 else 0
                fo.write(struct.pack("B",btn_status + 48))
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
            self.init_ui_buttons()
            #there is a file, load it
            if self.read_buttons_labels_from_file():
                self.read_buttons_out_file()
            else:
                #no match, so write the new ones
                self.hasChanges = True
                self.write_buttons_labels_to_file()
                self.write_buttons_out_file()
        else:
            #there is no file, create it
            self.init_ui_buttons()
            self.hasChanges = True
            self.write_buttons_labels_to_file()
            self.write_buttons_out_file()
        #send events to initiate UI
        self.isLive = True
        self.send_button_info()
        self.CS.UE.uiSetCarEvent(self.car_folder,self.car_name)

    def init_ui_buttons(self):
        self.btns = []
        try:
            self.CS.init_ui_buttons()
            print "Buttons iniatlized with custom CS code"  
        except AttributeError:
            # no init method
            print "Buttons iniatlized with just base code"
        for i in range(0,len(self.CS.btns_init)):
            self.btns.append(UIButton(self.CS.btns_init[i][0],self.CS.btns_init[i][1],1,self.CS.btns_init[i][2][0],i))

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

    def set_button_status(self,btn_name,btn_status):
        btn = self.get_button(btn_name)
        if btn:
            #if we change from enable to disable or the other way, save to file
            if btn.btn_status * btn_status == 0 and btn.btn_status != btn_status:
                self.hasChanges = True
            btn.btn_status = btn_status
            self.CS.UE.uiButtonInfoEvent(self.btns.index(btn),btn.btn_name, \
                btn.btn_label,btn.btn_status,btn.btn_label2)
        if self.hasChanges:
            self.write_buttons_out_file()

    def set_button_status_from_ui(self,id,btn_status):
        old_btn_status = self.btns[id].btn_status
        old_btn_label2 = self.btns[id].btn_label2
        if old_btn_status * btn_status == 0 and old_btn_status != btn_status:
            self.hasChanges = True
        self.update_ui_buttons(id,btn_status)
        new_btn_status = self.btns[id].btn_status
        if new_btn_status * btn_status == 0 and new_btn_status != btn_status:
            self.hasChanges = True
        if self.hasChanges:
            self.CS.UE.uiButtonInfoEvent(id,self.btns[id].btn_name, \
                    self.btns[id].btn_label,self.btns[id].btn_status,self.btns[id].btn_label2)
            if old_btn_label2 != self.btns[id].btn_label2:
                self.write_buttons_labels_to_file()
            self.write_buttons_out_file()
        

    def get_button_label2(self, btn_name):
        btn = self.get_button(btn_name)
        if btn:
            return btn.btn_label2
        else:
            return -1    

    def get_button_label2_index(self, btn_name):
        btn = self.get_button(btn_name)
        if btn:
            b_index = -1
            bpos = self.btns.index(btn)
            for i in range(0,len(self.CS.btns_init[bpos][2])):
                if btn.btn_label2 == self.CS.btns_init[bpos][2][i]:
                    b_index = i
            return b_index
        else:
            return -1
    
    def update_ui_buttons(self,id,btn_status):
        if self.CS.cstm_btns.btns[id].btn_status > 0:
        #if we have more than one Label2 then we toggle
            if (len(self.CS.btns_init[id][2]) > 1):
                status2 = 0
                for i in range(0,len(self.CS.btns_init[id][2])):
                    if self.CS.cstm_btns.btns[id].btn_label2 == self.CS.btns_init[id][2][i]:
                        status2 = (i + 1 ) % len(self.CS.btns_init[id][2])
                self.CS.cstm_btns.btns[id].btn_label2 = self.CS.btns_init[id][2][status2]
                self.CS.cstm_btns.hasChanges = True
            else:
                self.CS.cstm_btns.btns[id].btn_status = btn_status * self.CS.cstm_btns.btns[id].btn_status
        else:
            self.CS.cstm_btns.btns[id].btn_status = btn_status
