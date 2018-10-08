#library to work with buttons and ui.c via buttons.msg file
import struct
from ctypes import create_string_buffer
import os
import pickle
from datetime import datetime


class UIButton:
    def __init__(self, btn_name, btn_label, btn_status, btn_label2, btn_index):
        self.btn_name = btn_name
        self.btn_label = btn_label
        self.btn_label2 = btn_label2
        self.btn_status = btn_status
        self.btn_index = btn_index


class UIButtons:
    def write_buttons_out_file(self):
        if self.hasChanges:
            fo = open(self.buttons_status_out_path, buttons_file_rw)
            for btn in self.btns:
                btn_status = 1 if btn.btn_status > 0 else 0
                fo.write(struct.pack("B",btn_status + 48))
            fo.close()
        self.hasChanges = False

    def read_buttons_out_file(self):
        if os.path.exists(self.buttons_status_out_path):
            try:
                with open(self.buttons_status_out_path, "rb") as fo:
                    self.btns = pickle.load(fo)
                    self.btn_map = self._map_buttons(self.btns)
                return True
            except Exception as e:
                print "Failed to read button file %s" % self.buttons_status_out_path
                print str(e)
        return False

    def send_button_info(self):
        if self.isLive:
            for btn in self.btns:
                self.CS.UE.uiButtonInfoEvent(btn.btn_index,
                                             btn.btn_name,
                                             btn.btn_label,
                                             btn.btn_status,
                                             btn.btn_label2)

    def __init__(self, carstate, car, folder):
        self.isLive = False
        self.CS = carstate
        self.car_folder = folder
        self.car_name = car
        self.buttons_status_out_path = "/data/openpilot/selfdrive/car/"+self.car_folder+"/buttons.pickle"
        self.btns = []
        self.btn_map = {}
        self.hasChanges = False
        self.last_in_read_time = datetime.min 
        if not self.read_buttons_out_file():
            # there is no file, create it
            self.btns = self.CS.init_ui_buttons()
            self.btn_map = self._map_buttons(self.btns)
            self.write_buttons_out_file()
        # send events to initiate UI
        self.isLive = True
        self.send_button_info()
        self.CS.UE.uiSetCarEvent(self.car_folder, self.car_name)

    def get_button(self, btn_name):
        if btn_name in self.btn_map:
            return self.btn_map[btn_name]
        else:
            return None

    def get_button_status(self, btn_name):
        if btn_name in self.btn_map:
            return self.btn_map[btn_name].btn_status
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
        if old_btn_status * btn_status == 0 and old_btn_status != btn_status:
            self.hasChanges = True
        self.CS.update_ui_buttons(id,btn_status)
        new_btn_status = self.btns[id].btn_status
        if new_btn_status * btn_status == 0 and new_btn_status != btn_status:
            self.hasChanges = True
        if self.hasChanges:
            self.CS.UE.uiButtonInfoEvent(id,self.btns[id].btn_name, \
                    self.btns[id].btn_label,self.btns[id].btn_status,self.btns[id].btn_label2)
            self.write_buttons_out_file()
        

    def get_button_label2(self, btn_name):
        if btn_name in self.btn_map:
            return self.btn_map[btn_name].btn_label2
        else:
            return -1
            
    # Convert the button list to a map, keyed based on btn_name. Allows o(1)
    # lookup time for buttons based on name.
    def _map_buttons(self, btn_list):
        btn_map = {}
        for btn in btn_list:
            btn_map[btn.btn_name] = btn
        return btn_map
