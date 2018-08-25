#library to work with buttons and ui.c via buttons.msg file
import struct
from ctypes import create_string_buffer
import os
import pickle
from datetime import datetime
import threading
import time


class UIButton:
    def __init__(self, btn_name, btn_label, btn_status, btn_label2, btn_index):
        self.btn_name = btn_name
        self.btn_label = btn_label
        self.btn_label2 = btn_label2
        self.btn_status = btn_status
        self.btn_index = btn_index


class UIButtons:
    def write_buttons_out_file(self):
        """ Write button file in a non-blocking manner.
        The button file is used to resume settings after a reboot. It is not
        time sensitive or critical. So do these writes in a separate thread,
        allowing the main control loop to continue unblocked.
    
        """
        thread = threading.Thread(target=self._write_buttons_blocking,
                                  args=(time.time()))
        thread.start()
            
    def _write_buttons_blocking(self, nonce):
        try:
            with self.file_lock:
                if nonce > self.file_nonce:
                    with open(self.buttons_status_out_path, "wb") as fo:
                        pickle.dump(self.btns, fo)
                    self.file_nonce = nonce
        except Exception as e:
            print "Failed to write button file %s" % self.buttons_status_out_path
            print str(e)

    def read_buttons_out_file(self):
        if os.path.exists(self.buttons_status_out_path):
            try:
                with self.file_lock:
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
        # TODO: a rwlock would perform better (allowing parallel reads)
        self.file_lock = threading.Lock()
        # An ever-increasing count to ensure writes don't happen out of order.
        self.file_nonce = -1

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

    def set_button_status(self, btn_name, btn_status):
        btn = self.get_button(btn_name)
        if btn and btn.btn_status != btn_status:
            btn.btn_status = btn_status
            self.CS.UE.uiButtonInfoEvent(btn.btn_index,
                                         btn.btn_name,
                                         btn.btn_label,
                                         btn.btn_status,
                                         btn.btn_label2)
            self.write_buttons_out_file()

    def set_button_status_from_ui(self, id, btn_status):
        if self.btns[id].btn_status != btn_status:
            self.CS.update_ui_buttons(id, btn_status)
            self.CS.UE.uiButtonInfoEvent(id,
                                         self.btns[id].btn_name,
                                         self.btns[id].btn_label,
                                         self.btns[id].btn_status,
                                         self.btns[id].btn_label2)
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
