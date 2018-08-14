from cereal import ui
from common import realtime
import selfdrive.messaging as messaging
from selfdrive.services import service_list
import zmq

class UIEvents(object):
    def __init__(self,carstate):
        self.CS = carstate
        context = zmq.Context()
        self.buttons_poller = zmq.Poller()
        self.uiCustomAlert = messaging.sub_sock(context, service_list['uiCustomAlert'].port, conflate=True, poller=None)
        self.uiButtonInfo = messaging.sub_sock(context, service_list['uiButtonInfo'].port, conflate=True, poller=None)
        self.uiSetCar = messaging.sub_sock(context, service_list['uiSetCar'].port, conflate=True, poller=None)
        self.uiButtonStatus = messaging.sub_sock(context, service_list['uiButtonStatus'].port, conflate=True, poller=self.buttons_poller)

    def uiCustomAlertEvent(self,status,message):
        dat = ui.UIEvent.new_message()
        dat.logMonoTime = int(realtime.sec_since_boot() * 1e9)
        dat.init('uiCustomAlert')
        dat.uiCustomAlert = {
            "ca_status": status,
            "ca_text": message
        }
        self.uiCustomAlert.send(dat.to_bytes())
    
    def uiButtonInfoEvent(self,id,name,label,status,label2):
        dat = ui.UIEvent.new_message()
        dat.logMonoTime = int(realtime.sec_since_boot() * 1e9)
        dat.init('uiButtonInfo')
        dat.uiButtonInfo = {
            "btn_id": id,
            "btn_name": name,
            "btn_label": label,
            "btn_status": status,
            "btn_label2": label2
        }
        self.uiButtonInfo.send(dat.to_bytes())
    
    def uiSetCarEvent(self,car_folder,car_name):
        dat = ui.UIEvent.new_message()
        dat.logMonoTime = int(realtime.sec_since_boot() * 1e9)
        dat.init('UISetCar')
        dat.UISetCar = {
            "ic_carFolder": car_folder,
            "ic_carName": car_name
        }
        self.uiSetCar.send(dat.to_bytes())
    
    def custom_alert_message(self,message,duration):
        self.uiCustomAlertEvent(1,message)
        self.CS.custom_alert_counter = duration

    def update_custom_ui(self):
        btn_message = None
        for socket, event in self.buttons_poller.poll(0):
            if socket is self.uiButtonStatus:
                btn_message = messaging.recv_one(socket)
        if btn_message is not None:
            btn_id = btn_message.uiButtonStatus.btn_id
            self.CS.cstm_btns.set_button_status_from_ui(btn_id,btn_message.uiButtonStatus.btn_status)
        if (self.CS.custom_alert_counter > 0):
            self.CS.custom_alert_counter -= 1
            if (self.CS.custom_alert_counter ==0):
                self.custom_alert_message("",0)
                self.CS.custom_alert_counter = -1