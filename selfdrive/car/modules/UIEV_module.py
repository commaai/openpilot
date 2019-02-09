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
        self.uiCustomAlert = messaging.pub_sock(context, service_list['uiCustomAlert'].port)
        self.uiButtonInfo = messaging.pub_sock(context, service_list['uiButtonInfo'].port)
        self.uiSetCar = messaging.pub_sock(context, service_list['uiSetCar'].port)
        self.uiPlaySound = messaging.pub_sock(context, service_list['uiPlaySound'].port)
        self.uiButtonStatus = messaging.sub_sock(context, service_list['uiButtonStatus'].port, conflate=True, poller=self.buttons_poller)
        self.prev_cstm_message = ""
        self.prev_cstm_status = -1

    def uiCustomAlertEvent(self,status,message):
        dat = ui.UICustomAlert.new_message()
        dat.caStatus = status
        dat.caText = message+'\0'
        self.uiCustomAlert.send(dat.to_bytes())
    
    def uiButtonInfoEvent(self,btnid,name,label,status,label2):
        dat = ui.UIButtonInfo.new_message()
        dat.btnId = btnid
        dat.btnName = name #+ '\0'
        dat.btnLabel = label #+ '\0'
        dat.btnStatus = status
        dat.btnLabel2 = label2 #+ '\0'
        self.uiButtonInfo.send(dat.to_bytes())
    
    def uiSetCarEvent(self,car_folder,car_name):
        dat = ui.UISetCar.new_message()
        dat.icCarFolder = car_folder
        dat.icCarName = car_name
        self.uiSetCar.send(dat.to_bytes())

    def uiPlaySoundEvent(self,sound):
        if self.CS.cstm_btns.get_button_status("sound") > 0:
            dat = ui.UIPlaySound.new_message()
            dat.sndSound = sound
            self.uiPlaySound.send(dat.to_bytes())

    # for status we will use one of these values
    # NO_STATUS_ALTERATION -1
    # STATUS_STOPPED 0
    # STATUS_DISENGAGED 1
    # STATUS_ENGAGED 2
    # STATUS_WARNING 3
    # STATUS_ALERT 4
    # STATUS_MAX 5

    #for sound we will use one of these values
    # NO_SOUND -1
    # disable.wav 1
    # enable.wav 2
    # info.wav 3
    # attention.wav 4
    # error.wav 5

    def custom_alert_message(self,status,message,duration,sound=-1):
        if (status > -1) and (self.prev_cstm_status > status) and \
          (self.CS.custom_alert_counter > 55):
          #dont change lessage to a lower importance one if we still have more than half second of display time
          return
        if (sound > -1) and ((self.prev_cstm_message != message) or (self.prev_cstm_status != status)):
            self.uiPlaySoundEvent(sound)
        self.uiCustomAlertEvent(status,message)
        self.CS.custom_alert_counter = duration
        self.prev_cstm_message = message
        self.prev_cstm_status = status

    def update_custom_ui(self):
        btn_message = None
        for socket, event in self.buttons_poller.poll(0):
            if socket is self.uiButtonStatus:
                btn_message = ui.UIButtonStatus.from_bytes(socket.recv())
        if btn_message is not None:
            btn_id = btn_message.btnId
            self.CS.cstm_btns.set_button_status_from_ui(btn_id,btn_message.btnStatus)
        if (self.CS.custom_alert_counter > 0):
            self.CS.custom_alert_counter -= 1
            if (self.CS.custom_alert_counter ==0):
                self.custom_alert_message(-1,"",0)
                self.CS.custom_alert_counter = -1
