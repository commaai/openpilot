#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import glob
import sys
import struct
import time
import threading
import yaml
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QGroupBox, QPushButton, QMessageBox, QVBoxLayout, QHeaderView, QDoubleSpinBox,
    QSplitter, QHBoxLayout, QLabel, QScrollArea, QListWidget, QFormLayout, QCheckBox, QProgressDialog,
    QDialogButtonBox, QComboBox, QSpinBox, QDialog, QTabWidget, QLineEdit, QGridLayout,
    QTableWidget, QTableWidgetItem, QApplication, QStyle, QListWidgetItem, QRadioButton, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon
from tcp_interface import Connection
from flexray_config import (default_fr_config, verify_config, verify_config_format)
from constants import *


# FlexRay configuration form fields
# Field format: [name,
#               description,
#               (min, max) or [(opt1_name, opt1_value), (opt2_name, opt2_value)] or {True: value1, False: value2},
#               unit_name]
config_fields = {
    'Cluster': [
        ['gdMinPropagationDelay', 'gdMinPropagationDelay', (0.0, 2.5), 'µs'],
        ['gdMaxInitializationError', 'gdMaxInitializationError', (0.0, 11.7), 'µs'],
        ['gdMacrotick', 'gdMacrotick', (1, 6), 'µs'],
        ['gdTSSTransmitter', 'gdTSSTransmitter', (1, 15), 'gdBit'],
        ['gPayloadLengthStatic', 'gPayloadLengthStatic', (0, cPayloadLengthMax), 'words'],
        ['gNumberOfStaticSlots', 'gNumberOfStaticSlots', (2, cStaticSlotIDMax), ''],
        ['gdStaticSlot', 'gdStaticSlot', (3, 664), 'MT'],
        ['gdActionPointOffset', 'gdActionPointOffset', (1, 63), 'MT'],
        ['gNumberOfMinislots', 'gNumberOfMinislots', (0, 7988), ''],
        ['gdMinislot', 'gdMinislot', (2, 63), 'MT'],
        ['gdMiniSlotActionPointOffset', 'gdMiniSlotActionPointOffset', (1, 31), 'MT'],
        ['gdSymbolWindow', 'gdSymbolWindow', (0, 162), 'MT'],
        ['gOffsetCorrectionMax', 'gOffsetCorrectionMax', (0.15, 383.567), 'µs'],
        ['gOffsetCorrectionStart', 'gOffsetCorrectionStart', (7, 15999), 'MT'],
        ['gMaxWithoutClockCorrectionFatal', 'gMaxWithoutClockCorrectionFatal', (1, 15), 'pairs'],
        ['gMaxWithoutClockCorrectionPassive', 'gMaxWithoutClockCorrectionPassive', (1, 15), 'pairs'],
        ['gdNIT', 'gdNIT', (2, 15978), 'MT'],
        ['gdWakeupRxWindow', 'gdWakeupRxWindow', (76, 301), 'gdBit'],
        ['gColdStartAttempts', 'gColdStartAttempts', (2, 31), ''],
        ['gListenNoise', 'gListenNoise', (2, 16), ''],
        ['gNetworkManagementVectorLength', 'gNetworkManagementVectorLength', (0, 12), 'bytes'],
        ['gSyncFrameIDCountMax', 'gSyncFrameIDCountMax', (2, cSyncFrameIDCountMax), ''],
        ['gdCasRxLowMax', 'gdCasRxLowMax', (28, 254), 'gdBit'],
        ['gdDynamicSlotIdlePhase', 'gdDynamicSlotIdlePhase', (0, 2), 'Minislot'],
        ['gdWakeupSymbolRxIdle', 'gdWakeupSymbolRxIdle', (8, 59), 'gdBit'],
        ['gdWakeupSymbolRxLow', 'gdWakeupSymbolRxLow', (8, 59), 'gdBit'],
        ['gdWakeupSymbolTxActive', 'gdWakeupSymbolTxActive', (15, 60), 'gdBit'],
        ['gdWakeupSymbolTxIdle', 'gdWakeupSymbolTxIdle', (45, 180), 'gdBit'],
    ],
    'Node': [
        ['pChannels', 'pChannels', [('CHANNEL_A', 0),('CHANNEL_B', 1), ('CHANNEL_AB', 2)], ''],
        ['pMicroPerCycle', 'pMicroPerCycle', (960, 1280000), 'uT'],
        ['pdListenTimeout', 'pdListenTimeout', (1926, 2567692), 'uT'],
        ['pOffsetCorrectionOut', 'pOffsetCorrectionOut', (15, 16082), 'ut'],
        ['pRateCorrectionOut', 'pRateCorrectionOut', (3, 3846), 'uT'],
        ['pLatestTx', 'pLatestTx', (0, 7988), 'Minislot'],
        ['pDecodingCorrection', 'pDecodingCorrection', (12, 136), 'uT'],
        ['pDelayCompensationA', 'pDelayCompensationA', (4, 211), 'uT'],
        ['pDelayCompensationB', 'pDelayCompensationB', (4, 211), 'uT'],
        ['pMacroInitialOffsetA', 'pMacroInitialOffsetA', (2, 68), 'MT'],
        ['pMacroInitialOffsetB', 'pMacroInitialOffsetB', (2, 68), 'MT'],
        ['pMicroInitialOffsetA', 'pMicroInitialOffsetA', (0, 239), 'uT'],
        ['pMicroInitialOffsetB', 'pMicroInitialOffsetB', (0, 239), 'uT'],
        ['pWakeupChannel', 'pWakeupChannel', [('CHANNEL_A', 0),('CHANNEL_B', 1)], ''],
        ['pWakeupPattern', 'pWakeupPattern', (0, 63), ''],
        ['pPayloadLengthDynMax', 'pPayloadLengthDynMax', (0, cPayloadLengthMax), 'words'],
        ['pKeySlotId', 'pKeySlotId', (0, 1023), ''],
        ['pKeySlotOnlyEnabled', 'pKeySlotOnlyEnabled', {True: 1, False: 0}, ''],
        ['pKeySlotUsedForStartup', 'pKeySlotUsedForStartup', {True: 1, False: 0}, ''],
        ['pKeySlotUsedForSync', 'pKeySlotUsedForSync', {True: 1, False: 0}, ''],
        ['pdAcceptedStartupRange', 'pdAcceptedStartupRange', (0, 2743), 'uT'],
        ['pAllowPassiveToActive', 'pAllowPassiveToActive', (0, 31), 'cycle pairs'],
        ['pClusterDriftDamping', 'pClusterDriftDamping', (0, 10), 'uT'],
        ['pAllowHaltDueToClock', 'pAllowHaltDueToClock', {True: 1, False: 0}, ''],
        ['pdMaxDrift', 'pdMaxDrift', (2, 1923), 'uT'],
    ],
    'Board': [
        ['CLOCK_SRC', 'Clock Source', [('Crystal Oscillator', 0), ('PLL', 1)], ''],
        ['BIT_RATE', 'Bit Rate', [(10, 0), (5, 1), (2.5, 2), (8, 3)], 'Mbps'],
        ['SCM_EN', 'Single Channel Mode Enabled', {True: 1, False: 0}, ''],
        ['LOG_STATUS_DATA', 'Log protocol and sync frame status data', {True: 1, False: 0}, '']
    ],
    'Rx FIFO': [
        ['FIFOA_EN','FIFOA Enabled', {True: 1, False: 0}, ''],
        ['FIFOA_Depth', 'FIFOA Depth', (0, 255), ''],
        ['FIFOA_MIAFV', 'FIFOA Message ID Acceptance Filter Value', (0, 65535), ''],
        ['FIFOA_MIAFM', 'FIFOA Message ID Acceptance Filter Mask', (0, 65535), ''],
        ['FIFOA_F0_EN', 'FIFOA Frame ID Rangle Filter 0 Enabled', {True: 1, False: 0}, ''],
        ['FIFOA_F0_MODE', 'FIFOA Frame ID Rangle Filter 0 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOA_F0_SID_LOWER', 'FIFOA Frame ID Rangle Filter 0 Slot Id Lower', (0, 1023), ''],
        ['FIFOA_F0_SID_UPPER', 'FIFOA Frame ID Rangle Filter 0 Slot Id Upper', (0, 1023), ''],
        ['FIFOA_F1_EN', 'FIFOA Frame ID Rangle Filter 1 Enabled', {True: 1, False: 0}, ''],
        ['FIFOA_F1_MODE', 'FIFOA Frame ID Rangle Filter 1 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOA_F1_SID_LOWER', 'FIFOA Frame ID Rangle Filter 1 Slot Id Lower', (0, 1023), ''],
        ['FIFOA_F1_SID_UPPER', 'FIFOA Frame ID Rangle Filter 1 Slot Id Upper', (0, 1023), ''],
        ['FIFOA_F2_EN', 'FIFOA Frame ID Rangle Filter 2 Enabled', {True: 1, False: 0}, ''],
        ['FIFOA_F2_MODE', 'FIFOA Frame ID Rangle Filter 2 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOA_F2_SID_LOWER', 'FIFOA Frame ID Rangle Filter 2 Slot Id Lower', (0, 1023), ''],
        ['FIFOA_F2_SID_UPPER', 'FIFOA Frame ID Rangle Filter 2 Slot Id Upper', (0, 1023), ''],
        ['FIFOA_F3_EN', 'FIFOA Frame ID Rangle Filter 3 Enabled', {True: 1, False: 0}, ''],
        ['FIFOA_F3_MODE', 'FIFOA Frame ID Rangle Filter 3 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOA_F3_SID_LOWER', 'FIFOA Frame ID Rangle Filter 3 Slot Id Lower', (0, 1023), ''],
        ['FIFOA_F3_SID_UPPER', 'FIFOA Frame ID Rangle Filter 3 Slot Id Upper', (0, 1023), ''],
        ['FIFOB_EN','FIFOB Enabled', {True: 1, False: 0}, ''],
        ['FIFOB_Depth', 'FIFOB Depth', (0, 255), ''],
        ['FIFOB_MIAFV', 'FIFOB Message ID Acceptance Filter Value', (0, 65535), ''],
        ['FIFOB_MIAFM', 'FIFOB Message ID Acceptance Filter Mask', (0, 65535), ''],
        ['FIFOB_F0_EN', 'FIFOB Frame ID Rangle Filter 0 Enabled', {True: 1, False: 0}, ''],
        ['FIFOB_F0_MODE', 'FIFOB Frame ID Rangle Filter 0 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOB_F0_SID_LOWER', 'FIFOB Frame ID Rangle Filter 0 Slot Id Lower', (0, 1023), ''],
        ['FIFOB_F0_SID_UPPER', 'FIFOB Frame ID Rangle Filter 0 Slot Id Upper', (0, 1023), ''],
        ['FIFOB_F1_EN', 'FIFOB Frame ID Rangle Filter 1 Enabled', {True: 1, False: 0}, ''],
        ['FIFOB_F1_MODE', 'FIFOB Frame ID Rangle Filter 1 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOB_F1_SID_LOWER', 'FIFOB Frame ID Rangle Filter 1 Slot Id Lower', (0, 1023), ''],
        ['FIFOB_F1_SID_UPPER', 'FIFOB Frame ID Rangle Filter 1 Slot Id Upper', (0, 1023), ''],
        ['FIFOB_F2_EN', 'FIFOB Frame ID Rangle Filter 2 Enabled', {True: 1, False: 0}, ''],
        ['FIFOB_F2_MODE', 'FIFOB Frame ID Rangle Filter 2 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOB_F2_SID_LOWER', 'FIFOB Frame ID Rangle Filter 2 Slot Id Lower', (0, 1023), ''],
        ['FIFOB_F2_SID_UPPER', 'FIFOB Frame ID Rangle Filter 2 Slot Id Upper', (0, 1023), ''],
        ['FIFOB_F3_EN', 'FIFOB Frame ID Rangle Filter 3 Enabled', {True: 1, False: 0}, ''],
        ['FIFOB_F3_MODE', 'FIFOB Frame ID Rangle Filter 3 Mode', [('Acceptance', 0),('Rejection', 1)], ''],
        ['FIFOB_F3_SID_LOWER', 'FIFOB Frame ID Rangle Filter 3 Slot Id Lower', (0, 1023), ''],
        ['FIFOB_F3_SID_UPPER', 'FIFOB Frame ID Rangle Filter 3 Slot Id Upper', (0, 1023), ''],
    ]
}


HISTORY_CONFIG_FILE = os.path.expanduser("~/.flexray_adapter/config.yaml")


def mkdirs_exists_ok(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def load_history_fr_config_files():
    # Add test config files
    r = [f for f in glob.glob(os.path.join(os.path.abspath('./'), 'test', '*.yml'))]
    mkdirs_exists_ok(os.path.dirname(HISTORY_CONFIG_FILE))
    if os.path.exists(HISTORY_CONFIG_FILE):
        with open(HISTORY_CONFIG_FILE, 'r') as f:
            r += yaml.load(f)
    return [f for f in (set(r)) if os.path.exists(f)]


def save_history_fr_config_files(history_files):
    r = [f for f in glob.glob(os.path.join(os.path.abspath('./'), 'test', '*.yml'))]
    mkdirs_exists_ok(os.path.dirname(HISTORY_CONFIG_FILE))
    with open(HISTORY_CONFIG_FILE, 'w') as outfile:
        yaml.dump(history_files, outfile)


class ConnectThread(QThread):
    _connected_signal = pyqtSignal(Connection)
    _connect_failed_signal = pyqtSignal('QString')

    def __init__(self, on_connected, on_connect_failed, config):
        QThread.__init__(self)
        self._connected_signal.connect(on_connected)
        self._connect_failed_signal.connect(on_connect_failed)
        self._config = config

    def __del__(self):
        self.wait()

    def run(self):
        conn = Connection(self._config)
        try:
            conn.connect()
            self._connected_signal.emit(conn)
        except Exception as e:
            self._connect_failed_signal.emit('Connect failed: ' + str(e))


class ReceivePacketsThread(QThread):
    _frame_received_signal = pyqtSignal('QString', 'QString', 'int')
    _joined_cluster_signal = pyqtSignal()
    _disconnected_from_cluster_signal = pyqtSignal()
    _join_cluster_failed_signal = pyqtSignal()
    _flexray_fatal_error_signal = pyqtSignal()
    _exit_signal = pyqtSignal()
    _exception_signal = pyqtSignal('QString')
    _status_data_signal = pyqtSignal('QString')

    def __init__(self, conn, on_frame_received, on_exit, on_exception, on_joined_cluster, on_disonnected_from_cluster,
                 on_join_cluster_failed, on_fatal_error, on_status_data):
        QThread.__init__(self)
        self._conn = conn
        self._frame_received_signal.connect(on_frame_received)
        self._exit_signal.connect(on_exit)
        self._exception_signal.connect(on_exception)
        self._joined_cluster_signal.connect(on_joined_cluster)
        self._disconnected_from_cluster_signal.connect(on_disonnected_from_cluster)
        self._join_cluster_failed_signal.connect(on_join_cluster_failed)
        self._status_data_signal.connect(on_status_data)
        self._flexray_fatal_error_signal.connect(on_fatal_error)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    @staticmethod
    def parse_pifr0(pifr0, r):
        if pifr0 & FR_PIFR0_TBVA_IF_U16:
            r.append('Transmission across boundary on channel A')
        if pifr0 & FR_PIFR0_TBVB_IF_U16:
            r.append('Transmission across boundary on channel B')
        if pifr0 & FR_PIFR0_LTXA_IF_U16:
            r.append('pLatestTx Violation on Channel A')
        if pifr0 & FR_PIFR0_LTXB_IF_U16:
            r.append('pLatestTx Violation on Channel B')
        if pifr0 & FR_PIER0_CCL_IF_U16:
            r.append('Clock Correction Limit Reached')
        if pifr0 & FR_PIFR0_MOC_IF_U16:
            r.append('Missing Offset Correction')
        if pifr0 & FR_PIFR0_MRC_IF_U16:
            r.append('Missing Rate Correction')
        if pifr0 & FR_PIFR0_INTL_IF_U16:
            r.append('Internal Protocol Error')

    @staticmethod
    def parse_psr0_psr1(psr0, psr1, r):
        if (psr1 & FR_PSR1_FRZ_U16) != 0:
            r.append('POC state: halt, freezed. ')
        else:
            poc_state = psr0 & FR_PSR0_PROTSTATE_MASK_U16
            if poc_state == FR_PSR0_PROTSTATE_DEFAULT_CONFIG_U16:
                r.append('POC state: default config')
            elif poc_state == FR_PSR0_PROTSTATE_CONFIG_U16:
                r.append('POC state: config')
            elif poc_state == FR_PSR0_PROTSTATE_WAKEUP_U16:
                r.append('POC state: wakeup')
            elif poc_state == FR_PSR0_PROTSTATE_READY_U16:
                r.append('POC state: ready')
            elif poc_state == FR_PSR0_PROTSTATE_NORMAL_PASSIVE_U16:
                r.append('POC state: normal passive')
            elif poc_state == FR_PSR0_PROTSTATE_NORMAL_ACTIVE_U16:
                r.append('POC state: normal active')
            elif poc_state == FR_PSR0_PROTSTATE_HALT_U16:
                r.append('POC state: halt')
            elif poc_state == FR_PSR0_PROTSTATE_STARTUP_U16:
                r.append('POC state: startup')
                startup_status = psr0 & FR_PSR0_STARTUP_MASK_U16
                if startup_status == FR_PSR0_STARTUP_CCR_U16:
                    r.append('Startup status: coldstart collision resolution')
                elif startup_status == FR_PSR0_STARTUP_CL_U16:
                    r.append('Startup status: coldstart listen')
                elif startup_status == FR_PSR0_STARTUP_ICOC_U16:
                    r.append('Startup status: integration consistency check')
                elif startup_status == FR_PSR0_STARTUP_IL_U16:
                    r.append('Startup status: integration listen')
                elif startup_status == FR_PSR0_STARTUP_IS_U16:
                    r.append('Startup status: initialize schedule')
                elif startup_status == FR_PSR0_STARTUP_CCC_U16:
                    r.append('Startup status: coldstart consistency check')
                elif startup_status == FR_PSR0_STARTUP_ICLC_U16:
                    r.append('Startup status: integration coldstart check')
                elif startup_status == FR_PSR0_STARTUP_CG_U16:
                    r.append('Startup status: coldstart gap')
                elif startup_status == FR_PSR0_STARTUP_CJ_U16:
                    r.append('Startup status: coldstart join')
                else:
                    r.append('Startup status: Unknown: {}'.format(startup_status))
            err_mode = psr0 & FR_PSR0_ERRMODE_MASK_U16
            if err_mode == FR_PSR0_ERRMODE_ACTIVE_U16:
                r.append('Error mode: active')
            elif err_mode == FR_PSR0_ERRMODE_PASSIVE_U16:
                r.append('Error mode: passive')
            elif err_mode == FR_PSR0_ERRMODE_COMM_HALT_U16:
                r.append('Error mode: halt')

    @staticmethod
    def parse_psr2(psr2, r):
        if psr2 & FR_PSR2_NBVB_MASK_U16:
            r.append('NIT Boundary Violation on Channel B')
        if psr2 & FR_PSR2_NSEB_MASK_U16:
            r.append('NIT Syntax Error on Channel B')
        if psr2 & FR_PSR2_STCB_MASK_U16:
            r.append('Symbol Window Transmit Conflict on Channel B')
        if psr2 & FR_PSR2_SBVB_MASK_U16:
            r.append('Symbol Window Boundary Violation on Channel B')
        if psr2 & FR_PSR2_SSEB_MASK_U16:
            r.append('Symbol Window Syntax Error on Channel B')
        if psr2 & FR_PSR2_MTB_MASK_U16:
            r.append('Media Access Test Symbol MTS Received on Channel B')
        if psr2 & FR_PSR2_NBVA_MASK_U16:
            r.append('NIT Boundary Violation on Channel A')
        if psr2 & FR_PSR2_NSEA_MASK_U16:
            r.append('NIT Syntax Error on Channel A')
        if psr2 & FR_PSR2_STCA_MASK_U16:
            r.append('Symbol Window Transmit Conflict on Channel A')
        if psr2 & FR_PSR2_SBVA_MASK_U16:
            r.append('Symbol Window Boundary Violation on Channel A')
        if psr2 & FR_PSR2_SSEA_MASK_U16:
            r.append('Symbol Window Syntax Error on Channel A')
        if psr2 & FR_PSR2_MTA_MASK_U16:
            r.append('Media Access Test Symbol MTS Received on Channel A')
        r.append('Clock Correction Failed Counter: {}'.format(psr2 & FR_PSR2_CKCORFCNT_MASK_U16))

    @staticmethod
    def parse_psr3(psr3, r):
        if psr3 & FR_PSR3_ABVB_U16:
            r.append('Aggregated Boundary Violation on Channel B.')
        if psr3 & FR_PSR3_AACB_U16:
            r.append('Aggregated Additional Communication on Channel B.')
        if psr3 & FR_PSR3_ACEB_U16:
            r.append('Aggregated Content Error on Channel B.')
        if psr3 & FR_PSR3_ASEB_U16:
            r.append('Aggregated Syntax Error on Channel B.')
        if psr3 & FR_PSR3_AVFB_U16:
            r.append('Aggregated Valid Frame on Channel B.')
        if psr3 & FR_PSR3_ABVA_U16:
            r.append('Aggregated Boundary Violation on Channel A.')
        if psr3 & FR_PSR3_AACA_U16:
            r.append('Aggregated Additional Communication on Channel A.')
        if psr3 & FR_PSR3_ACEA_U16:
            r.append('Aggregated Content Error on Channel A.')
        if psr3 & FR_PSR3_ASEA_U16:
            r.append('Aggregated Syntax Error on Channel A.')
        if psr3 & FR_PSR3_AVFA_U16:
            r.append('Aggregated Valid Frame on Channel A.')

    @staticmethod
    def parse_sync_frame_table(a_even_cnt, b_even_cnt, a_odd_cnt, b_odd_cnt, sft, r):
        if a_even_cnt + b_even_cnt + a_odd_cnt + b_odd_cnt == 0:
            return r.append('Sync frame table is empty or invalid')
        r.append('Sync Frame Table:\n\tChannel\tCycle\tFrameID\tDeviation')
        aggregated_i = 0
        for i in range(a_even_cnt):
            r.append('\tA\teven\t{}\t{}'.format(sft[aggregated_i], sft[aggregated_i + 1]))
            aggregated_i += 2
        for i in range(b_even_cnt):
            r.append('\tB\teven\t{}\t{}'.format(sft[aggregated_i], sft[aggregated_i + 1]))
            aggregated_i += 2
        for i in range(a_odd_cnt):
            r.append('\tA\todd\t{}\t{}'.format(sft[aggregated_i], sft[aggregated_i + 1]))
            aggregated_i += 2
        for i in range(b_odd_cnt):
            r.append('\tB\todd\t{}\t{}'.format(sft[aggregated_i], sft[aggregated_i + 1]))
            aggregated_i += 2

    def on_pkt_recved(self, pkt_type, frame_id, payload):
        if pkt_type == PACKET_TYPE_FLEXRAY_FRAME:
            hexes = ' '.join([format(x, 'X') for x in payload])
            self._frame_received_signal.emit(str(frame_id), hexes, len(payload))
        elif pkt_type == PACKET_TYPE_FLEXRAY_JOINED_CLUSTER:
            self._joined_cluster_signal.emit()
        elif pkt_type == PACKET_TYPE_FLEXRAY_JOIN_CLUSTER_FAILED:
            self._join_cluster_failed_signal.emit()
        elif pkt_type == PACKET_TYPE_FLEXRAY_DISCONNECTED_FROM_CLUSTER:
            self._disconnected_from_cluster_signal.emit()
        elif pkt_type == PACKET_TYPE_FLEXRAY_FATAL_ERROR:
            self._flexray_fatal_error_signal.emit()
        elif pkt_type == PACKET_TYPE_HEALTH:
            t = self._conn.parse_health_packet(payload)
            if not t:
                return
            psr0, psr1, psr2, psr3, pifr0, max_rc, max_oc, min_rc, min_oc, a_even_cnt, b_even_cnt, a_odd_cnt, b_odd_cnt, sft = t
            r = []
            self.parse_psr0_psr1(psr0, psr1, r)
            self.parse_psr2(psr2, r)
            self.parse_psr3(psr3, r)
            self.parse_pifr0(pifr0, r)
            r.append('Rate correction Max: {}, Min: {}'.format(max_rc, min_rc))
            r.append('Offset correction Max: {}, Min: {}'.format(max_oc, min_oc))
            self.parse_sync_frame_table(a_even_cnt, b_even_cnt, a_odd_cnt, b_odd_cnt, sft, r)
            self._status_data_signal.emit('\n'.join(r))

    def on_peer_shutdown(self):
        self.stop()

    def run(self):
        try:
            while not self.stopped():
                self._conn.receive_packet(self.on_pkt_recved, self.on_peer_shutdown)
        except Exception as e:
            self._exception_signal.emit('Receive packet error: ' + str(e))
        finally:
            self._exit_signal.emit()


class SendFrameThread(QThread):
    _exit_signal = pyqtSignal()
    _exception_signal = pyqtSignal('QString')
    _sent_frame_signal = pyqtSignal('int')

    def __init__(self, conn, frame_ids, payload, interval, on_exception, on_sent_frame, on_exit):
        QThread.__init__(self)
        self.conn = conn
        self.frame_ids = frame_ids
        self.payload = payload
        self.interval = interval
        self._exception_signal.connect(on_exception)
        self._sent_frame_signal.connect(on_sent_frame)
        self._exit_signal.connect(on_exit)
        self.timer = QTimer()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def on_timer(self):
        if self.stopped():
            self.quit()
        else:
            try:
                for frame_id in self.frame_ids:
                    sent = self.conn.send_frame(frame_id, self.payload)
                    self._sent_frame_signal.emit(sent)
            except Exception as e:
                self._exception_signal.emit('Send frame error: ' + str(e))
            self.timer.singleShot(self.interval, self.on_timer)

    def run(self):
        self.timer.singleShot(self.interval, self.on_timer)
        self.exec()
        self._exit_signal.emit()


class ConnectOrConfigDialog(QDialog):
    def __init__(self, config, mode='config'):
        super(ConnectOrConfigDialog, self).__init__()
        self.cur_config = config
        self.form_widgets = {}
        self.mode = mode
        self.verify_result = []
        self.timer = QTimer()
        tabs = QTabWidget()
        for group_name, fields in config_fields.items():
            layout = QFormLayout()
            self.generate_form_fields(fields, layout)
            w = QWidget()
            w.setLayout(layout)
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(w)
            tabs.addTab(scroll_area, group_name)

        self.rx_msg_bufs_table = QTableWidget()
        headers = ['', 'Frame ID', 'Channels', 'CycleCounterFilter', 'CCF Value', 'CCF Mask']
        self.rx_msg_bufs_table.setColumnCount(len(headers))
        self.rx_msg_bufs_table.setHorizontalHeaderLabels(headers)
        for i in range(len(headers)):
            self.rx_msg_bufs_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.rx_msg_bufs_table)
        add_rx_buf_btn = QPushButton('Add')
        add_rx_buf_btn.clicked.connect(self.add_rx_msg_buf)
        clear_rx_buf_btn = QPushButton('Remove All')
        clear_rx_buf_btn.clicked.connect(self.clear_rx_msg_buf)
        btn_box = QDialogButtonBox(Qt.Horizontal)
        btn_box.addButton(add_rx_buf_btn, QDialogButtonBox.ActionRole)
        btn_box.addButton(clear_rx_buf_btn, QDialogButtonBox.ActionRole)
        layout = QVBoxLayout()
        layout.addWidget(btn_box)
        layout.addWidget(scroll_area)
        gb = QGroupBox()
        gb.setLayout(layout)
        tabs.addTab(gb, 'Rx Message Buffers')

        self.tx_msg_bufs_table = QTableWidget()
        headers = ['', 'Frame ID', 'Channels', 'PayloadLenMax', 'DynPayloadLen', 'PayloadPreamble', 'CycleCounterFilter', 'CCF Value', 'CCF Mask']
        self.tx_msg_bufs_table.setColumnCount(len(headers))
        self.tx_msg_bufs_table.setHorizontalHeaderLabels(headers)
        for i in range(len(headers)):
            self.tx_msg_bufs_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.tx_msg_bufs_table)
        add_tx_buf_btn = QPushButton('Add')
        add_tx_buf_btn.clicked.connect(self.add_tx_msg_buf)
        clear_tx_buf_btn = QPushButton('Remove All')
        clear_tx_buf_btn.clicked.connect(self.clear_tx_msg_buf)
        btn_box = QDialogButtonBox(Qt.Horizontal)
        btn_box.addButton(add_tx_buf_btn, QDialogButtonBox.ActionRole)
        btn_box.addButton(clear_tx_buf_btn, QDialogButtonBox.ActionRole)
        layout = QVBoxLayout()
        layout.addWidget(btn_box)
        layout.addWidget(scroll_area)
        gb = QGroupBox()
        gb.setLayout(layout)
        tabs.addTab(gb, 'Tx Message Buffers')
        tabs.setMinimumSize(600, 400)

        self.config_cb = QComboBox(self)
        self.history = load_history_fr_config_files()
        for path in self.history:
            self.config_cb.addItem(path)

        layout = QHBoxLayout()
        layout.addWidget(self.config_cb)
        load_btn = QPushButton("&Load selected file")
        load_btn.clicked.connect(lambda : self.load_config(self.config_cb.currentText()))
        layout.addWidget(load_btn)
        open_btn = QPushButton("&Open another file...")
        open_btn.clicked.connect(self.open_config_file)
        layout.addWidget(open_btn)
        layout.setStretch(0, 1000)
        top_w = QGroupBox(self)
        top_w.setLayout(layout)

        if self.mode == 'config':
            btn_box = QDialogButtonBox(QDialogButtonBox.Close)
            save_as_btn = QPushButton("&Save as...")
            save_as_btn.clicked.connect(self.save_config_to_file)
            btn_box.addButton(save_as_btn, QDialogButtonBox.ActionRole)
            verify_btn = QPushButton("&Verify")
            verify_btn.clicked.connect(self.verify_config)
            btn_box.addButton(verify_btn, QDialogButtonBox.ActionRole)
            btn_box.rejected.connect(self.reject)
            self.setWindowTitle("Manage FlexRay configurations")
        else:
            btn_box = QDialogButtonBox(QDialogButtonBox.Cancel)
            save_as_btn = QPushButton("&Save as...")
            save_as_btn.clicked.connect(self.save_config_to_file)
            btn_box.addButton(save_as_btn, QDialogButtonBox.ActionRole)
            connect_btn = QPushButton("&Connect")
            btn_box.addButton(connect_btn, QDialogButtonBox.AcceptRole)
            btn_box.accepted.connect(self.accept)
            btn_box.rejected.connect(self.reject)
            self.setWindowTitle("Connect to FlexRay adapter & join into FlexRay network")

        layout = QGridLayout()
        layout.addWidget(top_w, 0, 0, 1, 2)
        layout.addWidget(tabs, 1, 0, 1, 2)
        layout.addWidget(btn_box, 2, 1)
        layout.setRowStretch(1, 100)
        self.setLayout(layout)
        self.update_form_fields()
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)

    def accept(self):
        ok, err = verify_config(self.cur_config)
        while not ok:
            mb = QMessageBox(QMessageBox.Information, "Invalid FlexRay configuration", '')
            if type(err) == str:
                mb.setText(err)
                mb.exec()
                return
            elif type(err) == tuple:
                mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                mb.setDefaultButton(QMessageBox.Yes)
                mb.setText('{} should be {}, correct it and reconnect now?'.format(err[0], err[1]))
                if mb.exec() != QMessageBox.Yes:
                    return
                self.cur_config[err[0]] = err[1]
                self.update_form_fields()
                ok, err = verify_config(self.cur_config)
        self.verify_result = err
        super(ConnectOrConfigDialog, self).accept()

    def verify_config(self):
        ok, err = verify_config(self.cur_config)
        if not ok:
            mb = QMessageBox(QMessageBox.Information, "Invalid FlexRay configuration", '')
            if type(err) == str:
                mb.setText(err)
            elif type(err) == tuple:
                mb.setText('{} should be {}.'.format(err[0], err[1]))
            mb.exec()
        else:
            QMessageBox(QMessageBox.Information, "Verify Result", 'Verify succeeded.\n\n' + '\n'.join(err)).exec()

    def show_progress_dlg(self, t):
        progress_dlg = QProgressDialog('Loading config', None, 0, 100, self, Qt.Window)
        progress_dlg.setAutoReset(True)
        progress_dlg.setAutoClose(True)
        progress_dlg.setWindowModality(Qt.WindowModal)
        progress_dlg.setWindowTitle('Loading...')
        progress_dlg.setLabelText(t)
        progress_dlg.setValue(10)
        self.timer.singleShot(500, lambda dlg=progress_dlg: dlg.setValue(50))
        self.timer.singleShot(1000, lambda dlg=progress_dlg: dlg.setValue(100))
        progress_dlg.show()

    def load_config(self, path):
        if not os.path.exists(path):
            return False
        with open(path, 'r') as f:
            c = yaml.load(f)
        ok, missed_keys = verify_config_format(c)
        if not ok:
            mb = QMessageBox(QMessageBox.Information, 'Invalid FlexRay config', '')
            mb.setText('Missing parameters: ' + ', '.join(missed_keys) + '\nFix it now?')
            mb.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            mb.setDefaultButton(QMessageBox.Yes)
            if mb.exec() != QMessageBox.Yes:
                return False
            for k in missed_keys:
                c[k] = default_fr_config[k]
        self.cur_config = c
        self.update_form_fields()
        self.show_progress_dlg('Loaded config: {}'.format(os.path.basename(path)))
        return True

    def save_config_to_file(self):
        t = QFileDialog.getSaveFileName(
            None, 'Save config to file', os.path.expanduser("~"), "YAML files (*.yml)")
        if len(t[0]) == 0:
            return
        mkdirs_exists_ok(os.path.dirname(t[0]))
        with open(t[0], 'w') as outfile:
            yaml.dump(self.cur_config, outfile)

        self.show_progress_dlg('Saving config to: {}'.format(t[0]))

    def open_config_file(self):
        t = QFileDialog.getOpenFileName(
            None, 'Open file', os.path.expanduser("~"), "YAML files (*.yml)")
        if len(t[0]) == 0:
            return
        if self.load_config(t[0]):
            i = self.config_cb.findText(t[0])
            if i == -1:
                self.history.append(t[0])
                save_history_fr_config_files(self.history)
                self.config_cb.insertItem(0, t[0])
                self.config_cb.setCurrentIndex(0)
            else:
                self.config_cb.setCurrentIndex(i)

    def update_config_by_spinbox(self, key, val):
        self.cur_config[key] = val

    def update_config_by_checkbox(self, opts, key, cb):
        self.cur_config[key] = opts[cb.isChecked()]

    def update_config_by_combobox(self, opts, key, selected_idx):
        self.cur_config[key] = opts[selected_idx][1]

    def generate_form_fields(self, fields, layout):
        for f in fields:
            if type(f[2]) is tuple:
                if type(f[2][0]) is float:
                    w = QDoubleSpinBox()
                    w.setRange(f[2][0], f[2][1])
                    self.form_widgets[f[0]] = w
                    w.valueChanged.connect(lambda val, key=f[0]: self.update_config_by_spinbox(key, val))
                else:
                    w = QSpinBox()
                    self.form_widgets[f[0]] = w
                    w.setMinimum(f[2][0])
                    w.setMaximum(f[2][1])
                    w.valueChanged.connect(lambda val, key=f[0]: self.update_config_by_spinbox(key, val))
                hbox = QHBoxLayout()
                hbox.addWidget(w)
                hbox.addWidget(QLabel(f[-1]))
                layout.addRow('{} ({} - {})'.format(f[1], f[2][0], f[2][1]), hbox)
            elif type(f[2]) is dict:
                cb = QCheckBox()
                self.form_widgets[f[0]] = cb
                cb.stateChanged.connect(lambda _, key=f[0], cb1=cb, opts=f[2]: self.update_config_by_checkbox(opts, key, cb1))
                layout.addRow(f[1], cb)
            elif type(f[2]) is list:
                cb = QComboBox()
                self.form_widgets[f[0]] = cb
                for (opt_name, opt_val) in f[2]:
                    cb.addItem(str(opt_name))
                cb.currentIndexChanged.connect(
                    lambda selected_idx, key=f[0], opts=f[2]: self.update_config_by_combobox(opts, key, selected_idx))
                hbox = QHBoxLayout()
                hbox.addWidget(cb)
                hbox.addWidget(QLabel(f[-1]))
                layout.addRow(f[1], hbox)

    def update_form_fields(self):
        for group_name, fields in config_fields.items():
            for f in fields:
                if f[0] not in self.cur_config:
                    continue
                if type(f[2]) is tuple:
                    assert (type(self.form_widgets[f[0]]) == QSpinBox or type(
                        self.form_widgets[f[0]]) == QDoubleSpinBox)
                    self.form_widgets[f[0]].setValue(self.cur_config[f[0]])
                elif type(f[2]) is dict:
                    assert (type(self.form_widgets[f[0]]) == QCheckBox)
                    for k, v in f[2].items():
                        if v == self.cur_config[f[0]]:
                            self.form_widgets[f[0]].setChecked(k)
                            break
                elif type(f[2]) is list:
                    assert (type(self.form_widgets[f[0]]) == QComboBox)
                    for (k, v) in f[2]:
                        if v == self.cur_config[f[0]]:
                            self.form_widgets[f[0]].setCurrentText(str(k))
                            break
        self.rx_msg_bufs_table.setRowCount(len(self.cur_config['RxMsgBufs']))
        for i, rmb in enumerate(self.cur_config['RxMsgBufs']):
            remove_btn = QPushButton()
            remove_btn.clicked.connect(lambda _, i=i: self.remove_rx_msg_buf(i))
            remove_btn.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_LineEditClearButton)))
            self.rx_msg_bufs_table.setCellWidget(i, 0, remove_btn)

            sb = QSpinBox()
            sb.setMinimum(1)
            sb.setMaximum(cSlotIDMax)
            sb.setValue(rmb['FrameId'])
            sb.valueChanged.connect(lambda val, rmb=rmb, key='FrameId': rmb.__setitem__(key, val))
            self.rx_msg_bufs_table.setCellWidget(i, 1, sb)

            cb = QComboBox()
            opts = [('CHANNEL_A', 1), ('CHANNEL_B', 2), ('CHANNEL_AB', 3)]
            for k, v in opts:
                cb.addItem(k)
            selected = [index for index, v in enumerate(opts) if v[1] == rmb['Channels']]
            cb.setCurrentIndex(selected[0])
            cb.currentIndexChanged.connect(
                lambda selected_idx, rmb=rmb, key='Channels', opts=opts: rmb.__setitem__(key, opts[selected_idx][1]))
            self.rx_msg_bufs_table.setCellWidget(i, 2, cb)

            cb = QCheckBox()
            cb.setChecked(True) if rmb['CCF_EN'] == 1 else cb.setChecked(False)
            cb.stateChanged.connect(lambda _, rmb=rmb, key='CCF_EN', cb=cb: rmb.__setitem__(key, (1 if cb.isChecked() else 0)))
            self.rx_msg_bufs_table.setCellWidget(i, 3, cb)

            sb = QSpinBox()
            sb.setMinimum(0)
            sb.setMaximum(cCycleCountMax)
            sb.setValue(rmb['CCF_VAL'])
            sb.valueChanged.connect(lambda val, rmb=rmb, key='CCF_VAL': rmb.__setitem__(key, val))
            self.rx_msg_bufs_table.setCellWidget(i, 4, sb)

            sb = QSpinBox()
            sb.setMinimum(0)
            sb.setMaximum(255)
            sb.setValue(rmb['CCF_MASK'])
            sb.valueChanged.connect(lambda val, rmb=rmb, key='CCF_MASK': rmb.__setitem__(key, val))
            self.rx_msg_bufs_table.setCellWidget(i, 5, sb)

        self.tx_msg_bufs_table.setRowCount(len(self.cur_config['TxMsgBufs']))
        for i, rmb in enumerate(self.cur_config['TxMsgBufs']):
            remove_btn = QPushButton()
            remove_btn.clicked.connect(lambda _, i=i: self.remove_tx_msg_buf(i))
            remove_btn.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_LineEditClearButton)))
            self.tx_msg_bufs_table.setCellWidget(i, 0, remove_btn)

            sb = QSpinBox()
            sb.setMinimum(1)
            sb.setMaximum(cSlotIDMax)
            sb.setValue(rmb['FrameId'])
            sb.valueChanged.connect(lambda val, rmb=rmb, key='FrameId': rmb.__setitem__(key, val))
            self.tx_msg_bufs_table.setCellWidget(i, 1, sb)

            cb = QComboBox()
            opts = [('CHANNEL_A', 1), ('CHANNEL_B', 2), ('CHANNEL_AB', 3)]
            for k, v in opts:
                cb.addItem(k)
            selected = [index for index, v in enumerate(opts) if v[1] == rmb['Channels']]
            cb.setCurrentIndex(selected[0])
            cb.currentIndexChanged.connect(
                lambda selected_idx, rmb=rmb, key='Channels', opts=opts: rmb.__setitem__(key, opts[selected_idx][1]))
            self.tx_msg_bufs_table.setCellWidget(i, 2, cb)

            sb = QSpinBox()
            sb.setMinimum(0)
            sb.setMaximum(cPayloadLengthMax)
            sb.setValue(rmb['PayloadLenMax'])
            sb.valueChanged.connect(lambda val, rmb=rmb, key='PayloadLenMax': rmb.__setitem__(key, val))
            self.tx_msg_bufs_table.setCellWidget(i, 3, sb)

            cb = QCheckBox()
            cb.setChecked(True) if rmb['DynPayloadLen'] == 1 else cb.setChecked(False)
            cb.stateChanged.connect(lambda _, rmb=rmb, key='DynPayloadLen', cb=cb: rmb.__setitem__(key, (1 if cb.isChecked() else 0)))
            self.tx_msg_bufs_table.setCellWidget(i, 4, cb)

            cb = QCheckBox()
            cb.setChecked(True) if rmb['PPI'] == 1 else cb.setChecked(False)
            cb.stateChanged.connect(lambda _, rmb=rmb, key='PPI', cb=cb: rmb.__setitem__(key, (1 if cb.isChecked() else 0)))
            self.tx_msg_bufs_table.setCellWidget(i, 5, cb)

            cb = QCheckBox()
            cb.setChecked(True) if rmb['CCF_EN'] == 1 else cb.setChecked(False)
            cb.stateChanged.connect(lambda _, rmb=rmb, key='CCF_EN', cb=cb: rmb.__setitem__(key, (1 if cb.isChecked() else 0)))
            self.tx_msg_bufs_table.setCellWidget(i, 6, cb)

            sb = QSpinBox()
            sb.setMinimum(0)
            sb.setMaximum(cCycleCountMax)
            sb.setValue(rmb['CCF_VAL'])
            sb.valueChanged.connect(lambda val, rmb=rmb, key='CCF_VAL': rmb.__setitem__(key, val))
            self.tx_msg_bufs_table.setCellWidget(i, 7, sb)

            sb = QSpinBox()
            sb.setMinimum(0)
            sb.setMaximum(255)
            sb.setValue(rmb['CCF_MASK'])
            sb.valueChanged.connect(lambda val, rmb=rmb, key='CCF_MASK': rmb.__setitem__(key, val))
            self.tx_msg_bufs_table.setCellWidget(i, 8, sb)

    def remove_rx_msg_buf(self, i):
        del self.cur_config['RxMsgBufs'][i]
        self.update_form_fields()

    def add_rx_msg_buf(self, i):
        max_frame_id = 0
        channels = 0
        if len(self.cur_config['RxMsgBufs']) > 0:
            max_frame_id = max([b['FrameId'] for b in self.cur_config['RxMsgBufs']])
            channels = self.cur_config['RxMsgBufs'][-1]['Channels']
        self.cur_config['RxMsgBufs'].append({
            'FrameId': (max_frame_id + 1 if max_frame_id < cSlotIDMax else 1),
            'Channels': channels,
            'CCF_EN': 0,
            'CCF_VAL': 0,
            'CCF_MASK': 0,
        })
        self.update_form_fields()

    def clear_rx_msg_buf(self, i):
        del self.cur_config['RxMsgBufs'][:]
        self.update_form_fields()

    def remove_tx_msg_buf(self, i):
        del self.cur_config['TxMsgBufs'][i]
        self.update_form_fields()

    def add_tx_msg_buf(self, i):
        max_frame_id = 0
        payload_len_max = 0
        channels = 0
        if len(self.cur_config['TxMsgBufs']) > 0:
            max_frame_id = max([b['FrameId'] for b in self.cur_config['TxMsgBufs']])
            payload_len_max = self.cur_config['TxMsgBufs'][-1]['PayloadLenMax']
            channels = self.cur_config['TxMsgBufs'][-1]['Channels']
        self.cur_config['TxMsgBufs'].append({
            'FrameId': (max_frame_id + 1 if max_frame_id < cSlotIDMax else 1),
            'PayloadLenMax': payload_len_max,
            'Channels': channels,
            'CCF_EN': 0,
            'CCF_VAL': 0,
            'CCF_MASK': 0,
            'DynPayloadLen': 0,
            'PPI': 0,
        })
        self.update_form_fields()

    def clear_tx_msg_buf(self, i):
        del self.cur_config['TxMsgBufs'][:]
        self.update_form_fields()



class SendFrameDialog(QDialog):
    def __init__(self, config):
        super(SendFrameDialog, self).__init__()
        layout = QFormLayout()
        self.frame_ids_lv = QListWidget()
        for tmb in config['TxMsgBufs']:
            if tmb['FrameId'] > config['gNumberOfStaticSlots']:
                item = QListWidgetItem('Dynamic slot ' + str(tmb['FrameId']))
            else:
                item = QListWidgetItem('Static slot ' + str(tmb['FrameId']))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.frame_ids_lv.addItem(item)
        self.frame_ids_lv.itemChanged.connect(self.enable_disable_start_send_btn)
        hbox = QHBoxLayout()
        hbox.addWidget(self.frame_ids_lv)
        select_all_btn = QPushButton("&Select All")
        select_all_btn.clicked.connect(lambda: self.select_deselect_all_slots(Qt.Checked))
        deselect_all_btn = QPushButton("&Deselect All")
        deselect_all_btn.clicked.connect(lambda: self.select_deselect_all_slots(Qt.Unchecked))
        btn_box = QDialogButtonBox(Qt.Vertical)
        btn_box.addButton(select_all_btn, QDialogButtonBox.ActionRole)
        btn_box.addButton(deselect_all_btn, QDialogButtonBox.ActionRole)
        hbox.addWidget(btn_box)
        layout.addRow('Send on slots', hbox)
        self.payload_le = QLineEdit()
        self.payload_le.setInputMask('HH ' * max(config['gPayloadLengthStatic'], config['pPayloadLengthDynMax']) * 2 + ';_')
        self.payload_le.textChanged.connect(self.on_payload_le_text_changed)
        layout.addRow('Payload (HEX)', self.payload_le)
        self.payload_desc = QLabel('0 bytes')
        layout.addRow('', self.payload_desc)
        self.one_time_rb = QRadioButton('Send once')
        self.one_time_rb.setChecked(True)
        self.one_time_rb.toggled.connect(self.on_one_time_rb_state_change)
        layout.addRow('', self.one_time_rb)
        self.periodically_rb = QRadioButton('Send periodically')
        self.periodically_rb.setChecked(False)
        self.periodically_rb.toggled.connect(self.on_periodically_rb_state_change)
        layout.addRow('', self.periodically_rb)
        self.interval_sb = QSpinBox()
        self.interval_sb.setMinimum(0)
        self.interval_sb.setMaximum(10 * 1000)
        self.interval_sb.setValue(1000)
        layout.addRow('Interval (millisecond)', self.interval_sb)

        gb = QGroupBox()
        gb.setLayout(layout)
        self.start_send_btn = QPushButton('Start Sending')
        self.start_send_btn.setEnabled(False)
        btn_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        btn_box.addButton(self.start_send_btn, QDialogButtonBox.AcceptRole)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addWidget(gb)
        layout.addWidget(btn_box)
        self.setLayout(layout)
        self.setWindowTitle("Send FlexRay Frames")
        self.setMinimumSize(640, 360)
        self.setWindowFlags(self.windowFlags() | Qt.WindowSystemMenuHint | Qt.WindowMinMaxButtonsHint)

    def select_deselect_all_slots(self, check):
        for i in range(self.frame_ids_lv.count()):
            self.frame_ids_lv.item(i).setCheckState(check)

    def payload_hex_list(self):
        return [int(x, 16) for x in self.payload_le.text().split()]

    def enable_disable_start_send_btn(self):
        if len(self.payload_hex_list()) > 0 and any(self.frame_ids_lv.item(i).checkState() == Qt.Checked for i in range(self.frame_ids_lv.count())):
            if not self.start_send_btn.isEnabled():
                self.start_send_btn.setEnabled(True)
        else:
            if self.start_send_btn.isEnabled():
                self.start_send_btn.setEnabled(False)

    def on_periodically_rb_state_change(self, _):
        if self.periodically_rb.isChecked():
            if self.one_time_rb.isChecked():
                self.one_time_rb.setChecked(False)
            self.interval_sb.setEnabled(True)

    def on_one_time_rb_state_change(self, _):
        if self.one_time_rb.isChecked():
            self.periodically_rb.setChecked(False)
            self.interval_sb.setEnabled(False)

    def on_payload_le_text_changed(self, text):
        self.payload_desc.setText('{} bytes'.format(len([int(x, 16) for x in text.split()])))
        self.enable_disable_start_send_btn()


class FlexRayGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer)
        self.recv_packets_thread = None
        self.send_frame_thread = None
        self.conn = None
        self.cur_config = default_fr_config
        self.send_frame_dlg = None
        self.tx_frames = self.tx_bytes = self.tx_bps = self.tx_bytes_within_this_second = 0
        self.rx_frames = self.rx_bytes = self.rx_bps = self.rx_bytes_within_this_second = 0
        self.rx_time = self.tx_time = time.time()
        header = ['Time', 'Frame ID', 'Length', 'HEX']
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(len(header))
        self.frame_table.setHorizontalHeaderLabels(header)
        self.frame_table.setMinimumSize(400, 300)
        self.frame_table.setSortingEnabled(True)

        clear_rx_frames_btn = QPushButton("&Remove All")
        clear_rx_frames_btn.clicked.connect(self.clear_rx_frame_table)
        clear_rx_frames_btn.setMaximumWidth(100)
        clear_rx_frames_btn.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_TrashIcon)))
        layout = QVBoxLayout()
        layout.addWidget(clear_rx_frames_btn)
        layout.addWidget(self.frame_table)
        self.rx_frames_gb = QGroupBox('Rx Frames')
        self.rx_frames_gb.setLayout(layout)

        self.connect_btn = QPushButton("Join into FlexRay network")
        self.connect_btn.setFixedWidth(200)
        self.connect_btn.clicked.connect(self.connect_or_disconnect)
        self.config_btn = QPushButton("&Manage configurations")
        self.config_btn.setFixedWidth(200)
        self.config_btn.clicked.connect(self.manage_config)
        self.send_frame_btn = QPushButton("&Send Frame")
        self.send_frame_btn.setFixedWidth(200)
        self.send_frame_btn.setEnabled(False)
        self.send_frame_btn.clicked.connect(self.start_or_stop_send_frame_thd)
        layout = QHBoxLayout()
        layout.addWidget(self.connect_btn)
        layout.addWidget(self.config_btn)
        layout.addWidget(self.send_frame_btn)
        tool_bar = QWidget()
        tool_bar.setLayout(layout)

        icon_btn_style = "QPushButton { background-color: #FFFFFF; border: 1px solid grey; height: 48;}"
        self.disconnected_text_style = "QPushButton {border: 0px; text-decoration: underline; text-align: center; color : black;}"
        self.connected_text_style = "QPushButton {border: 0px; text-decoration: underline; text-align: center; color : green;}"
        self.error_text_style = "QPushButton {border: 0px; text-decoration: underline; text-align: center; color : red;}"
        layout = QGridLayout()
        btn = QPushButton('My Computer')
        btn.setIcon(QIcon(QApplication.style().standardIcon(QStyle.SP_ComputerIcon)))
        btn.setStyleSheet(icon_btn_style)
        layout.addWidget(btn, 0, 0)
        self.status_label_left = QPushButton('   Not connected   ')
        self.status_label_left.setStyleSheet(self.disconnected_text_style)
        layout.addWidget(self.status_label_left, 0, 1)
        btn = QPushButton('FlexRay Adapter')
        btn.setStyleSheet(icon_btn_style)
        layout.addWidget(btn, 0, 2)
        self.status_label_right = QPushButton('   Not connected   ')
        self.status_label_right.setStyleSheet(self.disconnected_text_style)
        layout.addWidget(self.status_label_right, 0, 3)
        btn = QPushButton('FlexRay Network')
        btn.setStyleSheet(icon_btn_style)
        layout.addWidget(btn, 0, 4)
        self.detail_status = QLabel()
        self.detail_status.setWordWrap(True)
        layout.addWidget(self.detail_status, 1, 0)
        self.statistics_label = QLabel()
        self.statistics_label.setWordWrap(True)
        layout.addWidget(self.statistics_label, 1, 2)
        stats_gb = QGroupBox('Status')
        stats_gb.setLayout(layout)

        self.log_lv = QListWidget()
        bottom_vbox_layout = QVBoxLayout()
        bottom_vbox_layout.addWidget(self.log_lv)

        bottom_group_box = QGroupBox('Logs')
        bottom_group_box.setLayout(bottom_vbox_layout)

        layout = QVBoxLayout()
        layout.addWidget(tool_bar)
        layout.addWidget(self.rx_frames_gb)
        layout.addWidget(stats_gb)
        w = QWidget()
        w.setLayout(layout)
        splitter = QSplitter(Qt.Vertical, self)
        splitter.addWidget(w)
        splitter.addWidget(bottom_group_box)
        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setGeometry(200, 200, 800, 450)
        self.setWindowTitle('FlexRay Tool')
        self.show()

    def add_log(self, t):
        self.log_lv.addItem(t)
        self.log_lv.scrollToBottom()

    def connect_or_disconnect(self):
        if not self.recv_packets_thread:
            connect_dlg = ConnectOrConfigDialog(self.cur_config, mode='connect')
            r = connect_dlg.exec()
            self.cur_config = connect_dlg.cur_config
            if r != QDialog.Accepted:
                return
            for t in connect_dlg.verify_result:
                self.add_log(t)
            self.status_label_left.setText('Connecting...')
            self.connect_btn.setEnabled(False)
            ConnectThread(self.on_connected, self.on_connect_failed, self.cur_config).start()

        else:
            self.recv_packets_thread.stop()
            self.connect_btn.setEnabled(False)

    def manage_config(self):
        cfg_dlg = ConnectOrConfigDialog(self.cur_config, mode='config')
        cfg_dlg.exec()
        self.cur_config = cfg_dlg.cur_config

    def on_connected(self, conn):
        self.conn = conn
        self.recv_packets_thread = ReceivePacketsThread(
            conn, self.on_frame_received, self.on_sock_disconnect, self.on_recv_pkt_thd_exception, self.on_joined_cluster,
            self.on_disconnected_from_cluster, self.on_join_cluster_failed, self.on_fatal_error, self.on_status_data)
        self.recv_packets_thread.start()
        self.connect_btn.setText('Disconnect')
        self.connect_btn.setEnabled(True)
        self.status_label_left.setText('   Connected   ')
        self.status_label_left.setStyleSheet(self.connected_text_style)
        self.status_label_right.setText('   Joining cluster...   ')

    def on_connect_failed(self, e):
        self.add_log(e)
        self.detail_status.setText('')
        self.status_label_left.setText('   Connect failed.   ')
        self.status_label_left.setStyleSheet(self.disconnected_text_style)
        self.connect_btn.setText('Join into FlexRay network')
        self.connect_btn.setEnabled(True)

    def on_sock_disconnect(self):
        if self.timer.isActive():
            self.timer.stop()
        self.recv_packets_thread.wait()
        self.recv_packets_thread = None
        if self.send_frame_thread:
            self.start_or_stop_send_frame_thd()
        self.conn.close()
        self.detail_status.setText('')
        self.status_label_left.setText('   Disconnected.   ')
        self.status_label_left.setStyleSheet(self.disconnected_text_style)
        self.status_label_right.setText('   Disconnected.   ')
        self.status_label_right.setStyleSheet(self.disconnected_text_style)
        self.send_frame_btn.setEnabled(False)
        self.rx_frames = self.rx_bytes = self.rx_bps = self.rx_bytes_within_this_second = 0
        self.tx_frames = self.tx_bytes = self.tx_bps = self.tx_bytes_within_this_second = 0
        self.connect_btn.setText('Join into FlexRay network')
        self.connect_btn.setEnabled(True)

    def start_or_stop_send_frame_thd(self):
        if not self.send_frame_thread:
            self.send_frame_dlg = SendFrameDialog(self.cur_config)
            if self.send_frame_dlg.exec() != QDialog.Accepted:
                return
            frame_ids = [self.cur_config['TxMsgBufs'][i]['FrameId'] for i in range(self.send_frame_dlg.frame_ids_lv.count()) if self.send_frame_dlg.frame_ids_lv.item(i).checkState() == Qt.Checked]
            hex_list = self.send_frame_dlg.payload_hex_list()
            payload = struct.pack('>' + 'B' * len(hex_list), *hex_list)
            if self.send_frame_dlg.periodically_rb.isChecked():
                self.send_frame_thread = SendFrameThread(self.conn, frame_ids, payload, self.send_frame_dlg.interval_sb.value(), self.on_send_frame_thd_exception, self.on_sent_frame, self.on_send_frame_thd_exit)
                self.send_frame_thread.start()
                self.send_frame_btn.setText('Stop Sending')
            else:
                for frame_id in frame_ids:
                    self.conn.send_frame(frame_id, payload)
        else:
            self.send_frame_thread.stop()
            self.send_frame_thread = None
            self.send_frame_btn.setText('Stopping send...')
            self.send_frame_btn.setEnabled(False)
            self.send_frame_dlg = None
            self.tx_bps = self.tx_bytes_within_this_second = 0

    def on_recv_pkt_thd_exception(self, e):
        self.add_log(datetime.now().strftime('%H:%M:%S.%f')[:-3] + ' '+ e)

    def on_joined_cluster(self):
        self.status_label_right.setText('   Connected   ')
        self.status_label_right.setStyleSheet(self.connected_text_style)
        self.timer.start(1000)
        self.add_log(datetime.now().strftime('%H:%M:%S.%f')[:-3] + ' Joined into cluster, sniffing on FlexRay bus...')
        self.send_frame_btn.setEnabled(True)

    def on_join_cluster_failed(self):
        self.status_label_right.setText('   Join cluster failed   ')
        self.status_label_right.setStyleSheet(self.error_text_style)
        self.add_log(datetime.now().strftime('%H:%M:%S.%f')[:-3] + 'Join cluster failed, please check the configuration')

    def on_disconnected_from_cluster(self):
        self.status_label_right.setText('   Disconnected   ')
        self.status_label_right.setStyleSheet(self.error_text_style)
        if self.send_frame_thread:
            self.start_or_stop_send_frame_thd()
        self.send_frame_btn.setEnabled(False)
        self.add_log(datetime.now().strftime('%H:%M:%S.%f')[:-3] + ' Disconnected from cluster, please check the FlexRay cable')

    def on_fatal_error(self):
        self.status_label_right.setText('   Fatal Error   ')
        self.status_label_right.setStyleSheet(self.error_text_style)
        if self.send_frame_thread:
            self.start_or_stop_send_frame_thd()
        self.send_frame_btn.setEnabled(False)
        self.add_log(datetime.now().strftime('%H:%M:%S.%f')[:-3] + ' FlexRay fatal error')

    def on_send_frame_thd_exception(self, e):
        self.add_log(datetime.now().strftime('%H:%M:%S.%f')[:-3] + ' ' + e)

    def on_send_frame_thd_exit(self):
        self.send_frame_btn.setText('Send Frame')
        self.send_frame_btn.setEnabled(True)

    def on_frame_received(self, frame_id, payload_hex, payload_len):
        row_count = self.frame_table.rowCount()
        self.frame_table.setRowCount(row_count + 1)
        self.frame_table.setItem(row_count, 0, QTableWidgetItem(datetime.now().strftime('%H:%M:%S.%f')[:-3]))
        self.frame_table.setItem(row_count, 1, QTableWidgetItem(frame_id))
        self.frame_table.setItem(row_count, 2, QTableWidgetItem(str(payload_len)))
        self.frame_table.setItem(row_count, 3, QTableWidgetItem(payload_hex))
        self.rx_frames += 1
        self.rx_frames_gb.setTitle('Rx Frames ({})'.format(self.rx_frames))
        self.rx_bytes += payload_len
        self.rx_bytes_within_this_second += payload_len

    def on_status_data(self, text):
        self.detail_status.setText(text)

    def clear_rx_frame_table(self):
        self.frame_table.clearContents()
        self.frame_table.setRowCount(0)
        self.rx_frames = 0
        self.rx_bytes = 0
        self.rx_frames_gb.setTitle('Rx Frames ({})'.format(self.rx_frames))

    def on_sent_frame(self, payload_len):
        self.tx_frames += 1
        self.tx_bytes += payload_len
        self.tx_bytes_within_this_second += payload_len

    def update_statistics_label(self):
        if self.tx_bps > 1000:
            tx_speed_text = '{}K'.format(int(self.tx_bps / 1000.0))
        else:
            tx_speed_text = str(self.tx_bps)
        if self.rx_bps > 1000:
            rx_speed_text = '{}K'.format(int(self.rx_bps / 1000.0))
        else:
            rx_speed_text = str(self.tx_bps)
        self.statistics_label.setText(
            'Tx Frames: {}\nTx bytes: {}\nTx Speed: {}bps\nRx frames: {}\nRx bytes: {}\nRx Speed: {}bps'.format(
                self.tx_frames, self.tx_bytes, tx_speed_text, self.rx_frames, self.rx_bytes, rx_speed_text))

    def on_timer(self):
        self.tx_bps = self.tx_bytes_within_this_second * 8
        self.tx_time = time.time()
        self.tx_bytes_within_this_second = 0
        self.rx_bps = self.rx_bytes_within_this_second * 8
        self.rx_time = time.time()
        self.rx_bytes_within_this_second = 0
        self.update_statistics_label()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FlexRayGUI()
    sys.exit(app.exec())
