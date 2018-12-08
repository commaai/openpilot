#!/usr/bin/env python
'''
	Flexrayd test app, write FlexRay config db file, send/receive FlexRay frames to/from zmq
'''
import yaml
import threading
import zmq
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from common.params import Params
from selfdrive.services import service_list
import selfdrive.messaging as messaging
from flexray_adapter.python.flexray_config import (config_to_c_struct, map_frame_id_to_tx_msg_buf_idx)


def build_send_flexray_capnp(frame_id, payload):
	dat = messaging.new_message()
	dat.init('sendFlexRay', 1)
	dat.sendFlexRay[0].frameId = frame_id
	dat.sendFlexRay[0].dat = payload
	return dat

class recv_flexray_thd(threading.Thread):
	def __init__(self):
			super(recv_flexray_thd, self).__init__()
			self._stop_event = threading.Event()

	def stop(self):
			self._stop_event.set()

	def stopped(self):
			return self._stop_event.is_set()

	def run(self):
		print 'Listening for FlexRay frames'
		try:
			context = zmq.Context()
			log_flexray = messaging.sub_sock(context, service_list['flexRay'].port)
			frames_count = 0
			while not self.stopped():
				for a in messaging.drain_sock(log_flexray):
					for f in a.flexRay:
						print 'Frame', frames_count, ', ID:', f.frameId, ', Payload Len:', len(f.dat), [hex(ord(x)) for x in f.dat]
						frames_count += 1
		finally:
			pass
			
def print_slot_ids(config):
	print 'Tx Static Slot ids: ', [f_id for f_id in frame_id_to_tx_msg_buf_idx.keys() if f_id <= config['gNumberOfStaticSlots']]
	print 'Tx Dynamic Slot ids: ', [f_id for f_id in frame_id_to_tx_msg_buf_idx.keys() if f_id > config['gNumberOfStaticSlots']]

if __name__ == "__main__":
	if len(sys.argv) > 1:
		file_path = sys.argv[1]
		if not os.path.exists(file_path):
			print('File {} not exist').format(file_path)
			sys.exit(-1)
	else:
		file_path = '../../flexray_adapter/python/test/node1.yml'
	# Convert YML to c struct, then write config db file for flexrayd.
	print('Loading FlexRay config file: {}...'.format(file_path))
	with open(file_path, 'r') as f:
		config = yaml.load(f)
		# Avoid performance lost by ensuring logging of protcol status data is disabled.
		config['LOG_STATUS_DATA'] = 0
		params = Params()
		params.put('FlexRayParams', config_to_c_struct(config))
		frame_id_to_tx_msg_buf_idx = map_frame_id_to_tx_msg_buf_idx(config)
	print_slot_ids(config)
	recv_thd = recv_flexray_thd()
	recv_thd.start()
	context = zmq.Context()
	send_flexray = messaging.pub_sock(context, service_list['sendFlexRay'].port)
	try:
		while True:
			text = raw_input("Type a tx slot id to send a frame, \'q\' to exit, \'h\' to show tx slot ids:")
			if text.isdigit():
				if int(text) not in frame_id_to_tx_msg_buf_idx:
					print 'Invalid Tx slot id:', text
				else:
					print('Start sending on slot id {0}...'.format(text))
					send_flexray.send(build_send_flexray_capnp(frame_id_to_tx_msg_buf_idx[int(text)], b'abcdefgh').to_bytes())
			elif text == 'q':
				break
			elif text == 'h':
				print_slot_ids(config)
	finally:
		recv_thd.stop()
		recv_thd.join()
		pass
