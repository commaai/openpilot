#!/usr/bin/env python3
import cereal.messaging as messaging
import pcap

def main():
  ethernetData = messaging.pub_sock('ethernetData')

  for ts, pkt in pcap.pcap('eth0'):
    dat = messaging.new_message('ethernetData', 1)
    dat.ethernetData[0].ts = ts
    dat.ethernetData[0].pkt = str(pkt)
    ethernetData.send(dat.to_bytes())

if __name__ == "__main__":
  main()

