# mimic a python serial port
class PandaSerial(object):
  def __init__(self, panda, port, baud):
    self.panda = panda
    self.port = port
    self.panda.set_uart_parity(self.port, 0)
    self.panda.set_uart_baud(self.port, baud)
    self.buf = b""

  def read(self, l=1):
    tt = self.panda.serial_read(self.port)
    if len(tt) > 0:
      #print "R: ", tt.encode("hex")
      self.buf += tt
    ret = self.buf[0:l]
    self.buf = self.buf[l:]
    return ret

  def write(self, dat):
    #print "W: ", dat.encode("hex")
    #print '  pigeon_send("' + ''.join(map(lambda x: "\\x%02X" % ord(x), dat)) + '");'
    return self.panda.serial_write(self.port, dat)

  def close(self):
    pass


