from messaging_pyx import Context, SubSocket, PubSocket

if __name__ == "__main__":
  c = Context()
  pub_sock = PubSocket()
  pub_sock.connect(c, "controlsState")

  for i in range(int(1e10)):
    print(i)
    sub_sock = SubSocket()
    sub_sock.connect(c, "controlsState")

    pub_sock.send(b'a')
    print(sub_sock.receive())
