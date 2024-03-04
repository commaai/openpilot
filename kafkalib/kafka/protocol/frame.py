class KafkaBytes(bytearray):
    def __init__(self, size):
        super(KafkaBytes, self).__init__(size)
        self._idx = 0

    def read(self, nbytes=None):
        if nbytes is None:
            nbytes = len(self) - self._idx
        start = self._idx
        self._idx += nbytes
        if self._idx > len(self):
            self._idx = len(self)
        return bytes(self[start:self._idx])

    def write(self, data):
        start = self._idx
        self._idx += len(data)
        self[start:self._idx] = data

    def seek(self, idx):
        self._idx = idx

    def tell(self):
        return self._idx

    def __str__(self):
        return 'KafkaBytes(%d)' % len(self)

    def __repr__(self):
        return str(self)
