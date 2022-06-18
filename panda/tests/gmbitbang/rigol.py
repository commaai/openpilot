#!/usr/bin/env python3
# pylint: skip-file
# type: ignore
import numpy as np
import visa
import matplotlib.pyplot as plt

resources = visa.ResourceManager()
print(resources.list_resources())

scope = resources.open_resource('USB0::0x1AB1::0x04CE::DS1ZA184652242::INSTR', timeout=2000, chunk_size=1024000)
print(scope.query('*IDN?').strip())

#voltscale = scope.ask_for_values(':CHAN1:SCAL?')[0]
#voltoffset = scope.ask_for_values(":CHAN1:OFFS?")[0]

#scope.write(":STOP")
scope.write(":WAV:POIN:MODE RAW")
scope.write(":WAV:DATA? CHAN1")[10:]
rawdata = scope.read_raw()
data = np.frombuffer(rawdata, 'B')
print(data.shape)

s1 = data[0:650]
s2 = data[650:]
s1i = np.argmax(s1 > 100)
s2i = np.argmax(s2 > 100)
s1 = s1[s1i:]
s2 = s2[s2i:]

plt.plot(s1)
plt.plot(s2)
plt.show()
#data = (data - 130.0 - voltoffset/voltscale*25) / 25 * voltscale

print(data)
