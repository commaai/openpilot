
import struct, sys, os

#-----------------------------------------------------------
#--- Block recoverer program for bzip2                    --
#---                                      bzip2recover.py --
#-----------------------------------------------------------

#  This program is bzip2recover, a program to attempt data 
#  salvage from damaged files.

#  Copyright (C) 1996-2002 Julian R Seward.
#  (C) 2014 Tim Sheerman-Chase. All rights reserved.

#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:

#  1. Redistributions of source code must retain the above copyright
#	  notice, this list of conditions and the following disclaimer.

#  2. The origin of this software must not be misrepresented; you must 
#	  not claim that you wrote the original software.  If you use this 
#	  software in a product, an acknowledgment in the product 
#	  documentation would be appreciated but is not required.

#  3. Altered source versions must be plainly marked as such, and must
#	  not be misrepresented as being the original software.

#  4. The name of the author may not be used to endorse or promote 
#	  products derived from this software without specific prior written 
#	  permission.

#  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
#  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
#  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#  Julian Seward, Cambridge, UK.
#  jseward@acm.org
#  bzip2/libbzip2 version 1.0 of 21 March 2000

#  This program is a complete hack and should be rewritten
#  properly.  It isn't very complicated.

# Source: https://gist.github.com/TimSC/8579121

#---------------------------------------------------
#--- Header bytes                                ---
#---------------------------------------------------

BZ_HDR_B = "\x42"								 # 'B' 
BZ_HDR_Z = "\x5a"								 # 'Z' 
BZ_HDR_h = "\x68"								 # 'h' 
BZ_HDR_0 = "\x30"								 # '0'

#---------------------------------------------------
#--- Bit stream I/O                              ---
#---------------------------------------------------

class BitStream(object):

    def __init__(self, stream, mode = 'r'):
        self.handle = stream
        self.buffer = 0
        self.buffLive = 0
        self.mode = mode
        self.bytesOut = 0

    def __del__(self):
        if self.mode == 'w':
            while self.buffLive < 8:
                self.buffLive += 1
                self.buffer = self.buffer << 1

            self.handle.write(chr(self.buffer))
            self.bytesOut += 1
            self.handle.flush()

            self.handle.close()
 
    def putBit (self, bit):

        if self.buffLive == 8:
            self.handle.write(chr(self.buffer))
            self.bytesOut += 1
            self.buffLive = 1
            self.buffer = bool(bit) & 0x1
        else:
            self.buffer = (self.buffer << 1) | (bool(bit) & 0x1)
            self.buffLive += 1

    def getBit (self):

        if self.buffLive > 0:
            self.buffLive -= 1
            return ((self.buffer) >> (self.buffLive)) & 0x1
        else:
            retVal = self.handle.read(1)
            if len(retVal)==0:
                return 2

            self.buffLive = 7
            self.buffer = ord(retVal[0])
            return bool(((self.buffer) >> 7) & 0x1)

    def putUChar(self, c):
        for i in range(7, -1, -1):
            b = (ord(c) >> i) & 0x1
            self.putBit(b)

    def putUInt32(self, c):
        cbin = struct.pack(">I", c)
        for ch in cbin:
            self.putUChar(ch)

#---------------------------------------------------
#---                                             ---
#---------------------------------------------------

BLOCK_HEADER_HI = 0x00003141
BLOCK_HEADER_LO = 0x59265359

BLOCK_ENDMARK_HI = 0x00001772
BLOCK_ENDMARK_LO = 0x45385090

def recover(inFileName):

    inFileName = None
    outFileName = None
    progName = None
    progName = sys.argv[0]

    bStart = []
    bEnd = []
    rbStart = []
    rbEnd = []

    print("{0} 1.0.0: extracts blocks from damaged .bz2 files.".format(progName))

    # if len(sys.argv) != 2:
    # 	print "{0}: usage is `{1} damaged_file_name'.".format(progName, progName)
    #	exit(1)

    # inFileName = sys.argv[1]
    inFile = open(inFileName, "rb")
    bsIn = BitStream(inFile)
    print("{0}: searching for block boundaries ...".format(progName))

    bitsRead = 0
    buffHi, buffLo = 0, 0
    currBlock = 0
    bStart.append(0)
    rbCtr = 0

    while True:
        b = bsIn.getBit()
        bitsRead += 1
        if b == 2:
            if bitsRead >= bStart[currBlock] and (bitsRead - bStart[currBlock]) >= 40:
                bEnd[currBlock] = bitsRead-1
                if currBlock > 0:
                    print("block {0} runs from {1} to {2} (incomplete)".format(currBlock, bStart[currBlock], bEnd[currBlock]))
            else:
                currBlock -= 1
            break

        buffHi = (buffHi << 1) | (buffLo >> 31)
        buffHi = buffHi & 0xffffffff
        buffLo = (buffLo << 1) | (b & 1)
        buffLo = buffLo & 0xffffffff

        if ( (buffHi & 0x0000ffff) == BLOCK_HEADER_HI
                 and buffLo == BLOCK_HEADER_LO) or \
              ( (buffHi & 0x0000ffff) == BLOCK_ENDMARK_HI
                 and buffLo == BLOCK_ENDMARK_LO):

            if bitsRead > 49:
                bEnd.append(bitsRead-49)
            else:
                bEnd.append(0)

            if currBlock > 0 and (bEnd[currBlock] - bStart[currBlock]) >= 130:
                print(" block {0} runs from {1} to {2}".format(
                             rbCtr+1,  bStart[currBlock], bEnd[currBlock]))
                rbStart.append(bStart[currBlock])
                rbEnd.append(bEnd[currBlock])
                rbCtr += 1

            currBlock += 1

            bStart.append(bitsRead)

    del bsIn

    #-- identified blocks run from 1 to rbCtr inclusive. --
    if rbCtr < 1:
        print("{0}: sorry, I couldn't find any block boundaries.".format(progName))
        exit(1)

    print("{0}: splitting into blocks".format(progName))

    inFile = open(inFileName,"rb")
    bsIn = BitStream(inFile)

    #-- placate gcc's dataflow analyser
    blockCRC = 0
    bsWr = None

    bitsRead = 0
    outFile = None
    wrBlock = 0
    while True:
        b = bsIn.getBit()
        if b == 2:
            break
        if wrBlock >= len(rbStart):
            break

        buffHi = (buffHi << 1) | (buffLo >> 31)
        buffHi = buffHi & 0xffffffff
        buffLo = (buffLo << 1) | (b & 1)
        buffLo = buffLo & 0xffffffff

        if bitsRead == 47+rbStart[wrBlock]:
            blockCRC = (buffHi << 16) | (buffLo >> 16)

        if bsWr is not None and bitsRead >= rbStart[wrBlock] and bitsRead <= rbEnd[wrBlock]:
            bsWr.putBit(b)

        bitsRead += 1

        if bitsRead == rbEnd[wrBlock]+1:
            if outFile is not None:
                bsWr.putUChar('\x17')
                bsWr.putUChar ('\x72')
                bsWr.putUChar('\x45')
                bsWr.putUChar ('\x38')
                bsWr.putUChar('\x50')
                bsWr.putUChar ('\x90')
                bsWr.putUInt32(blockCRC & 0xffffffff)
                del bsWr
                bsWr = None

            if wrBlock >= rbCtr:
                break
            wrBlock += 1

        elif bitsRead == rbStart[wrBlock]:

            baseName = os.path.split(inFileName)[1]
            baseName = os.path.splitext(baseName)[0]
            outFileName = "rec{0:05d}{1}".format(wrBlock+1, baseName)
            if outFileName[-4:] != ".bz2":
                outFileName += ".bz2"

            outFile = open(outFileName, "wb")
            bsWr = BitStream(outFile,"w")
            bsWr.putUChar(BZ_HDR_B)
            bsWr.putUChar(BZ_HDR_Z)
            bsWr.putUChar(BZ_HDR_h)
            bsWr.putUChar(chr(ord(BZ_HDR_0) + 9))
            bsWr.putUChar('\x31')
            bsWr.putUChar('\x41')
            bsWr.putUChar('\x59')
            bsWr.putUChar('\x26')
            bsWr.putUChar('\x53')
            bsWr.putUChar('\x59')

    print("{0}: finished".format(progName))

#-----------------------------------------------------------
#--- end                                 bzip2recover.py ---
#-----------------------------------------------------------

