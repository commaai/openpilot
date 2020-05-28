#!/usr/bin/env python
import sys
import random
import math

# Simple hacky Matroska generator
# Reads mp3 file "q.mp3" and jpeg images from img/0.jpg, img/1.jpg and so on and
# writes Matroska file with mjpeg and mp3 to stdout

# License=MIT

# unsigned
def big_endian_number(number):
    if(number<0x100):
        return chr(number)
    return big_endian_number(number>>8) + chr(number&0xFF)

ben=big_endian_number

def ebml_encode_number(number):
    def trailing_bits(rest_of_number, number_of_bits):
        # like big_endian_number, but can do padding zeroes
        if number_of_bits==8:
            return chr(rest_of_number&0xFF);
        else:
            return trailing_bits(rest_of_number>>8, number_of_bits-8) + chr(rest_of_number&0xFF)

    if number == -1:
        return chr(0xFF)
    if number < 2**7 - 1:
        return chr(number|0x80)
    if number < 2**14 - 1:
        return chr(0x40 | (number>>8)) + trailing_bits(number, 8)
    if number < 2**21 - 1:
        return chr(0x20 | (number>>16)) + trailing_bits(number, 16)
    if number < 2**28 - 1:
        return chr(0x10 | (number>>24)) + trailing_bits(number, 24)
    if number < 2**35 - 1:
        return chr(0x08 | (number>>32)) + trailing_bits(number, 32)
    if number < 2**42 - 1:
        return chr(0x04 | (number>>40)) + trailing_bits(number, 40)
    if number < 2**49 - 1:
        return chr(0x02 | (number>>48)) + trailing_bits(number, 48)
    if number < 2**56 - 1:
        return chr(0x01) + trailing_bits(number, 56)
    raise Exception("NUMBER TOO BIG")

def ebml_element(element_id, data, length=None):
    if length==None:
        length = len(data)
    return big_endian_number(element_id) + ebml_encode_number(length) + data


def write_ebml_header(f, content_type, version, read_version):
    f.write(
        ebml_element(0x1A45DFA3, "" # EBML
            + ebml_element(0x4286, ben(1))   # EBMLVersion
            + ebml_element(0x42F7, ben(1))   # EBMLReadVersion
            + ebml_element(0x42F2, ben(4))   # EBMLMaxIDLength
            + ebml_element(0x42F3, ben(8))   # EBMLMaxSizeLength
            + ebml_element(0x4282, content_type) # DocType
            + ebml_element(0x4287, ben(version))   # DocTypeVersion
            + ebml_element(0x4285, ben(read_version))   # DocTypeReadVersion
            ))

def write_infinite_segment_header(f):
    # write segment element header
    f.write(ebml_element(0x18538067,"",-1)) # Segment (unknown length)

def random_uid():
    def rint():
        return int(random.random()*(0x100**4))
    return ben(rint()) + ben(rint()) + ben(rint()) + ben(rint())


def example():
    write_ebml_header(sys.stdout, "matroska", 2, 2)
    write_infinite_segment_header(sys.stdout)


    # write segment info (optional)
    sys.stdout.write(ebml_element(0x1549A966, "" # SegmentInfo
        + ebml_element(0x73A4, random_uid()) # SegmentUID
        + ebml_element(0x7BA9, "mkvgen.py test") # Title
        + ebml_element(0x4D80, "mkvgen.py") # MuxingApp
        + ebml_element(0x5741, "mkvgen.py") # WritingApp
        ))

    # write trans data (codecs etc.)
    sys.stdout.write(ebml_element(0x1654AE6B, "" # Tracks
        + ebml_element(0xAE, "" # TrackEntry
            + ebml_element(0xD7, ben(1)) # TrackNumber
            + ebml_element(0x73C5, ben(0x77)) # TrackUID
            + ebml_element(0x83, ben(0x01)) # TrackType
                #0x01 track is a video track
                #0x02 track is an audio track
                #0x03 track is a complex track, i.e. a combined video and audio track
                #0x10 track is a logo track
                #0x11 track is a subtitle track
                #0x12 track is a button track
                #0x20 track is a control track
            + ebml_element(0x536E, "mjpeg data") # Name
            + ebml_element(0x86, "V_MJPEG") # CodecID
            #+ ebml_element(0x23E383, ben(100000000)) # DefaultDuration (opt.), nanoseconds
            #+ ebml_element(0x6DE7, ben(100)) # MinCache
            + ebml_element(0xE0, "" # Video
                + ebml_element(0xB0, ben(640)) # PixelWidth
                + ebml_element(0xBA, ben(480)) # PixelHeight
                )
            )
        + ebml_element(0xAE, "" # TrackEntry
            + ebml_element(0xD7, ben(2)) # TrackNumber
            + ebml_element(0x73C5, ben(0x78)) # TrackUID
            + ebml_element(0x83, ben(0x02)) # TrackType
                #0x01 track is a video track
                #0x02 track is an audio track
                #0x03 track is a complex track, i.e. a combined video and audio track
                #0x10 track is a logo track
                #0x11 track is a subtitle track
                #0x12 track is a button track
                #0x20 track is a control track
            + ebml_element(0x536E, "content of mp3 file") # Name
            #+ ebml_element(0x6DE7, ben(100)) # MinCache
            + ebml_element(0x86, "A_MPEG/L3") # CodecID
            #+ ebml_element(0xE1, "") # Audio
         )
        ))


    mp3file = open("q.mp3", "rb")
    mp3file.read(500000);

    def mp3framesgenerator(f):
        debt=""
        while True:
            for i in range(0,len(debt)+1):
                if i >= len(debt)-1:
                    debt = debt + f.read(8192)
                    break
                #sys.stderr.write("i="+str(i)+" len="+str(len(debt))+"\n")
                if ord(debt[i])==0xFF and (ord(debt[i+1]) & 0xF0)==0XF0 and i>700:
                    if i>0:
                        yield debt[0:i]
                        #   sys.stderr.write("len="+str(i)+"\n")
                        debt = debt[i:]
                        break


    mp3 = mp3framesgenerator(mp3file)
    next(mp3)


    for i in range(0,530):
        framefile = open("img/"+str(i)+".jpg", "rb")
        framedata = framefile.read()
        framefile.close()

        # write cluster (actual video data)

        if random.random()<1:
            sys.stdout.write(ebml_element(0x1F43B675, "" # Cluster
                + ebml_element(0xE7, ben(int(i*26*4))) # TimeCode, uint, milliseconds
                # + ebml_element(0xA7, ben(0)) # Position, uint
                + ebml_element(0xA3, "" # SimpleBlock
                    + ebml_encode_number(1) # track number
                    + chr(0x00) + chr(0x00) # timecode, relative to Cluster timecode, sint16, in milliseconds
                    + chr(0x00) # flags
                    + framedata
                    )))

        for u in range(0,4):
            mp3f=next(mp3)
            if random.random()<1:
                sys.stdout.write(ebml_element(0x1F43B675, "" # Cluster
                    + ebml_element(0xE7, ben(i*26*4+u*26)) # TimeCode, uint, milliseconds
                    + ebml_element(0xA3, "" # SimpleBlock
                        + ebml_encode_number(2) # track number
                        + chr(0x00) + chr(0x00) # timecode, relative to Cluster timecode, sint16, in milliseconds
                        + chr(0x00) # flags
                        + mp3f
                        )))





if __name__ == '__main__':
    example()
