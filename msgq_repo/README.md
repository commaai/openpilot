# MSGQ: A lock free single producer multi consumer message queue

## What is this library?
MSGQ is a generic high performance IPC pub sub system with a single publisher and multiple subscribers. MSGQ is designed to be a high performance replacement for ZMQ-like SUB/PUB patterns. It uses a ring buffer in shared memory to efficiently read and write data. Each read requires a copy. Writing can be done without a copy, as long as the size of the data is known in advance. While MSGQ is the core of this library, this library also allows replacing the MSGQ backend with ZMQ or a spoofed implementation that can be used for deterministic testing. This library also contains visionipc, an IPC system specifically for large contiguous buffers (like images/video).

## Storage
The storage for the queue consists of an area of metadata, and the actual buffer. The metadata contains:

1. A counter to the number of readers that are active
2. A pointer to the head of the queue for writing. From now on referred to as *write pointer*
3. A cycle counter for the writer. This counter is incremented when the writer wraps around
4. N pointers, pointing to the current read position for all the readers. From now on referred to as *read pointer*
5. N counters,  counting the number of cycles for all the readers
6. N booleans, indicating validity for all the readers. From now on referred to as *validity flag*

The counter and the pointer are both 32 bit values, packed into 64 bit so they can be read and written atomically.

The data buffer is a ring buffer. All messages are prefixed by an 8 byte size field, followed by the data. A size of -1 indicates a wrap-around, and means the next message is stored at the beginning of the buffer.


## Writing
Writing involves the following steps:

1. Check if the area that is to be written overlaps with any of the read pointers, mark those readers as invalid by clearing the validity flag.
2. Write the message
3. Increase the write pointer by the size of the message

In case there is not enough space at the end of the buffer, a special empty message with a prefix of -1 is written. The cycle counter is incremented by one. In this case step 1 will check there are no read pointers pointing to the remainder of the buffer. Then another write cycle will start with the actual message.

There always needs to be 8 bytes of empty space at the end of the buffer. By doing this there is always space to write the -1.

## Reset reader
When the reader is lagging too much behind the read pointer becomes invalid and no longer points to the beginning of a valid message. To reset a reader to the current write pointer, the following steps are performed:

1. Set valid flag
2. Set read cycle counter to that of the writer
3. Set read pointer to write pointer

## Reading
Reading involves the following steps:

1. Read the size field at the current read pointer
2. Read the validity flag
3. Copy the data out of the buffer
4. Increase the read pointer by the size of the message
5. Check the validity flag again

Before starting the copy, the valid flag is checked. This is to prevent a race condition where the size prefix was invalid, and the read could read outside of the buffer. Make sure that step 1 and 2 are not reordered by your compiler or CPU.

If a writer overwrites the data while it's being copied out, the data will be invalid. Therefore the validity flag is also checked after reading it. The order of step 4 and 5 does not matter.

If at steps 2 or 5 the validity flag is not set, the reader is reset. Any data that was already read is discarded. After the reader is reset, the reading starts from the beginning.

If a message with size -1 is encountered, step 3 and 4 are replaced by increasing the cycle counter and setting the read pointer to the beginning of the buffer. After that another read is performed.
