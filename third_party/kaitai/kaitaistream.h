#ifndef KAITAI_STREAM_H
#define KAITAI_STREAM_H

// Kaitai Struct runtime API version: x.y.z = 'xxxyyyzzz' decimal
#define KAITAI_STRUCT_VERSION 9000L

#include <istream>
#include <sstream>
#include <stdint.h>
#include <sys/types.h>

namespace kaitai {

/**
 * Kaitai Stream class (kaitai::kstream) is an implementation of
 * <a href="https://doc.kaitai.io/stream_api.html">Kaitai Struct stream API</a>
 * for C++/STL. It's implemented as a wrapper over generic STL std::istream.
 *
 * It provides a wide variety of simple methods to read (parse) binary
 * representations of primitive types, such as integer and floating
 * point numbers, byte arrays and strings, and also provides stream
 * positioning / navigation methods with unified cross-language and
 * cross-toolkit semantics.
 *
 * Typically, end users won't access Kaitai Stream class manually, but would
 * describe a binary structure format using .ksy language and then would use
 * Kaitai Struct compiler to generate source code in desired target language.
 * That code, in turn, would use this class and API to do the actual parsing
 * job.
 */
class kstream {
public:
    /**
     * Constructs new Kaitai Stream object, wrapping a given std::istream.
     * \param io istream object to use for this Kaitai Stream
     */
    kstream(std::istream* io);

    /**
     * Constructs new Kaitai Stream object, wrapping a given in-memory data
     * buffer.
     * \param data data buffer to use for this Kaitai Stream
     */
    kstream(std::string& data);

    void close();

    /** @name Stream positioning */
    //@{
    /**
     * Check if stream pointer is at the end of stream. Note that the semantics
     * are different from traditional STL semantics: one does *not* need to do a
     * read (which will fail) after the actual end of the stream to trigger EOF
     * flag, which can be accessed after that read. It is sufficient to just be
     * at the end of the stream for this method to return true.
     * \return "true" if we are located at the end of the stream.
     */
    bool is_eof() const;

    /**
     * Set stream pointer to designated position.
     * \param pos new position (offset in bytes from the beginning of the stream)
     */
    void seek(uint64_t pos);

    /**
     * Get current position of a stream pointer.
     * \return pointer position, number of bytes from the beginning of the stream
     */
    uint64_t pos();

    /**
     * Get total size of the stream in bytes.
     * \return size of the stream in bytes
     */
    uint64_t size();
    //@}

    /** @name Integer numbers */
    //@{

    // ------------------------------------------------------------------------
    // Signed
    // ------------------------------------------------------------------------

    int8_t read_s1();

    // ........................................................................
    // Big-endian
    // ........................................................................

    int16_t read_s2be();
    int32_t read_s4be();
    int64_t read_s8be();

    // ........................................................................
    // Little-endian
    // ........................................................................

    int16_t read_s2le();
    int32_t read_s4le();
    int64_t read_s8le();

    // ------------------------------------------------------------------------
    // Unsigned
    // ------------------------------------------------------------------------

    uint8_t read_u1();

    // ........................................................................
    // Big-endian
    // ........................................................................

    uint16_t read_u2be();
    uint32_t read_u4be();
    uint64_t read_u8be();

    // ........................................................................
    // Little-endian
    // ........................................................................

    uint16_t read_u2le();
    uint32_t read_u4le();
    uint64_t read_u8le();

    //@}

    /** @name Floating point numbers */
    //@{

    // ........................................................................
    // Big-endian
    // ........................................................................

    float read_f4be();
    double read_f8be();

    // ........................................................................
    // Little-endian
    // ........................................................................

    float read_f4le();
    double read_f8le();

    //@}

    /** @name Unaligned bit values */
    //@{

    void align_to_byte();
    uint64_t read_bits_int_be(int n);
    uint64_t read_bits_int(int n);
    uint64_t read_bits_int_le(int n);

    //@}

    /** @name Byte arrays */
    //@{

    std::string read_bytes(std::streamsize len);
    std::string read_bytes_full();
    std::string read_bytes_term(char term, bool include, bool consume, bool eos_error);
    std::string ensure_fixed_contents(std::string expected);

    static std::string bytes_strip_right(std::string src, char pad_byte);
    static std::string bytes_terminate(std::string src, char term, bool include);
    static std::string bytes_to_str(std::string src, std::string src_enc);

    //@}

    /** @name Byte array processing */
    //@{

    /**
     * Performs a XOR processing with given data, XORing every byte of input with a single
     * given value.
     * @param data data to process
     * @param key value to XOR with
     * @return processed data
     */
    static std::string process_xor_one(std::string data, uint8_t key);

    /**
     * Performs a XOR processing with given data, XORing every byte of input with a key
     * array, repeating key array many times, if necessary (i.e. if data array is longer
     * than key array).
     * @param data data to process
     * @param key array of bytes to XOR with
     * @return processed data
     */
    static std::string process_xor_many(std::string data, std::string key);

    /**
     * Performs a circular left rotation shift for a given buffer by a given amount of bits,
     * using groups of 1 bytes each time. Right circular rotation should be performed
     * using this procedure with corrected amount.
     * @param data source data to process
     * @param amount number of bits to shift by
     * @return copy of source array with requested shift applied
     */
    static std::string process_rotate_left(std::string data, int amount);

#ifdef KS_ZLIB
    /**
     * Performs an unpacking ("inflation") of zlib-compressed data with usual zlib headers.
     * @param data data to unpack
     * @return unpacked data
     * @throws IOException
     */
    static std::string process_zlib(std::string data);
#endif

    //@}

    /**
     * Performs modulo operation between two integers: dividend `a`
     * and divisor `b`. Divisor `b` is expected to be positive. The
     * result is always 0 <= x <= b - 1.
     */
    static int mod(int a, int b);

    /**
     * Converts given integer `val` to a decimal string representation.
     * Should be used in place of std::to_string() (which is available only
     * since C++11) in older C++ implementations.
     */
    static std::string to_string(int val);

    /**
     * Reverses given string `val`, so that the first character becomes the
     * last and the last one becomes the first. This should be used to avoid
     * the need of local variables at the caller.
     */
    static std::string reverse(std::string val);

    /**
     * Finds the minimal byte in a byte array, treating bytes as
     * unsigned values.
     * @param val byte array to scan
     * @return minimal byte in byte array as integer
     */
    static uint8_t byte_array_min(const std::string val);

    /**
     * Finds the maximal byte in a byte array, treating bytes as
     * unsigned values.
     * @param val byte array to scan
     * @return maximal byte in byte array as integer
     */
    static uint8_t byte_array_max(const std::string val);

private:
    std::istream* m_io;
    std::istringstream m_io_str;
    int m_bits_left;
    uint64_t m_bits;

    void init();
    void exceptions_enable() const;

    static uint64_t get_mask_ones(int n);

    static const int ZLIB_BUF_SIZE = 128 * 1024;
};

}

#endif
