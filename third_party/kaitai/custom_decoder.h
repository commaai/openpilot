#ifndef KAITAI_CUSTOM_DECODER_H
#define KAITAI_CUSTOM_DECODER_H

#include <string>

namespace kaitai {

class custom_decoder {
public:
    virtual ~custom_decoder() {};
    virtual std::string decode(std::string src) = 0;
};

}

#endif
