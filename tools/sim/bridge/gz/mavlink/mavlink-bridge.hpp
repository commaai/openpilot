#include <gz/msgs.hh>
#include <gz/transport.hh>

class MavlinkBridge{
    public:
        explicit MavlinkBridge();
        ~MavlinkBridge(){};
        void run();

    private:
        static std::vector<uint8_t> encode_image(const gz::msgs::Image &_msg);
        std::vector<uint8_t> last_frame;
        void on_image(const gz::msgs::Image &_msg);
};
