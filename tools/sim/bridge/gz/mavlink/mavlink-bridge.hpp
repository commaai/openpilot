#include <gz/msgs.hh>
#include <gz/transport.hh>
#include <mavsdk/connection_result.h>
#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <thread>

class MavlinkBridge{
    public:
        explicit MavlinkBridge();
        ~MavlinkBridge(){};
        void run();
        static std::vector<uint8_t> encode_image(const gz::msgs::Image &_msg);

    private:
        gz::msgs::Image last_frame;
        void on_image(const gz::msgs::Image &_msg);
        bool connect(const std::string);
        bool sub_camera(const std::string);
        void run_tcp_server(const uint16_t port);
        mavsdk::Mavsdk mavsdk_;
        std::optional<std::shared_ptr<mavsdk::System>> system_ = std::nullopt;
        std::shared_ptr<mavsdk::Action> action_;
        std::shared_ptr<mavsdk::Offboard> offboard_;
        std::shared_ptr<mavsdk::Telemetry> telemetry_;
        std::thread tcp_server_thread_;
        gz::transport::Node node_;
};
