#pragma once

#include <curl/curl.h>
#include <string>

#include "common/util.h"
#include "third_party/json11/json11.hpp"

namespace CommaApi2 {

const std::string BASE_URL = util::getenv("API_HOST", "https://api.commadotai.com").c_str();
std::string create_token(bool use_jwt, const json11::Json& payloads = {}, int expiry = 3600);
std::string httpGet(const std::string &url, long *response_code = nullptr);

}  // namespace CommaApi2
