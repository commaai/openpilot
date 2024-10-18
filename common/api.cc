
#include "common/api.h"

#include <openssl/pem.h>
#include <openssl/rsa.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "common/params.h"
#include "system/hardware/hw.h"

namespace CommaApi {

// Base64 URL-safe character set (uses '-' and '_' instead of '+' and '/')
static const std::string base64url_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789-_";

std::string base64url_encode(const std::string &in) {
  std::string out;
  int val = 0, valb = -6;
  for (unsigned char c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(base64url_chars[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) {
    out.push_back(base64url_chars[((val << 8) >> (valb + 8)) & 0x3F]);
  }

  return out;
}

RSA *get_rsa_private_key() {
  static std::unique_ptr<RSA, decltype(&RSA_free)> rsa_private(nullptr, RSA_free);
  if (!rsa_private) {
    FILE *fp = fopen(Path::rsa_file().c_str(), "rb");
    if (!fp) {
      std::cerr << "No RSA private key found, please run manager.py or registration.py" << std::endl;
      return nullptr;
    }
    rsa_private.reset(PEM_read_RSAPrivateKey(fp, NULL, NULL, NULL));
    fclose(fp);
  }
  return rsa_private.get();
}

std::string rsa_sign(const std::string &data) {
  RSA *rsa_private = get_rsa_private_key();
  if (!rsa_private) return {};

  std::vector<uint8_t> sig(RSA_size(rsa_private));
  unsigned int sig_len;
  int ret = RSA_sign(NID_sha256, (unsigned char *)data.data(), data.size(),
                     sig.data(), &sig_len, rsa_private);
  assert(ret == 1);
  assert(sig.size() == sig_len);
  return std::string(sig.begin(), sig.begin() + sig_len);
}

std::string create_jwt(const json11::Json &extra, int exp_time) {
  int now = std::chrono::seconds(std::time(nullptr)).count();
  std::string dongle_id = Params().get("DongleId");

  // Create header and payload
  json11::Json header = json11::Json::object{{"alg", "RS256"}};
  auto payload = json11::Json::object{
      {"identity", dongle_id},
      {"iat", now},
      {"nbf", now},
      {"exp", now + exp_time},
  };
  // Merge extra payload
  for (const auto &item : extra.object_items()) {
    payload[item.first] = item.second;
  }

  // JWT construction
  std::string jwt = base64url_encode(header.dump()) + '.' +
                    base64url_encode(json11::Json(payload).dump());

  // Hash and sign
  std::string hash(SHA256_DIGEST_LENGTH, '\0');
  SHA256((uint8_t *)jwt.data(), jwt.size(), (uint8_t *)hash.data());
  std::string signature = rsa_sign(hash);

  return jwt + "." + base64url_encode(signature);
}

std::string create_token(bool use_jwt, const json11::Json &payloads, int expiry) {
  if (use_jwt) {
    return create_jwt(payloads, expiry);
  }

  std::string token_json = util::read_file(util::getenv("HOME") + "/.comma/auth.json");
  std::string err;
  auto json = json11::Json::parse(token_json, err);
  if (!err.empty()) {
    std::cerr << "Error parsing auth.json " << err << std::endl;
    return "";
  }
  return json["access_token"].string_value();
}

}  // namespace CommaApi
