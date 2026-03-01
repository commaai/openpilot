
#include "tools/replay/api.h"

#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

#include <cassert>
#include <chrono>
#include <iostream>

#include "common/params.h"
#include "common/version.h"
#include "system/hardware/hw.h"

namespace CommaApi2 {

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

EVP_PKEY *get_rsa_private_key() {
  static std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> rsa_private(nullptr, EVP_PKEY_free);
  if (!rsa_private) {
    FILE *fp = fopen(Path::rsa_file().c_str(), "rb");
    if (!fp) {
      std::cerr << "No RSA private key found, please run manager.py or registration.py" << std::endl;
      return nullptr;
    }
    rsa_private.reset(PEM_read_PrivateKey(fp, NULL, NULL, NULL));
    fclose(fp);
  }
  return rsa_private.get();
}

std::string rsa_sign(const std::string &data) {
  EVP_PKEY *private_key = get_rsa_private_key();
  if (!private_key) return {};

  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  assert(mdctx != nullptr);

  std::vector<uint8_t> sig(EVP_PKEY_size(private_key));
  uint32_t sig_len;

  EVP_SignInit(mdctx, EVP_sha256());
  EVP_SignUpdate(mdctx, data.data(), data.size());
  int ret = EVP_SignFinal(mdctx, sig.data(), &sig_len, private_key);

  EVP_MD_CTX_free(mdctx);

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

std::string httpGet(const std::string &url, long *response_code) {
  CURL *curl = curl_easy_init();
  assert(curl);

  std::string readBuffer;
  const std::string token = CommaApi2::create_token(!Hardware::PC());

  // Set up the lambda for the write callback
  // The '+' makes the lambda non-capturing, allowing it to be used as a C function pointer
  auto writeCallback = +[](char *contents, size_t size, size_t nmemb, std::string *userp) ->size_t{
    size_t totalSize = size * nmemb;
    userp->append((char *)contents, totalSize);
    return totalSize;
  };

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  // Handle headers
  struct curl_slist *headers = nullptr;
  headers = curl_slist_append(headers, "User-Agent: openpilot-" COMMA_VERSION);
  if (!token.empty()) {
    headers = curl_slist_append(headers, ("Authorization: JWT " + token).c_str());
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

  CURLcode res = curl_easy_perform(curl);

  if (response_code) {
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, response_code);
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return res == CURLE_OK ? readBuffer : std::string{};
}

}  // namespace CommaApi
