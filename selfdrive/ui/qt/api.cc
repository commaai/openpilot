#include <QWidget>
#include <QFile>

#include "api.hpp"

#if defined(QCOM) || defined(QCOM2)
const std::string private_key_path = "/persist/comma/id_rsa";
#else
const std::string private_key_path = util::getenv_default("HOME", "/.comma/persist/comma/id_rsa", "/persist/comma/id_rsa");
#endif

QByteArray CommaApi::rsa_sign(QByteArray data) {
  auto file = QFile(private_key_path.c_str());
  bool r = file.open(QIODevice::ReadOnly);
  assert(r);

  auto key = file.readAll();

  BIO *mem = BIO_new_mem_buf(key.data(), key.size());
  assert(mem);

  RSA *rsa_private = PEM_read_bio_RSAPrivateKey(mem, NULL, NULL, NULL);
  assert(rsa_private);

  auto sig = QByteArray();
  sig.resize(RSA_size(rsa_private));

  unsigned int sig_len;
  int ret = RSA_sign(NID_sha256, (unsigned char*)data.data(), data.size(), (unsigned char*)sig.data(), &sig_len, rsa_private);

  assert(ret == 1);
  assert(sig_len == sig.size());

  BIO_free(mem);
  RSA_free(rsa_private);

  return sig;
}

