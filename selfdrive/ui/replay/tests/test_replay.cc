#include <QCryptographicHash>
#include <QString>
#include <future>

#include "catch2/catch.hpp"
#include "selfdrive/common/util.h"
#include "selfdrive/ui/replay/framereader.h"
#include "selfdrive/ui/replay/route.h"

const char *stream_url = "https://commadataci.blob.core.windows.net/openpilotci/0c94aa1e1296d7c6/2021-05-05--19-48-37/0/fcamera.hevc";

// TEST_CASE("FrameReader") {
//   SECTION("process&get") {
//     FrameReader fr;
//     REQUIRE(fr.load(stream_url) == true);
//     REQUIRE(fr.valid() == true);
//     REQUIRE(fr.getFrameCount() == 1200);
//     // random get 50 frames
//     // srand(time(NULL));
//     // for (int i = 0; i < 50; ++i) {
//     //   int idx = rand() % (fr.getFrameCount() - 1);
//     //   REQUIRE(fr.get(idx) != nullptr);
//     // }
//     // sequence get 50 frames {
//     for (int i = 0; i < 50; ++i) {
//       REQUIRE(fr.get(i) != nullptr);
//     }
//   }
// }

std::string sha_256(const QString &dat) {
  return QString(QCryptographicHash::hash(dat.toUtf8(), QCryptographicHash::Sha256).toHex()).toStdString();
}

TEST_CASE("httpMultiPartDownload") {
  char filename[] = "/tmp/XXXXXX";
  int fd = mkstemp(filename);
  REQUIRE(fd != -1);
  close(fd);

  SECTION("http 200") {
    REQUIRE(httpMultiPartDownload(stream_url, filename, 5));
    std::string content = util::read_file(filename);
    REQUIRE(content.size() == 37495242);
    std::string checksum = sha_256(QString::fromStdString(content));
    REQUIRE(checksum == "d8ff81560ce7ed6f16d5fb5a6d6dd13aba06c8080c62cfe768327914318744c4");
  }
  SECTION("http 404") {
    REQUIRE(httpMultiPartDownload(util::string_format("%s_abc", stream_url), filename, 5) == false);
  }
}
