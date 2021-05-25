#pragma once

#include <optional>
#include <vector>
#include <unordered_map>

#include <QElapsedTimer>
#include <QMultiMap>
#include <QNetworkAccessManager>
#include <QString>

#include <capnp/serialize.h>

class FileReader : public QObject {
  Q_OBJECT

public:
  FileReader(const QString& file_);
  void startRequest(const QUrl &url);
  ~FileReader();
  virtual void readyRead();
  void httpFinished();

public slots:
  void process();

protected:
  QNetworkReply *reply;

private:
  QNetworkAccessManager *qnam;
  QElapsedTimer timer;
  QString file;
};

typedef QMultiMap<uint64_t, capnp::FlatArrayMessageReader *> Events;
typedef std::unordered_map<int, std::pair<int, int> > EncodeIdxMap;

class LogReader : public FileReader {
  Q_OBJECT

public:
  LogReader(const QString &file);
  ~LogReader();
  bool ready() const { return ready_; }
  const Events &events() const { return events_; }
  std::optional<std::pair<int, int>> getFrameEncodeIdx(const std::string &type, uint32_t frame_id) const{
    if (auto edix_it = encoderIdx_.find(type); edix_it != encoderIdx_.end()) {
      if (auto it = edix_it->second.find(frame_id); it != edix_it->second.end()) {
        return it->second;
      }
    }
    return std::nullopt;

  }

signals:
  void done();

protected:
  void readyRead();
  void parseEvents(kj::ArrayPtr<const capnp::word> amsg);

  std::vector<uint8_t> raw_;
  Events events_;
  std::atomic<bool> ready_ = false;
  std::map<std::string, EncodeIdxMap> encoderIdx_;
};
