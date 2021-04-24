#include "api.hpp"
#include "home.hpp"

class RequestRepeater : public HttpRequest {

public:
  RequestRepeater(QObject *parent, QString requestURL, const QString &cache_key = "", int period_seconds = 0, bool disableWithScreen = true);
  bool disableWithScreen;
};
