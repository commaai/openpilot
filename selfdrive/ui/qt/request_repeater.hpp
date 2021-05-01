#include "api.hpp"
#include "ui.hpp"

class RequestRepeater : public HttpRequest {
public:
  RequestRepeater(QObject *parent, const QString &requestURL, const QString &cacheKey = "", int period = 0);

private:
  QTimer *timer;
};
