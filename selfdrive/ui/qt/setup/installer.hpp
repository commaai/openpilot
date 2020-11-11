#include <QWidget>
#include <QStackedLayout>

class Installer : public QWidget {
  Q_OBJECT

public:
  explicit Installer(QWidget *parent = 0);

public slots:
  int install();

signals:
  void done();
};
