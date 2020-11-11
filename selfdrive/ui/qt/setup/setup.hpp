#include <QWidget>
#include <QStackedLayout>

class Setup : public QWidget {
  Q_OBJECT

public:
  explicit Setup(QWidget *parent = 0);

private:
  QStackedLayout *layout;

  QWidget *getting_started();
  QWidget *network_setup();
  QWidget *software_selection();
  QWidget *downloading();

public slots:
  void nextPage();
};
