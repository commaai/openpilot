#include <QLabel>
#include <QPushButton>
#include <QWidget>

enum ResetMode {
  USER_RESET, // user initiated a factory reset from openpilot
  RECOVER,    // userdata is corrupt for some reason, give a chance to recover
  FORMAT,     // finish up an ABL factory reset
};

class Reset : public QWidget {
  Q_OBJECT

public:
  explicit Reset(ResetMode mode, QWidget *parent = 0);

private:
  QLabel *body;
  QPushButton *rejectBtn;
  QPushButton *rebootBtn;
  QPushButton *confirmBtn;
  void doErase();
  void startReset();

private slots:
  void confirm();
};
