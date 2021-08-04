#include <QLabel>
#include <QPushButton>
#include <QWidget>

class Reset : public QWidget {
  Q_OBJECT

public:
  explicit Reset(bool recover = false, QWidget *parent = 0);

private:
  QLabel *body;
  QPushButton *rejectBtn;
  QPushButton *rebootBtn;
  QPushButton *confirmBtn;
  void doReset();

private slots:
  void confirm();
};
