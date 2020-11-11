#include <QLabel>
#include <QWidget>

class ClickableLabel : public QLabel {
  Q_OBJECT

public:
    ClickableLabel(QWidget *parent = nullptr);

protected:
    void enterEvent(QEvent *e) override;
    void leaveEvent(QEvent *e) override;
    void mousePressEvent(QMouseEvent *e) override;
};
