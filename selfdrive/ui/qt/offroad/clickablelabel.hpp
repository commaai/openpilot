#include <QLabel>
#include <QWidget>

class ClickableLabel : public QLabel {
  Q_OBJECT

public:
    explicit ClickableLabel(QWidget *parent = nullptr, int index = -1);

    void emphasize();
    void deemphasize();

signals:
    void selected(int);

protected:
    void enterEvent(QEvent *e) override;
    void leaveEvent(QEvent *e) override;
    void mousePressEvent(QMouseEvent *e) override;

private:
    int index;
};
