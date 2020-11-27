#pragma once
#include <QtWidgets>

class Toggle : public QAbstractButton {
    Q_OBJECT
    Q_PROPERTY(int offset_circle READ offset_circle WRITE set_offset_circle)

public:
    Toggle(QWidget* parent = nullptr);
    void togglePosition();//Toggles the toggle

    int offset_circle() const {
        return _x_circle;
    }
    void set_offset_circle(int o) {
        _x_circle = o;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void enterEvent(QEvent*) override;

private:
    bool _on;
    int _x_circle, _y_circle, _padding_circle
    int _height, _radius;
    QPropertyAnimation *_anim = nullptr;
};
