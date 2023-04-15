#include "selfdrive/ui/qt/widgets/slider.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/offroad/settings.h"


CustomSlider::CustomSlider(const QString &param, 
                           CerealSetterFunction cerealSetFunc, 
                           const QString &unit,
                           const QString &title, 
                           double paramMin, 
                           double paramMax, 
                           double defaultVal, 
                           QWidget *parent) // Define the constructor
                          : // Call the base class constructor
                          QSlider(Qt::Horizontal, parent),
                          param(param), title(title), unit(unit),
                          paramMin(paramMin), paramMax(paramMax), defaultVal(defaultVal),
                          cerealSetFunc(cerealSetFunc) // Initialize the setter function
                          {
                            initialize(); // Call the initialize function
                          } // End of constructor

void CustomSlider::initialize()
{
  // Create UI elements
  sliderItem = new QWidget(parentWidget()); // Create a new widget
  // Create a vertical layout to stack the title and reset button on top of the slider
  QVBoxLayout *mainLayout = new QVBoxLayout(sliderItem); 
  // Create a horizontal layout to put the title and reset on left and right respectively
  QHBoxLayout *titleLayout = new QHBoxLayout();
  mainLayout->addLayout(titleLayout);

  // Create the title label
  label = new QLabel(title);
  label->setStyleSheet(LabelStyle);
  label->setTextFormat(Qt::RichText);
  titleLayout->addWidget(label, 0, Qt::AlignLeft);

  // Create the reset button
  ButtonControl *resetButton = new ButtonControl("  ", tr("RESET"));
  titleLayout->addWidget(resetButton, 0, Qt::AlignRight);
  // Connect the reset button to set the slider value to the default value
  connect(resetButton, &ButtonControl::clicked, [&]() {
    if (ConfirmationDialog::confirm(tr("Are you sure you want to reset ") + param + "?", tr("Reset"), this)) {
      this->setValue(sliderMin + (defaultVal - paramMin) / (paramMax - paramMin) * (sliderRange));
    } 
  });
  
  // slider settings
  setFixedHeight(100);
  setMinimum(sliderMin);
  setMaximum(sliderMax);

  // Set the default value of the slider to begin with
  setValue(sliderMin + (defaultVal - paramMin) / (paramMax - paramMin) * (sliderRange));
  label->setText(title + " " + QString::number(defaultVal, 'f', 2) + " " + unit);

  try // Try to get the value of the param from params. If it doesn't exist, catch the error
  {
    QString valueStr = QString::fromStdString(Params().get(param.toStdString()));
    double value = QString(valueStr).toDouble();
    // Set the value of the param in the behavior struct
    MessageBuilder msg;
    auto behavior = msg.initEvent().initBehavior();
    cerealSetFunc(behavior, value); 

    setValue(sliderMin + (value - paramMin) / (paramMax - paramMin) * (sliderRange)); // Set the value of the slider. The value is scaled to the slider range
    label->setText(title + " " + QString::number(value, 'f', 2) + " " + unit);
    
    // Set the slider to be enabled or disabled depending on the lock status
    bool locked = Params().getBool((param + "Lock").toStdString());
    setEnabled(!locked);
    setStyleSheet(locked ? lockedSliderStyle : SliderStyle);
    label->setStyleSheet(locked ? lockedLabelStyle : LabelStyle);

  }
  catch (const std::invalid_argument &e)
  {
    // Handle the error, e.g. lock the slider and display an error message as the label
    setValue(0);
    label->setText(title + "Error: Param not found. Add param to behaviord");
    setEnabled(false);
    setStyleSheet(lockedSliderStyle);
  }

  mainLayout->addWidget(this);

  connect(this, &CustomSlider::valueChanged, [=](int value)
  {
    // Update the label as the slider is moved. Don't save the value to params here
    double dValue = paramMin + (paramMax - paramMin) * (value - sliderMin) / (sliderRange);
    label->setText(title + " " + QString::number(dValue, 'f', 2) + " " + unit); 
    
  });

  connect(this, &CustomSlider::sliderReleasedWithValue, [this]() {
    // Call the sendAllSliderValues method from the BehaviorPanel
    auto parentBehaviorPanel = qobject_cast<BehaviorPanel *>(this->parentWidget());
    if (parentBehaviorPanel)
    {
      parentBehaviorPanel->sendAllSliderValues();
    }
  });


}
