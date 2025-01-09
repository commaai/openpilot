class PIDController {
private:
  float kp;
  float ki;
  float kd;

  float prev_error;
  float derv_error;
  float inte_error;

  void update_error(float error) {
    this->inte_error += error;
    this->derv_error = error - this->prev_error;
    this->prev_error = error;
  }

public:
  PIDController(float kp, float ki, float kd)
      : kp(kp), ki(ki), kd(kd), prev_error(0), derv_error(0), inte_error(0) {}

  float compute(float error) {
    this->update_error(error);
    return (this->kp * this->prev_error + this->ki * this->inte_error +
            this->kd * this->derv_error);
  }

  void reset() {
    prev_error = 0;
    derv_error = 0;
    inte_error = 0;
  }
};
