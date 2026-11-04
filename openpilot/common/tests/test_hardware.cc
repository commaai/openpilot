#include "common/hardware/comma/hardware.h"
#include "common/tests/native_test.h"

void test_ir_torch_brightness() {
  // the same current as the downstream kernel's led:torch_2, see hardware.py
  const std::pair<int, int> expected[] = {{0, 0}, {1, 11}, {5, 11}, {10, 22}, {33, 85}, {50, 128}, {99, 255}, {100, 255},
                                          {150, 255}, {-5, 0}};
  for (auto [percent, brightness] : expected) {
    REQUIRE(HardwareComma::ir_torch_brightness(percent) == brightness);
  }
}

int main() {
  return run_native_test([] {
    test_ir_torch_brightness();
  });
}
