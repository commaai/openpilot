#!/usr/bin/env python3
"""
Manual car selection tool.

Lets the user manually pick a vehicle platform instead of relying on the
automatic CAN fingerprinting done by opendbc's get_car().

Useful for regions (e.g. Middle East) where the local CAN firmware fingerprint
may not perfectly match the fingerprints shipped with opendbc, even though the
car is mechanically/electronically identical to a supported model.

The selection is stored in the persistent "ForceCarFingerprint" param, which
card.py reads on boot and feeds into get_car() as the cached carFingerprint.

Usage (on the device, over SSH or the on-screen terminal):
    python3 selfdrive/car/manual_car_setup.py

Run it after installing openpilot. Reboot for the change to take effect.
"""
import sys


MAZDA_MODELS = {
  "MAZDA_CX9_2021":       "Mazda CX-9 2021+ (Middle East / GCC supported)",
  "MAZDA_CX9_2021_DSE":   "Mazda CX-9 2021+ (Driver Support Edition)",
  "MAZDA_CX5_2017":       "Mazda CX-5 2017",
  "MAZDA_CX5_2022":       "Mazda CX-5 2022",
  "MAZDA_CX50_2023":      "Mazda CX-50 2023",
  "MAZDA_3_2019":         "Mazda 3 2019",
  "MAZDA_6_2017":         "Mazda 6 2017",
}

PARAM_KEY = "ForceCarFingerprint"


def get_params():
  from openpilot.common.params import Params
  return Params()


def discover_supported_cars():
  try:
    from opendbc.car.car_helpers import interfaces
    return sorted(interfaces.keys())
  except Exception:
    return None


def show_current(params):
  current = params.get(PARAM_KEY)
  if current:
    print(f"\n  الاختيار الحالي (manual): {current}")
  else:
    print("\n  لا يوجد اختيار يدوي — الكشف التلقائي مُفعّل (auto-detect)")


def main():
  print("=" * 60)
  print("  اختيار السيارة يدوياً — Manual Car Selection")
  print("=" * 60)

  try:
    params = get_params()
  except Exception as e:
    print(f"\nخطأ: لا يمكن الوصول إلى params ({e})")
    print("شغّل هذا السكربت على جهاز Comma (عبر SSH).")
    sys.exit(1)

  show_current(params)

  print("\n  --- موديلات Mazda ---")
  mazda_items = list(MAZDA_MODELS.items())
  for i, (fp, desc) in enumerate(mazda_items, start=1):
    print(f"  {i:>2}. {desc}  [{fp}]")

  print("\n   0. كشف تلقائي (auto-detect) — حذف الاختيار اليدوي")
  print("  99. إدخال بصمة مخصّصة (custom fingerprint)")
  print("  q. خروج بدون تغيير\n")

  choice = input("  اختر رقم الموديل: ").strip()

  if choice.lower() in ("q", "quit", "exit"):
    print("  لم يتم إجراء أي تغيير.")
    return

  if choice == "0":
    params.remove(PARAM_KEY)
    print(f"\n  ✓ تم تفعيل الكشف التلقائي (auto-detect).")
    print("    أعد تشغيل الجهاز لتطبيق التغيير.")
    return

  if choice == "99":
    fp = input("  أدخل بصمة السيارة (fingerprint): ").strip()
    if not fp:
      print("  بصمة غير صحيحة. تم الإلغاء.")
      return
    supported = discover_supported_cars()
    if supported is not None and fp not in supported:
      print(f"  ⚠ تحذير: البصمة '{fp}' غير معروفة في opendbc.")
      confirm = input("  هل تريد المتابعة؟ (y/N): ").strip().lower()
      if confirm != "y":
        print("  تم الإلغاء.")
        return
  else:
    try:
      idx = int(choice) - 1
      fp = mazda_items[idx][0]
    except (ValueError, IndexError):
      print(f"\n  خيار غير صالح: {choice}")
      sys.exit(1)

  params.put(PARAM_KEY, fp, block=True)
  print(f"\n  ✓ تم حفظ الاختيار اليدوي: {fp}")
  print("    أعد تشغيل الجهاز (reboot) لتطبيق التغيير.")
  print(f"    card.py سيستخدم هذه البصمة بدل الكشف التلقائي.")


if __name__ == "__main__":
  main()
