import os

from common.basedir import BASEDIR

with open(os.path.join(BASEDIR, 'docs/CARS.md'), 'r') as f:
  orig = f.read()

with open(os.path.join(BASEDIR, 'docs/CARS_generated.md'), 'r') as f:
  generated = f.read()


def get_cars(md_raw, is_orig=False):
  cars = {}
  for line in md_raw.splitlines():
    if line.strip().startswith('|') and 'Supported Package' not in line and '---' not in line:
      line = [i.strip() for i in line.strip().split('|') if len(i.strip())]
      car = line[0] + ' ' + line[1]
      if '<sup>' in car:
        idx = car.index('<sup>')
        car = car[0:idx]

      if is_orig:  # use new conventions so we don't get false positives
        car = car.replace('-2018', '-18').replace('-2019', '-19').replace('-2020', '-20').replace('-2021', '-21').replace('-2022', '-22').replace('-2023', '-23')
        car = car.replace('PHEV', 'Plug-In Hybrid').replace('Plug-in', 'Plug-In').replace('EV', 'Electric')
        car = car.replace('Rav4', 'RAV4')
        stars = line[-3:]
      else:
        stars = line[-5:]
        # print(stars)
      # print('---')
      cars[car + ' ' + line[2]] = stars

  return cars


orig_cars = get_cars(orig, is_orig=True)
gen_cars = get_cars(generated)

# Test for missing cars either way
# Civic "missing" here is fine
for car in orig_cars:
  if car not in gen_cars:
    print('{} not in generated cars doc!'.format(car))

print('-----')

for car in gen_cars:
  if car not in orig_cars:
    print('{} not in original cars doc!'.format(car))

print('-----')

# Test for incorrect car stars
for car in gen_cars:
  if car in orig_cars:
    gen_op_long = 'full' in gen_cars[car][0] or 'half' in gen_cars[car][0]
    orig_op_long = 'openpilot' in orig_cars[car][0] or "<sup>3<" in orig_cars[car][0]
    if gen_op_long != orig_op_long:
      print('{} openpilot long doesn\'t match! {} vs {}'.format(car, orig_op_long, gen_op_long))

    orig_full_range_acc = orig_cars[car][1].lower().replace(' ', '').startswith('0mph')
    gen_full_range_acc = 'full' in gen_cars[car][1].lower()
    if orig_full_range_acc != gen_full_range_acc:
      print('{} openpilot FSR long doesn\'t match! {} vs {}'.format(car, orig_full_range_acc, gen_full_range_acc))

    orig_full_range_steer = orig_cars[car][2].lower().replace(' ', '').startswith('0mph')
    gen_full_range_steer = 'full' in gen_cars[car][2].lower()
    if orig_full_range_steer != gen_full_range_steer:
      print('{} openpilot FSR steer doesn\'t match! {} vs {}'.format(car, orig_full_range_steer, gen_full_range_steer))
