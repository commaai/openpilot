import subprocess
import re
import datetime
import time
import glob


def get_coverage_from_report(file_dir):
  """Extracts coverage percentage from mypy report"""
  with open(f'{file_dir}/any-exprs.txt', 'r') as f:
    for line in f:
      match = re.search(r'(\d+\.\d+)%', line)
      if match:
        coverage = float(match.group(1))
  return coverage


def main():
  # Generate dated filename for report
  date_str = datetime.datetime.now().strftime("%Y%m%d")
  report_filename = f'mypy_report_{date_str}'

  # Invoke mypy and generate report
  subprocess.run(["mypy", "common", f'--any-exprs-report=./mypy_reports/{report_filename}'])
  time.sleep(15)  # Allow time for mypy to parse directories

  # Get coverage from new report
  new_coverage = get_coverage_from_report(f'mypy_reports/{report_filename}')
  if new_coverage is None:
    print("Failed to extract coverage from new report!")
    return

  # Get latest report before the current one
  prior_reports = sorted(glob.glob('mypy_reports/mypy_report_*'))
  prior_reports.pop(-1)  # Removes latest report from list
  if prior_reports:
    last_report = prior_reports[-2]
    last_coverage = get_coverage_from_report(last_report)
    if last_coverage is None:
      print("Failed to extract coverage from last report!")
      return

    # Compare coverages and determine result
    if new_coverage > last_coverage:
      print("mypy coverage increased")
    elif new_coverage < last_coverage:
      print("mypy overage decreased")
    else:
      print("No change in mypy coverage")
  else:
    print(f"Coverage from new report: {new_coverage}% - No prior report to compare with.")


if __name__ == "__main__":
  main()
