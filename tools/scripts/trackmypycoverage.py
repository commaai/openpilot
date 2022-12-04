import os
import configparser

class results:
  num_passed = 0
  num_tested = 0

# Generate report file
report_file = open("./tools/scripts/report.txt","w")
report_file.write("Mypy Coverage Report\n====================\n\n")
report_file.write("======================================\n")
report_file.write("| Module   | Imprecision    | Lines  |\n")
report_file.write("======================================\n")
report_file.close()

def parse_mypy_ini():
  config =  configparser.ConfigParser()
  mypy_ini = os.getcwd() + "\\mypy.ini"
  config.read(mypy_ini)
  dirToScan = config['mypy']['files'].split(", ")
  dirToExclude = config['mypy']['exclude'].split("/)|(")
    
  # Remove the extra characters at the beginning and end
  dirToExclude[0] = dirToExclude[0][2:]
  dirToExclude[-1] = dirToExclude[-1][:-2]
    
  return dirToScan, dirToExclude

files_total = []

def test_mypy_coverage(directory, exclude):
  r = results()
  files = os.listdir(directory)
  for file in files:
    f = directory + "/" + file
    if file.endswith(".py"):
      files_total.append(f)
      r.num_tested += 1
      b = os.system("mypy " + f + " --strict --txt-report ./tools/scripts")
      # open both files
      with open('./tools/scripts/index.txt','r') as indexfile, open('./tools/scripts/report.txt','a') as reportfile:
        data = indexfile.readlines()
        # read line 9 of from index.txt file and append line to report.txt file
        reportfile.write(data[8])
        if(b == 0):
          r.num_passed += 1
        elif(os.path.isdir(f) and file not in exclude):
          temp = test_mypy_coverage(f, exclude)
          r.num_passed += temp.num_passed
          r.num_tested += temp.num_tested
  return r

def output_file():
  # Add to end of report
  f = open('./tools/scripts/report.txt','a')
  if r.num_tested == 0:
    f.write("\nNo files tested")
  else:
    f.write("\nTested: " + str(r.num_tested) + " files\n")
    f.write("Passed: " + str(r.num_passed))
    f.write("\nPercentage Covered: " + str(round(float(r.num_passed/r.num_tested * 100),2)) + "%")

    # Delete index.txt
  if os.path.exists("./tools/scripts/index.txt"):
    os.remove("./tools/scripts/index.txt")

dirToScan, dirToExclude = parse_mypy_ini()

r = results()

for directory in dirToScan:
  temp = test_mypy_coverage("./" + directory, dirToExclude)
  r.num_passed += temp.num_passed
  r.num_tested += temp.num_tested

print(str(r.num_passed) + " / " + str(r.num_tested))

output_file()