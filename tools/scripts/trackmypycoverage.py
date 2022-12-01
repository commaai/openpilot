import os
import configparser

class results:
    num_passed = 0
    num_tested = 0

def parse_mypy_ini():
    config =  configparser.ConfigParser()
    mypy_ini = os.getcwd() + "\\mypy.ini"
    config.read(mypy_ini)
    dirToScan = config['mypy']['files'].split(", ")
    dirToExclude = config['mypy']['exclude'].split("/)|(")
    
    # Remove the extra characters at the begining and end
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
            print(f)
            b = os.system("mypy " + f + " --strict")
            if(b == 0):
                r.num_passed += 1
        elif(os.path.isdir(f) and file not in exclude):
            temp = test_mypy_coverage(f, exclude)
            r.num_passed += temp.num_passed
            r.num_tested += temp.num_tested
    return r

def output_file():
    # TODO Derek and Brandon
    print() # Placeholder

dirToScan, dirToExclude = parse_mypy_ini()

r = results()

for dir in dirToScan:
    temp = test_mypy_coverage("./" + dir, dirToExclude)
    r.num_passed += temp.num_passed
    r.num_tested += temp.num_tested

print(str(r.num_passed) + " / " + str(r.num_tested))

output_file()