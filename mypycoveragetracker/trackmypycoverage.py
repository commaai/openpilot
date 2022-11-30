import os;

def test_mypy_coverage(directory):
    numFilesPassed = 0;
    totalFilesTested = 0;
    files = os.listdir(directory)
    for file in files:
        f = directory + "/" + file
        if file.endswith(".py"):
            totalFilesTested += 1;
            print(f)
            b = os.system("mypy " + f + " --strict")
            if(b == 0):
                numFilesPassed += 1
        elif(os.path.isdir(file)):
            numFilesPassed += test_mypy_coverage(f)
    return numFilesPassed, totalFilesTested

num_passed, total = test_mypy_coverage("./common")

print(str(num_passed) + " / " + str(total))