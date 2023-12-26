import sys

if __name__ == "__main__":
    output_file = sys.argv[1]
    with open(output_file) as f:
        for line in f:
            print(line)
