import sys

if __name__ == "__main__":
    file = sys.argv[1]
    if "/" in file:
        name = file.split("/")[-1]
    else:
        name = file
    print(name[:-4])