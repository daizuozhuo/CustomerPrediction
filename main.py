#!/opt/local/bin/python

def main(filename):
    file = open(filename)
    print file.readline()
    pass

if __name__ == "__main__":
    main("data/orange_small_test.data")
