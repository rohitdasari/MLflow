import argparse 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name","-n" ,default="rohit" ,type = str)
    parser.add_argument("--age", "-a",default=25.0, type = float)
    parse_args = parser.parse_args()
    print(f"Name: {parse_args.name}")
    print(f"Age: {parse_args.age}")