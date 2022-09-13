import argparse
import glob


def indexing(input_file: str, output_file: str):
    indexes = []

    with open(input_file, "rb") as f:
        line = f.readline()
        while line:
            indexes.append(f.tell() - len(line))
            line = f.readline()

    with open(output_file, "w", encoding="utf8") as f:
        f.write("\n".join(map(str, indexes)))

    print(f"{input_file} done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", type=str, required=True)
    parser.add_argument("--index_suffix", "-s", type=str, default=".index")
    args = parser.parse_args()

    for input_file in glob.glob(args.inputs):
        if not input_file.endswith(args.index_suffix):
            output_file = input_file + args.index_suffix
            indexing(input_file, output_file)


if __name__ == "__main__":
    main()
