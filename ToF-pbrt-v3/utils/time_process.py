import os
import sys

if __name__ == "__main__":
    input_file = sys.argv[1]
    save_line  = int(sys.argv[2])
    results = []
    with open(input_file, 'r') as file:
        lines = file.readlines()
        num_lines = len(lines)
        for i in range(num_lines):
            if i & 1 != save_line: continue
            results.append(lines[i])
    with open(input_file, 'w') as file:
        for line in results:
            file.write(f"{line}")
    print(f"File '{input_file}' processed. Saving {'odd' if save_line == 1 else 'even'} lines.")