def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--correct_path', type=str)
    parser.add_argument('--incorrect_path', type=str)
    parser.add_argument('--ambiguous_path', type=str)
    args = parser.parse_args()

    with open(args.correct_path, 'r') as f:
        correct_lines = [q.strip()[1:] for q in f.readlines()]
    with open(args.incorrect_path, 'r') as f:
        incorrect_lines = [q.strip()[1:] for q in f.readlines()]

    for correct, incorrect in enumerate(zip(correct_lines, incorrect_lines)):
        ambiguous_lines.append("Knowledge1: " + correct + " [sep] Knowledge2: " + incorrect)
    with open(args.ambiguous_path, 'w') as f:
        f.write('\n'.join(ambiguous_lines))

if __name__ == '__main__':
    main()