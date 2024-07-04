import random

def shuffle_line(line):
    # Split the line into a list of strings
    words = line.split()
    # Shuffle the list of strings
    random.shuffle(words)
    # Join the shuffled strings back into a line
    shuffled_line = ' '.join(words)
    return shuffled_line

def shuffle_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            shuffled_line = shuffle_line(line)
            # Write the shuffled
            outfile.write(shuffled_line + '\n')

random.seed(42)

splits= ["val.tgt"]#["train.tgt","test.tgt", "valid.tgt"]

original_data_folders = ["wos/"] #["nyt/","rcv1/","wos/"]

for data_folder in original_data_folders:
    for split in splits:
        in_file =data_folder+split
        out_file = data_folder+"shuffled/"+split
        shuffle_file(in_file,out_file)

    
