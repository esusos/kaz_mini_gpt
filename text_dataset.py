import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        # Load the cleaned text from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # Get a list of all the unique characters in the text
        self.chars = sorted(list(set(self.text)))
        print(self.chars)

        # Create a dictionary to map characters to indices
        self.char_to_index = { char: index for index, char in enumerate(self.chars) }

        # Save the sequence length
        self.seq_length = seq_length

        # Calculate the number of sequences that can be generated from the text
        self.num_sequences = len(self.text) - seq_length


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        # Get the input and target sequences for the given index
        input_seq = self.text[index : index + self.seq_length]
        target_seq = self.text[index + self.seq_length]

        # Convert the input and target sequences to tensors of indices
        input_indices = torch.tensor([self.char_to_index[char] for char in input_seq], dtype=torch.long)
        target_index = torch.tensor(self.char_to_index[target_seq], dtype=torch.long)

        # Return the input and target sequences as a tuple of tensors
        return input_indices, target_index
