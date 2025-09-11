import os, time, json, multiprocessing
import regex as re

from typing import BinaryIO
from collections import Counter, defaultdict
from tqdm import tqdm


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BYTE_SIZE = 256

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def iter_split(pattern: re.Pattern, text: str):
    last = 0
    for m in pattern.finditer(text):
        if last != m.start():
            yield text[last:m.start()]  # chunk before match
        last = m.end()
    if last < len(text):
        yield text[last:]

def worker_pretokenize(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str]
) -> Counter:
    """ 
    Woroker function:
    First split the chunks by special tokens then split according to PAT
    """
    token_map = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

        special_token_pattern = re.compile(f"{'|'.join(re.escape(token) for token in special_tokens)}")

        for split_chunk in iter_split(special_token_pattern, chunk):
            matches = re.finditer(PAT, split_chunk)

            for match in matches:
                curr_token = match.group().encode("utf-8") # need to track in bytes

                if curr_token not in token_map:
                    token_map[curr_token] = 1
                else:
                    token_map[curr_token] += 1

    return token_map

def pretokenize(
    input_path: str | os.PathLike,
    num_processes: int,
    special_tokens: list[str]
) -> Counter:
    """
    Spawn num_processes to pretokenize the input file using worker_pretokenize
    """
    boundaries = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    final_token_counts = Counter()

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with multiprocessing.Pool(processes=num_processes) as pool:
        list_of_counters = pool.starmap(worker_pretokenize, tasks)

        for counter in list_of_counters:
            final_token_counts.update(counter)
    
    return final_token_counts

def merge_word_tokens(
    word_tuple: tuple[int, ...],
    pair_to_merge: tuple[int, int],
    new_token_id: int
) -> tuple[int, ...]:
    """Merges all occurrences of a specific pair within a single tokenized word."""
    new_word = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i+1]) == pair_to_merge:
            new_word.append(new_token_id)
            i += 2
        else:
            new_word.append(word_tuple[i])
            i += 1
    return tuple(new_word)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    assert vocab_size >= len(special_tokens) + BYTE_SIZE, "Vocab size is too small"

    # Initial vocab and merges
    # Merging process
    merges = []

    # Initial vocab size (256 bytes value + special tokens)
    vocab = {
        val: bytes([val]) for val in range(BYTE_SIZE)
    }

    for idx, special_token in enumerate(special_tokens):
        vocab[len(vocab)] = special_token.encode("utf-8")
    
    # Pretokenization
    pretokenize_start = time.time()
    token_counts = pretokenize(input_path, num_processes, special_tokens)
    pretokenize_end = time.time()
    print(f"Pretokenization took {pretokenize_end - pretokenize_start:.2f} seconds")

    # tuple convert bytes into integers
    word_freqs = {
        tuple(word): count for word, count in token_counts.items()
    }

    # pair frequencies
    pair_freqs= Counter()
    pair_to_words = defaultdict(set)

    for word, freq in word_freqs.items():
        if len(word) < 2:
            continue

        for first, second in zip(word[:-1], word[1:]):
            pair = (first, second)
            pair_freqs[(first, second)] += freq
            pair_to_words[pair].add(word)
    
    num_merges = vocab_size - len(vocab)

    # Merging loop
    merging_start = time.time()
    for i in tqdm(range(num_merges), desc="Merging tokens"):
        # Break when there's no more pair to merge
        if not pair_freqs:
            print("No more pairs to merge. Stopping early.")
            break

        # Find the most frequent pair, break ties by lexicographical order
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], (vocab[p[0]], vocab[p[1]])))

        # Create new token for the merged pair
        b0, b1 = vocab[best_pair[0]], vocab[best_pair[1]]
        merges.append((b0, b1))
        
        new_token_id = len(vocab)
        vocab[new_token_id] = b0 + b1

        # Update the frequency that has best_pair in it
        words_to_update = list(pair_to_words[best_pair])

        for word in words_to_update:
            freq = word_freqs.pop(word)

            # Decrement stats for all pairs in old_word
            for old_p1, old_p2 in zip(word[:-1], word[1:]):
                old_pair = (old_p1, old_p2)
                pair_freqs[old_pair] -= freq
                pair_to_words[old_pair].discard(word)

            # Create new word with the merged pair
            new_word = merge_word_tokens(word, best_pair, new_token_id)
            word_freqs[new_word] = freq

            if len(new_word) > 1:
                for new_p1, new_p2 in zip(new_word[:-1], new_word[1:]):
                    new_pair = (new_p1, new_p2)
                    pair_freqs[new_pair] += freq
                    pair_to_words[new_pair].add(new_word)

        # Update pair_frequency incrementally
        del pair_freqs[best_pair]
        del pair_to_words[best_pair]
    
    merging_end = time.time()
    print(f"Merging took {merging_end - merging_start:.2f} seconds")
    
    return vocab, merges

def save_tokenizer_data(vocab, merges, output_path):
    """Saves vocab and merges to a single JSON file."""
    
    # 1. Prepare vocab for JSON: encode bytes to Base64 strings
    # JSON keys must be strings, so we convert the integer token IDs
    json_vocab = {
        str(token_id): token_bytes.decode('latin1')  # Use 'latin1' to preserve byte values
        for token_id, token_bytes in vocab.items()
    }
    
    # 2. Prepare merges for JSON: encode bytes to Base64 strings
    json_merges = [
        (b0.decode('latin1'), b1.decode('latin1'))  # Use 'latin1' to preserve byte values
        for b0, b1 in merges
    ]
    
    # 3. Combine into a single dictionary
    tokenizer_data = {
        "vocab": json_vocab,
        "merges": json_merges
    }
    
    # 4. Save to a single file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, indent=2)

def load_tokenizer_data(input_path):
    """Loads vocab and merges from a JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
    
    # 1. Load and decode vocab
    vocab = {
        int(token_id): token_b64.encode('latin1')
        for token_id, token_b64 in tokenizer_data["vocab"].items()
    }
    
    # 2. Load and decode merges
    merges = [
        (b0_b64.encode('latin1'), b1_b64.encode('latin1'))
        for b0_b64, b1_b64 in tokenizer_data["merges"]
    ]
    
    return vocab, merges

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input text file for training BPE tokenizer",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Size of the vocabulary (including special tokens)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of processes to use for pretokenization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the vocab and merges files",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<|endoftext|>"],
        help="List of special tokens to be added to the vocabulary",
    )
    args = parser.parse_args()

    vocab, merges = train_bpe(
        args.input_path,
        args.vocab_size,
        args.special_tokens,
        args.num_processes
    )

    print("Final vocab size:", len(vocab))
    print("Final merges size:", len(merges))

    # Save vocab and merges
    save_tokenizer_data(vocab, merges, os.path.join(args.output_dir, "tokenizer.json"))
