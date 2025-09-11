import os
import regex as re
import multiprocessing

from typing import BinaryIO
from collections import Counter

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

        special_token_pattern = f"{'|'.join(re.escape(token) for token in special_tokens)}"
        split_chunks = re.split(special_token_pattern, chunk)

        for split_chunk in split_chunks:
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
    num_processes = 4
    token_counts = pretokenize(input_path, num_processes, special_tokens)

    # tuple convert bytes into integers
    word_freqs = {
        tuple(word): count for word, count in token_counts.items()
    }

    # pair frequencies
    pair_freqs= Counter()

    for word, freq in word_freqs.items():
        for first, second in zip(word[:-1], word[1:]):
            pair_freqs[(first, second)] += freq

    # Merging loop
    while len(vocab) < vocab_size:        
        # Break when there's no more pair to merge
        if not pair_freqs:
            print("No more pairs to merge. Stopping early.")
            break

        # Find the most frequent pair, break ties by lexicographical order
        best_pair = max(pair_freqs, key=pair_freqs.get)

        # Create new token for the merged pair
        b0, b1 = vocab[best_pair[0]], vocab[best_pair[1]]
        merges.append((b0, b1))
        
        new_token_id = len(vocab)
        vocab[new_token_id] = b0 + b1

        # Update the frequency that has best_pair in it
        new_word_freqs = Counter()

        for word, freq in word_freqs.items():
            if best_pair[0] not in word and best_pair[1] not in word:
                new_word_freqs[word] += freq
                continue
            
            # Potentially need to merge the pair in word
            new_word = merge_word_tokens(word, best_pair, new_token_id)
            new_word_freqs[new_word] += freq

            for p1, p2 in zip(word[:-1], word[1:]):
                if (p1, p2) != best_pair:
                    pair_freqs[(p1, p2)] -= freq
            
            for p1, p2 in zip(new_word[:-1], new_word[1:]):
                pair_freqs[(p1, p2)] += freq

        # Update pair_frequency incrementally
        word_freqs = new_word_freqs
        del pair_freqs[best_pair]
    
    return vocab, merges
