import random
from typing import List, Tuple

def generate_nominal_language_samples(
    num_samples: int, 
    max_length: int, 
    value_range: Tuple[float, float] = (0.0, 10.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for the nominal language a*.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the strings.
        value_range (Tuple[float, float]): Range for random float values.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: A tuple containing positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Generate a random constant `a` for positive samples
        a = random.uniform(*value_range)
        length = random.randint(1, max_length)

        # Positive sample: Repeat the same value `a`
        positive_samples.append([a] * length)

        # Negative sample: Ensure it contains at least two distinct values
        while True:
            neg_length = random.randint(1, max_length)
            neg_sample = [random.uniform(*value_range) for _ in range(neg_length)]
            if len(set(neg_sample)) > 1:  # Ensure at least two unique values
                negative_samples.append(neg_sample)
                break

    return positive_samples, negative_samples




def generate_ab_language_samples(
    num_samples: int, 
    max_length: int, 
    value_range: Tuple[float, float] = (0.0, 10.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for the nominal language (ab)*.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum number of (ab) pairs in the sequence.
        value_range (Tuple[float, float]): Range for random float values.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: A tuple containing positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Generate two distinct random values for 'a' and 'b'
        a = random.uniform(*value_range)
        b = random.uniform(*value_range)
        while b == a:  # Ensure a and b are distinct
            b = random.uniform(*value_range)

        # Positive sample: Alternating sequence of a and b
        length = random.randint(1, max_length)  # Number of (ab) pairs
        positive_samples.append([a, b] * length)

        # Negative sample: Deviates from (ab)*
        while True:
            neg_length = random.randint(1, max_length * 2)  # Random length, potentially odd
            neg_sample = [random.uniform(*value_range) for _ in range(neg_length)]
            
            # Check if it deviates from (ab)* pattern
            is_valid_negative = False
            for i in range(len(neg_sample) - 1):
                if neg_sample[i] == neg_sample[i + 1]:
                    is_valid_negative = True  # Adjacent values should not be identical
                    break
            if is_valid_negative or len(neg_sample) % 2 != 0:
                negative_samples.append(neg_sample)
                break

    return positive_samples, negative_samples



def generate_tomita3_samples_with_values(
    num_samples: int,
    max_length: int,
    value_range: Tuple[float, float] = (0.0, 10.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for the Tomita 3 language:
    "No odd number of 'a's after an odd number of 'b's."
    Values of 'a' and 'b' are floating-point numbers from a given range.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the strings.
        value_range (Tuple[float, float]): Range for random float values.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: Positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Generate distinct random values for 'a' and 'b'
        a = random.uniform(*value_range)
        b = random.uniform(*value_range)
        while b == a:  # Ensure distinct values
            b = random.uniform(*value_range)

        # Positive sample: Satisfying Tomita 3 condition
        while True:
            length = random.randint(1, max_length)
            sample = [random.choice([a, b]) for _ in range(length)]
            
            # Check if the sample satisfies the Tomita 3 condition
            odd_b = False  # Track if we're in an odd 'b' state
            valid = True
            for value in sample:
                if value == b:
                    odd_b = not odd_b
                elif value == a and odd_b:  # Odd 'b' followed by 'a'
                    valid = not valid  # Break if odd 'a's appear
                    break
            
            if valid:
                positive_samples.append(sample)
                break

        # Negative sample: Violating Tomita 3 condition
        while True:
            length = random.randint(1, max_length)
            sample = [random.choice([a, b]) for _ in range(length)]
            
            # Check if the sample violates the Tomita 3 condition
            odd_b = False
            valid = True
            odd_a_count_after_odd_b = 0
            for value in sample:
                if value == b:
                    odd_b = not odd_b
                    odd_a_count_after_odd_b = 0  # Reset 'a' count after 'b'
                elif value == a and odd_b:
                    odd_a_count_after_odd_b += 1
                if odd_a_count_after_odd_b % 2 == 1:
                    valid = False  # Invalid if odd 'a' count follows odd 'b'
                    break

            if not valid:
                negative_samples.append(sample)
                break

    return positive_samples, negative_samples


def generate_tomita4_samples_with_values(
    num_samples: int,
    max_length: int,
    value_range: Tuple[float, float] = (0.0, 10.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for the Tomita 4 language:
    "No substring 'aaa' exists."
    Values of 'a' and 'b' are floating-point numbers from a given range.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the strings.
        value_range (Tuple[float, float]): Range for random float values.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: Positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Generate distinct random values for 'a' and 'b'
        a = random.uniform(*value_range)
        b = random.uniform(*value_range)
        while b == a:  # Ensure distinct values
            b = random.uniform(*value_range)

        # Positive sample: Satisfying Tomita 4 condition
        while True:
            length = random.randint(1, max_length)
            sample = [random.choice([a, b]) for _ in range(length)]

            # Check if the sample satisfies the Tomita 4 condition
            if all(sample[i:i+3] != [a, a, a] for i in range(len(sample) - 2)):
                positive_samples.append(sample)
                break

        # Negative sample: Violating Tomita 4 condition
        while True:
            length = random.randint(3, max_length)  # At least length 3 for 'aaa'
            sample = [random.choice([a, b]) for _ in range(length)]

            # Check if the sample contains the forbidden 'aaa' substring
            if any(sample[i:i+3] == [a, a, a] for i in range(len(sample) - 2)):
                negative_samples.append(sample)
                break

    return positive_samples, negative_samples


def generate_tomita5_samples_with_values(
    num_samples: int,
    max_length: int,
    value_range: Tuple[float, float] = (0.0, 10.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for the Tomita 5 language:
    "Strings with an even number of 'a's and an even number of 'b's."
    Values of 'a' and 'b' are floating-point numbers from a given range.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the strings.
        value_range (Tuple[float, float]): Range for random float values.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: Positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Generate distinct random values for 'a' and 'b'
        a = random.uniform(*value_range)
        b = random.uniform(*value_range)
        while b == a:  # Ensure distinct values
            b = random.uniform(*value_range)

        # Positive sample: Satisfying Tomita 5 condition
        while True:
            length = random.randint(2, max_length)  # Minimum length 2 for even counts
            sample = [random.choice([a, b]) for _ in range(length)]

            # Count occurrences of 'a' and 'b'
            count_a = sample.count(a)
            count_b = sample.count(b)

            # Check if counts are even
            if count_a % 2 == 0 and count_b % 2 == 0:
                positive_samples.append(sample)
                break

        # Negative sample: Violating Tomita 5 condition
        while True:
            length = random.randint(1, max_length)
            sample = [random.choice([a, b]) for _ in range(length)]

            # Count occurrences of 'a' and 'b'
            count_a = sample.count(a)
            count_b = sample.count(b)

            # Ensure at least one of the counts is odd
            if count_a % 2 != 0 or count_b % 2 != 0:
                negative_samples.append(sample)
                break

    return positive_samples, negative_samples



def generate_tomita6_samples_with_values(
    num_samples: int,
    max_length: int,
    value_range: Tuple[float, float] = (0.0, 10.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for the Tomita 6 language:
    "The difference (number of 'a's - number of 'b's) mod 3 == 0."
    Values of 'a' and 'b' are floating-point numbers from a given range.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the strings.
        value_range (Tuple[float, float]): Range for random float values.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: Positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Generate distinct random values for 'a' and 'b'
        a = random.uniform(*value_range)
        b = random.uniform(*value_range)
        while b == a:  # Ensure distinct values
            b = random.uniform(*value_range)

        # Positive sample: Satisfying Tomita 6 condition
        while True:
            length = random.randint(1, max_length)
            sample = [random.choice([a, b]) for _ in range(length)]

            # Count occurrences of 'a' and 'b'
            count_a = sample.count(a)
            count_b = sample.count(b)

            # Check if (count_a - count_b) % 3 == 0
            if (count_a - count_b) % 3 == 0:
                positive_samples.append(sample)
                break

        # Negative sample: Violating Tomita 6 condition
        while True:
            length = random.randint(1, max_length)
            sample = [random.choice([a, b]) for _ in range(length)]

            # Count occurrences of 'a' and 'b'
            count_a = sample.count(a)
            count_b = sample.count(b)

            # Ensure (count_a - count_b) % 3 != 0
            if (count_a - count_b) % 3 != 0:
                negative_samples.append(sample)
                break

    return positive_samples, negative_samples


def generate_tomita7_samples_with_values(
    num_samples: int,
    max_length: int,
    value_range: Tuple[float, float] = (0.0, 10.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for the Tomita 7 language:
    "Strings in the pattern a^*b^*a^*b^*."
    Values of 'a' and 'b' are floating-point numbers from a given range.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the strings.
        value_range (Tuple[float, float]): Range for random float values.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: Positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Generate distinct random values for 'a' and 'b'
        a = random.uniform(*value_range)
        b = random.uniform(*value_range)
        while b == a:  # Ensure distinct values
            b = random.uniform(*value_range)

        # Positive sample: Satisfying Tomita 7 pattern
        length_a1 = random.randint(0, max_length // 4)
        length_b1 = random.randint(0, max_length // 4)
        length_a2 = random.randint(0, max_length // 4)
        length_b2 = random.randint(0, max_length // 4)

        sample = [a] * length_a1 + [b] * length_b1 + [a] * length_a2 + [b] * length_b2
        positive_samples.append(sample)

        # Negative sample: Violating Tomita 7 pattern
        while True:
            length = random.randint(1, max_length)
            sample = [random.choice([a, b]) for _ in range(length)]

            # Check if the sample violates the Tomita 7 pattern
            seen_a1 = seen_b1 = seen_a2 = seen_b2 = False
            valid = True
            for value in sample:
                if not seen_b1 and value == b:
                    seen_b1 = True
                elif seen_b1 and not seen_a2 and value == a:
                    seen_a2 = True
                elif seen_a2 and not seen_b2 and value == b:
                    seen_b2 = True
                elif seen_b2 and value == a:
                    valid = False  # 'a' after second 'b' violates pattern
                    break

            if not valid or not (seen_b1 and seen_a2 and seen_b2):
                negative_samples.append(sample)
                break

    return positive_samples, negative_samples

import random
from typing import List, Tuple

def generate_increasing_sequence_samples(
    num_samples: int,
    max_length: int,
    value_range: Tuple[float, float] = (0.0, 10.0),
    step_range: Tuple[float, float] = (0.1, 1.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for a nominal language with increasing sequences.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the sequences.
        value_range (Tuple[float, float]): Range for the starting value of the sequence.
        step_range (Tuple[float, float]): Range for the incremental step in positive sequences.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: A tuple containing positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Positive sample: Generate an increasing sequence
        start_value = random.uniform(*value_range)
        step = random.uniform(*step_range)
        length = random.randint(1, max_length)
        positive_samples.append([start_value + i * step for i in range(length)])

        # Negative sample: Ensure it does not follow a strictly increasing pattern
        while True:
            neg_length = random.randint(1, max_length)
            neg_sample = [random.uniform(*value_range) for _ in range(neg_length)]
            if not all(neg_sample[i] < neg_sample[i + 1] for i in range(len(neg_sample) - 1)):
                negative_samples.append(neg_sample)
                break

    return positive_samples, negative_samples

def generate_decreasing_sequence_samples(
    num_samples: int,
    max_length: int,
    value_range: Tuple[float, float] = (0.0, 10.0),
    step_range: Tuple[float, float] = (0.1, 1.0)
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Generate positive and negative samples for a nominal language with decreasing sequences.

    Args:
        num_samples (int): The number of positive and negative samples to generate.
        max_length (int): The maximum length of the sequences.
        value_range (Tuple[float, float]): Range for the starting value of the sequence.
        step_range (Tuple[float, float]): Range for the decremental step in positive sequences.

    Returns:
        Tuple[List[List[float]], List[List[float]]]: A tuple containing positive and negative samples.
    """
    positive_samples = []
    negative_samples = []

    for _ in range(num_samples):
        # Positive sample: Generate a decreasing sequence
        start_value = random.uniform(*value_range)
        step = random.uniform(*step_range)
        length = random.randint(1, max_length)
        positive_samples.append([start_value - i * step for i in range(length)])

        # Negative sample: Ensure it does not follow a strictly decreasing pattern
        while True:
            neg_length = random.randint(1, max_length)
            neg_sample = [random.uniform(*value_range) for _ in range(neg_length)]
            if not all(neg_sample[i] > neg_sample[i + 1] for i in range(len(neg_sample) - 1)):
                negative_samples.append(neg_sample)
                break

    return positive_samples, negative_samples


if __name__ == "__main__":
    num_samples = 5
    max_length = 10
    value_range = (5.0, 10.0)
    step_range = (0.1, 1.0)

    positive_samples, negative_samples = generate_decreasing_sequence_samples(
        num_samples, max_length, value_range, step_range
    )

    print("Positive Samples:")
    for sample in positive_samples:
        print(sample)

    print("\nNegative Samples:")
    for sample in negative_samples:
        print(sample)
