import random

# Flower types
flower_types = ['daisy', 'dandelion', 'grtz', 'irises', 'lilies', 'lisianthuses', 'orchids', 'peonies', 'roses', 'tulips']

def generate_caption():
    # Decide the number of flower types in the bouquet (1 or 2)
    num_types = random.choice([1, 2])
   
    selected_flowers = random.sample(flower_types, num_types)
    
    # Randomly allocate a number of flowers to each type, ensuring the total does not exceed 20
    if num_types == 1:
        flower_counts = [random.randint(1, 20)]
    else:
        first_count = random.randint(1, 19)
        second_count = random.randint(1, 20 - first_count)
        flower_counts = [first_count, second_count]
    
    # Create the caption
    parts = [f"{count} {flower}" for count, flower in zip(flower_counts, selected_flowers)]
    return "A flower bouquet consisting of " + ", ".join(parts) + "."

# Generate 100 captions
captions = [generate_caption() for _ in range(100)]


new_txt_path = 'test/test_prompts.txt'
with open(new_txt_path, 'w') as file:
    for line in captions:
        # Convert each number to string and join with spaces
        # file.write(' '.join(map(str, line)) + '\n')
        file.write(line+'\n')