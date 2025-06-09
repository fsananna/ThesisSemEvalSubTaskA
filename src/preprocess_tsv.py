import pandas as pd

# Path to the TSV file
tsv_file_path = r"F:\Sem7\Thesis\CODE\dataset\test\subtask_a_test.tsv"

# Read the TSV file
df = pd.read_csv(tsv_file_path, sep='\t')

# Function to clean and fix the expected_order column
def clean_and_fix_expected_order(order, compound):
    if isinstance(order, str):
        cleaned = order.strip()
        # Fix typo for "act of god"
        if compound == 'act of god':
            cleaned = cleaned.replace('73444821016.png', '7344821016.png')
        return cleaned
    return order

# Apply the cleaning and fixing function to the expected_order column
df['expected_order'] = df.apply(lambda row: clean_and_fix_expected_order(row['expected_order'], row['compound']), axis=1)

# Save the modified DataFrame back to the TSV file
df.to_csv(tsv_file_path, sep='\t', index=False)

print("TSV file has been preprocessed and saved successfully.")