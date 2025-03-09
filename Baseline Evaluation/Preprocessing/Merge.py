import pandas as pd

# Load the true and false datasets
true_links_df = pd.read_csv('path of true file obtained from Removing_Nan.py')
false_links_df = pd.read_csv('path of false file obtained from False_Link.py')


merged_df = pd.concat([true_links_df, false_links_df], ignore_index=True)

merged_df.to_csv('path to save the file', index=False)


print("Merged Dataset:")
print(merged_df.head())
