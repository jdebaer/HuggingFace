import pandas as pd

try:
    df_issues = pd.read_pickle('df_issues.pkl')
except:
    dataset_url = 'https://git.io/nlp-with-transformers'
    df_issues = pd.read_json(dataset_url, lines=True)
    df_issues.to_pickle('df_issues.pkl')

cols = ['url', 'id', 'title', 'user', 'labels', 'state', 'created_at', 'body']
#print(df_issues.loc[2, cols].to_frame())

# Filter out just the value of the 'name' key in each tag's JSON object (in the list of tag JSON objects).

df_issues['labels'] = df_issues['labels'].apply(lambda tag_obj_list: [tag_obj['name'] for tag_obj in tag_obj_list])

# Count how many rows (lists) have 0, 1, 2, ... tags.

#print(df_issues['labels'].apply(lambda tag_list: len(tag_list)).value_counts().to_frame().T)

# Find top 10 most frequently used labels.
df_labels_exploded = df_issues['labels'].explode()						# Each element in the list becomes a row.
df_counts = df_labels_exploded.value_counts()							# Now count the rows per label.



