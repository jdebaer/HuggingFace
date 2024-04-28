from datasets import get_dataset_config_names, load_dataset
#import pandas as pd
#pd.set_option('display.max_columns', None)

domains = get_dataset_config_names('subjqa')
#print(domains)

subjqa = load_dataset('subjqa', name='electronics')

# flatten() on DatasetDict flattens out each of the three Dicts.
# items() returns string, Dataset with the string being the name of the split (train/...).

dataframe_dict = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

#for split, df in dataframe_dict.items():
#    print(df.head())

# For trainining we need the cols title, question, answers.text, answers.answer_start, context.

