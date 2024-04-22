from tqdm import tqdm

# Split a long list in batch_sized lists.
def split_in_chunks(list_of_elements, batch_size):
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]

def evaluate_peg_summaries(dataset, metric, model, tokenizer, batch_size=16, device='cpu', column_text='article', column_summary='highlights'):

    article_list_of_lists = list(split_in_chunks(dataset[column_text], batch_size))
    summary_list_of_lists = list(split_in_chunks(dataset[column_summary], batch_size))

    # Run the eval chunk by chunk.
    for article_list, summary_list in tqdm(zip(article_list_of_lists, summary_list_of_lists), total=len(summary_list_of_lists)):

        tokens = tokenizer(article_list, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')

        summaries = model.generate(
            input_ids           = tokens['input_ids'].to(device),
            attention_mask      = tokens['attention_mask'].to(device),
            length_penalty      = 0.8,                                          # Prevent model from generating too much text.
            num_beams           = 8,
            max_length          = 128)

        decoded_summaries = [tokenizer.decode(summary, skip_special_tokens=True, clean_up_tokenization_spaces=True) for summary in summaries]

        decoded_summaries = [decoding.replace('<n>', ' ') for decoding in decoded_summaries]            # References don't have the '<n>'.

        metric.add_batch(predictions=decoded_summaries, references = summary_list)

    score = metric.compute()
    return score
