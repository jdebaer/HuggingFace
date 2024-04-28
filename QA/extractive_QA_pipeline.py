# The below is with Haystack 1 - redo with Haystack 2

# Delete all indexes:
# curl -X DELETE 'http://localhost:9200/_all'



from haystack.document_stores import ElasticsearchDocumentStore

#document_store = ElasticsearchDocumentStore(host = "localhost", port = 9200, embedding_dim = 768)
document_store = ElasticsearchDocumentStore(return_embedding=True)

# Details are in QA_data_intro.py.
from datasets import get_dataset_config_names, load_dataset
domains = get_dataset_config_names('subjqa')
subjqa = load_dataset('subjqa', name='electronics')
dataframe_dict = {split: dset.to_pandas() for split, dset in subjqa.flatten().items()}

## Fill ES with some documents.
#for split, df in dataframe_dict.items():
#    print(split)
#    
#    docs = [{'content': row['context'],
#             'meta':{'item_id': row['title'],
#                     'question_id': row['id'],
#                     'split': split}}
#            for _, row in df.drop_duplicates(subset='context').iterrows()]
#
#    document_store.write_documents(docs, index='document')
#
##print(document_store.get_document_count())
# One entry looks like this.
# {
# "_index":"document",
# "_type":"_doc",
# "_id":"fa05750683f6d65d1c589bece85245d5",
# "_score":1.0,
# "_source":{	"content":"I bought <text> is rather trivial.",
#		"content_type":"text",
#		"id_hash_keys":["content"],
#		"item_id":"B00004SB92",
#		"question_id":"053e466d5ff93037de4f37e583986e5c",
#		"split":"train"
#	  }
# }

#from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.nodes.retriever.sparse import BM25Retriever

#es_retriever = ElasticsearchRetriever(document_store=document_store)
es_retriever = BM25Retriever(document_store=document_store)

item_id='B0074BW614'
query = 'Is it good for reading?'

retrieved_docs = es_retriever.retrieve(query=query, top_k=3, filters={'item_id':[item_id], 'split':['train']})
# This is a list with elements of type haystack.schema.Document.

#print(retrieved_docs[0].content)
retrieved_docs_strings = [doc.content for doc in retrieved_docs] 
#print(type(retrieved_docs_strings))
#print(len(retrieved_docs_strings))
#print(retrieved_docs_strings[0])
#exit(0)


# We load a pre-trained model that we're going to fine-tune later on. This is similar to how we load pre-trained models with HF.

from haystack.nodes.reader.farm import FARMReader

model_ckpt = 'deepset/minilm-uncased-squad2'
max_seq_len, doc_stride = 384, 128
reader = FARMReader(	model_name_or_path 	= model_ckpt,
			progress_bar		= False,
			max_seq_len 		= max_seq_len,
			doc_stride		= doc_stride,
                    	use_gpu			= False, 
			return_no_answer	= True)

# Do inference test.
#print(reader.predict_on_texts(question=query, texts=retrieved_docs_strings, top_k=1))
#exit()


# Plug it into the Haystack framework via a pipeline.
from haystack.pipelines.standard_pipelines import ExtractiveQAPipeline

pipe = ExtractiveQAPipeline(reader, es_retriever)

n_answers = 3
preds = pipe.run(
		query 		= query,
		params		= {'Retriever': {'top_k': 3, 'filters':{'item_id': [item_id], 'split':['train']}}, 'Reader': {'top_k': 3}})

print(type(preds))
print(preds)

#print(type(preds['answers'][idx]))
# haystack.schema.Answer is the type.

for idx in range(n_answers):
    print(f"{preds['answers'][idx].answer}")


