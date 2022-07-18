# creates a Pyserini Lucene index from input documents in trecweb format    
python -m pyserini.index.lucene \
  --collection TrecwebCollection \
  --input $1 \
  --index $2 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw