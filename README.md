# genkit-astra-rag
Testing how to use astraDB vector DB with Google's genkit

Make sure to create the collection with 768 dimensions as textEmbedding004 supports 768 dimensions only. I used BAAI/bge-base-en-v1.5 to generate embeddings with Astra, as their free tire model only supports 1024 dimensions.

npm install
genkit start -- npm start
npm run genkit