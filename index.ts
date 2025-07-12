/**
 * Example RAG (Retrieval-Augmented Generation) setup using Genkit and AstraDB.
 * 
 * Genkit is an open-source framework for building AI workflows, including retrieval-augmented generation (RAG).
 * AstraDB is a serverless, cloud-native database built on Apache Cassandra, suitable for storing and retrieving embeddings for RAG.
 * 
 * This file demonstrates:
 *  - Setting up Genkit with Google AI and AstraDB plugins
 *  - Indexing web pages into AstraDB
 *  - Retrieving relevant documents and generating answers using Gemini models
 */

import {z, Document, genkit} from "genkit";
import { textEmbedding004, googleAI, gemini20Flash } from "@genkit-ai/googleai";
import { astraDBIndexerRef, astraDBRetrieverRef, astraDB } from "genkitx-astra-db";
import { Readability } from "@mozilla/readability";
import { JSDOM } from "jsdom";
import { chunk } from "llm-chunk";

// Name of the AstraDB collection to store indexed documents.
// Can be set via environment variable COLLECTION_NAME, defaults to "rag".
const collectionName = process.env.COLLECTION_NAME || "rag";

/**
 * Initialize Genkit with plugins:
 *  - googleAI: for embedding and generation models (Gemini, etc.)
 *  - astraDB: for storing and retrieving document embeddings
 * 
 * astraDB plugin parameters:
 *  - applicationToken: AstraDB API token
 *  - apiEndpoint: AstraDB API endpoint
 *  - collectionName: Name of the collection to store documents
 *  - embedder: Embedding model to use for indexing
 */
const ai = genkit({
    plugins: [
        googleAI(),
        astraDB([
            {
                clientParams: {
                    applicationToken: process.env.ASTRA_DB_APPLICATION_TOKEN || "",
                    apiEndpoint: process.env.ASTRA_DB_API_ENDPOINT || "",
                },
                collectionName: process.env.COLLECTION_NAME || "rag",
                embedder: textEmbedding004,
            },
        ]),
    ]
});

// Create indexer and retriever references for the specified collection.
// These are used to index documents and retrieve relevant documents for queries.
export const astraDbIndexer = astraDBIndexerRef({ collectionName });
export const astraDbRetriever = astraDBRetrieverRef({ collectionName });

/**
 * Fetches and extracts readable text content from a web page.
 * Uses JSDOM to parse HTML and Mozilla Readability to extract main article text.
 * @param url - The URL of the web page to extract text from.
 * @returns Extracted text content as a string.
 */
async function getFromWeb(url: string) {
    const html = await fetch(url).then((res) => res.text());
    const dom = new JSDOM(html, {url});
    const reader = new Readability(dom.window.document);
    const article = reader.parse();
    return article?.textContent || "";
}

/**
 * Flow to index a web page into AstraDB.
 * Steps:
 *  1. Extract text from the given URL.
 *  2. Chunk the text into manageable pieces for embedding.
 *  3. Convert chunks into Genkit Document objects.
 *  4. Index documents into AstraDB using the configured indexer.
 * 
 * genkit.defineFlow parameters:
 *  - name: Name of the flow ("indexWebPage")
 *  - inputSchema: Input validation (expects a URL string)
 *  - outputSchema: Output type (void)
 */
export const indexWebPage = ai.defineFlow(
    {
        name: "indexWebPage",
        inputSchema: z.string().url().describe("URL"),
        outputSchema: z.void()
    },
    // Flow implementation: indexes the content of a web page
    async(url: string) => {
        // Extract readable text from the web page
        const text = await ai.run("extract-text", () => getFromWeb(url));

        // Chunk the text for embedding (minLength, maxLength, overlap)
        const chunks = await ai.run("chunk-text", async () => chunk(text, {minLength: 128, maxLength: 1024, overlap: 128}));

        // Convert chunks into Genkit Document objects, tagging with source URL
        const documents = chunks.map((text) => {
            return Document.fromText(text, {url});
        });

        // Index documents into AstraDB
        // This step creates embeddings for each document and stores them in AstraDB
        return ai.index({
            indexer: astraDbIndexer,
            documents,
        });
    }
);

/**
 * RAG flow: Answers a question using retrieved context from AstraDB and Gemini model.
 * Steps:
 *  1. Retrieve top-k relevant documents from AstraDB using the retriever.
 *  2. Generate an answer using Gemini model, constrained to use only retrieved context.
 * 
 * genkit.defineFlow parameters:
 *  - name: Name of the flow ("ragFlow")
 *  - inputSchema: Input validation (expects a string question)
 *  - outputSchema: Output type (string answer)
 */
export const ragFlow = ai.defineFlow(
    {
        name: "ragFlow",
        inputSchema: z.string(),
        outputSchema: z.string()
    },
    // Flow implementation: answers a question using retrieved context
    async (input: string) => {
        // Retrieve top 3 relevant documents from AstraDB
        const docs = await ai.retrieve({
            retriever: astraDbRetriever,
            query: input,
            options: {
                k:3 // Number of documents to retrieve
            }
        });

        // Generate answer using Gemini model, using only retrieved context
        const response = await ai.generate({
            model: gemini20Flash, // The LLM model to use for generation
            prompt: "You are a helpful AI assistant that can answer questions." +
                    "Use only the context provided to answer the question." + 
                    "If you don't know, do not make up an answer." + 
                    `Question: ${input}`, // The prompt for the model, including instructions and the user's question
            docs, // Array of retrieved documents to use as context for the answer
        });
        return response.text;
    }
);