import {z, Document, genkit, retrieverRef} from "genkit";
import { textEmbedding004, googleAI, gemini20Flash, d } from "@genkit-ai/googleai";
import { astraDBIndexerRef, astraDBRetrieverRef, astraDB } from "genkitx-astra-db";
import { Readability } from "@mozilla/readability";
import { JSDOM } from "jsdom";
import { chunk } from "llm-chunk";

const collectionName = process.env.COLLECTION_NAME || "rag";

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

export const astraDbIndexer = astraDBIndexerRef({ collectionName });
export const astraDbRetriever = astraDBRetrieverRef({ collectionName });

async function getFromWeb(url: string) {
    const html = await fetch(url).then((res) => res.text());
    const dom = new JSDOM(html, {url});
    const reader = new Readability(dom.window.document);
    const article = reader.parse();
    return article?.textContent || "";
}

export const indexWebPage = ai.defineFlow(
    {
        name: "indexWebPage",
        inputSchema: z.string().url().describe("URL"),
        outputSchema: z.void()
    },
    async(url: string) => {
        const text = await ai.run("extract-text", () => getFromWeb(url));

        const chunks = await ai.run("chunk-text", async () => chunk(text, {minLength: 128, maxLength: 1024, overlap: 128}));

        const documents = chunks.map((text) => {
            return Document.fromText(text, {url});
        });

        return ai.index({
            indexer: astraDbIndexer,
            documents,
        });
    }
);

export const ragFlow = ai.defineFlow(
    {
        name: "ragFlow",
        inputSchema: z.string(),
        outputSchema: z.string()
    },
    async (input: string) => {
        const docs = await ai.retrieve({
            retriever: astraDbRetriever,
            query: input,
            options: {
                k:3
            }
        });

        const response = await ai.generate({
            model: gemini20Flash,
            prompt:"You are a helpful AI assistant that can answer questions." +

                    "Use only the context provided to answer the question." + 
                    "If you don't know, do not make up an answer." + 

                    `Question: ${input}`,
            docs,
        });
        return response.text;
    }
);