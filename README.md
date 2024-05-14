

## LlamaIndex_OpenAI_RAG

LLM chatbot using RAG methodology and based on OpenAI
The Retrieval-Augmented Generation (RAG) model first retrieves relevant information from the PDF documents stored in the "PDFs" folder using vector search techniques.[1] 
This retrieved information is then used as context to augment the user's query, providing the language model with relevant background knowledge from the PDFs.[2] 
The language model can then generate a more informed response by considering both the query and the retrieved context from the PDF documents.[3][4] 
This RAG approach allows the model to leverage external data sources like PDFs to enhance its knowledge and provide more accurate and substantive answers tailored to the specific domain covered by the ingested documents.[5] 
We used Streamlit to build our webapp and the chat interface [6]

Citations:
- [1] https://github.com/couchbase-examples/rag-demo-llama-index
- [2] https://www.latent.space/p/llamaindex
- [3] https://www.youtube.com/watch?v=hH4WkgILUD4
- [4] https://github.com/BrenoAV/RAG-llama-index-openai
- [5] https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/
- [6] https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/

### Requirements

To run this project, you need to install the following dependencies:

```
python-dotenv
streamlit
openai
llama-index
chromadb
```

You can install these dependencies using pip:

```
pip install python-dotenv streamlit openai llama-index chromadb
```

### Setup

This project is built using Python 3.10. To run it, you'll need to have Python 3.10 installed on your system.
1. Set up a virtual environment with Python 3.10 (should work with an higher version as well)
2. Create a `.env` file in the project root directory.
3. Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY="<insert_your_own_key>"
   ```
4. Add pdf documents to a file folder on your system, and update the path for your PDFs folder on line 62 of the **streamlit_app.py** `input_dir="../PDFs"`

5. Run streamlit from the integrated terminal with:
```
streamlit run streamlit_app.py
```

