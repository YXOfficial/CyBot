import sys
import os
import faiss
import tiktoken

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader
from llama_index.core import load_index_from_storage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from dotenv import load_dotenv
load_dotenv()

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
    sys.exit(1)

DOC_FILE_PATH = "13_2023_ND-CP_465185.txt"
PERSIST_DIR = "./storage_index"

FAISS_INDEX_PATH = os.path.join(PERSIST_DIR, "faiss_index.bin") # ADD THIS LINE

embedding_dimension = 768

embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
Settings.embed_model = embed_model
# Settings.chunk_size = 512
# Settings.chunk_overlap = 20

llm_model_to_use = "gemini-2.5-flash-preview-05-20"
llm = Gemini(model=llm_model_to_use, temperature=0.1)
Settings.llm = llm
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode, # Using a common tokenizer. Gemini specific might be "gemini" or need a different approach if tiktoken doesn't support it directly. For now, this is a reasonable proxy if exact Gemini tokenizer isn't exposed.
    verbose=False # Set to True for more detailed logging from the handler itself
)

Settings.callback_manager = CallbackManager([token_counter])
index = None # Initialize index variable

# Check if the persisted index exists
if not os.path.exists(PERSIST_DIR):
    print(f"Index not found in '{PERSIST_DIR}'. Creating new index...")

    # Ensure the document file exists
    if not os.path.exists(DOC_FILE_PATH):
        print(f"Error: Document file '{DOC_FILE_PATH}' not found.")
        print("Please ensure the .doc file is in the correct directory.")
        sys.exit(1)

    try:
        reader = SimpleDirectoryReader(input_files=[DOC_FILE_PATH])
        documents = reader.load_data()
        print(f"Đã tải {len(documents)} tài liệu từ '{DOC_FILE_PATH}'.")

    except Exception as e:
        print(f"Lỗi khi tải hoặc xử lý tài liệu '{DOC_FILE_PATH}': {e}")
        print("Vui lòng kiểm tra lại định dạng tệp hoặc quyền truy cập.")
        sys.exit(1)

    # Initialize Faiss and LlamaIndex Vector Store
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create the VectorStoreIndex from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # Create the persistence directory if it doesn't exist
    os.makedirs(PERSIST_DIR, exist_ok=True)
    # Persist the Faiss index
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    # Persist the LlamaIndex storage context (which includes the vector store metadata)
    storage_context.persist(persist_dir=PERSIST_DIR)

    print(f"Index đã được tạo và lưu vào '{PERSIST_DIR}'.")

else:
    print(f"Loading index from '{PERSIST_DIR}'...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    # Create the FaissVectorStore with the loaded index
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # Load the existing storage context, providing the vector store
    storage_context = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR,
        vector_store=vector_store # Crucial: pass the loaded vector store
    )
    # Load the index from the loaded storage context
    index = load_index_from_storage(storage_context=storage_context)
    print("Index đã được tải.")

# The rest of your code remains the same
query_engine = index.as_query_engine()
def rag_query(user_query: str):
    print(f"\nInput (user): {user_query}")

    # Thực hiện truy vấn
    response = query_engine.query(user_query)

    # Log các tài liệu được truy xuất
    if response.source_nodes:
        print("Log: Retrieved Nodes Info:")
        for i, node in enumerate(response.source_nodes):
            source_id = node.metadata.get('file_name', f"Doc {i + 1}")  # Use file_name if available, otherwise Doc X
            node_text = node.get_text()  # Get the text content of the node
            print(f"  Node {i + 1} (Source: {source_id}):")
            print(f"    Content (first 200 chars): {node_text[:200]}...")  # Print a snippet
            print(f"    Score: {node.score:.2f}")  # Also print retrieval score if available (useful for debugging)
            # You can print more of the content if needed, e.g., node_text
    else:
        print("Log: No relevant documents found.")

    # In câu trả lời từ AI
    print("AI:")
    print(str(response))

print("Nhập câu hỏi của bạn (hoặc 'exit'/'quit' để thoát):")

while True:
    user_input = input("Bạn: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Đang thoát hệ thống RAG. Tạm biệt!")
        break
    if not user_input:
        continue # Bỏ qua nếu người dùng nhập rỗng

    rag_query(user_input)
    print("\n" + "-" * 50) # Dấu phân cách giữa các lần truy vấn
    print("\n--- Current LLM Token Usage (since last reset) ---")
    print(f"  Prompt Tokens: {token_counter.prompt_llm_token_count}")
    print(f"  Completion Tokens: {token_counter.completion_llm_token_count}")
    print(f"  Total LLM Tokens: {token_counter.total_llm_token_count}")
    print(f"  Embedding Tokens: {token_counter.total_embedding_token_count}")
    token_counter.reset_counts()