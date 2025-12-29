import time
from pipeline import SimpleRAGPipeline
from eval import RAGEvaluator


def main():
    # 1. Setup
    print("Initializing RAG Pipeline...")
    # Use a generic tech blog or documentation for testing
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

    rag = SimpleRAGPipeline()
    rag.ingest_data(url)
    rag.build_chain()

    evaluator = RAGEvaluator()

    # 2. Query
    # query = "What are the key components of an autonomous agent?"
    # print(f"\nUser Query: {query}")

    # start_time = time.time()
    # result = rag.run_query(query)
    # latency = time.time() - start_time

    # # 3. Output
    # print("\n--- Generated Answer ---")
    # print(result["answer"])

    # print(f"\n--- Sources Retrieved ({len(result['source_documents'])}) ---")
    # for i, doc in enumerate(result["source_documents"]):
    #     print(f"[{i+1}] {doc.page_content[:100]}...")

    # # 4. Evaluation (The 'Stretch' Goal included)
    # print("\n--- Running Evaluation ---")
    # eval_result = evaluator.evaluate_faithfulness(
    #     query=result["query"], answer=result["answer"], context=result["context_str"]
    # )

    # print(f"Faithfulness Score: {eval_result['score']}/1")
    # print(f"Reasoning: {eval_result['reasoning']}")
    # print(f"Latency: {latency:.2f}s")

    # 2. Loop Interactive Query
    print("\n-------------------------------------------------")
    print("System Ready! Type 'exit' or 'quit' to stop.")

    while True:
        query = input("\n>> Enter your question: ")

        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not query.strip():
            continue

        start_time = time.time()
        result = rag.run_query(query)
        latency = time.time() - start_time

        # 3. Output
        print("\n--- Generated Answer ---")
        print(result["answer"])

        print(f"\n--- Sources Retrieved ({len(result['source_documents'])}) ---")
        for i, doc in enumerate(result["source_documents"]):
            # In ra source ngắn gọn để dễ nhìn
            content_preview = doc.page_content[:100].replace("\n", " ")
            print(f"[{i+1}] {content_preview}...")

        # 4. Evaluation
        print("\n--- Running Evaluation ---")
        # In thông báo đang chấm điểm để user chờ
        print("Evaluating faithfulness...")

        eval_result = evaluator.evaluate_faithfulness(
            query=result["query"],
            answer=result["answer"],
            context=result["context_str"],
        )

        print(f"Faithfulness Score: {eval_result['score']}/1")
        print(f"Reasoning: {eval_result['reasoning']}")
        print(f"Latency: {latency:.2f}s")
        print("-" * 50)


if __name__ == "__main__":
    main()
