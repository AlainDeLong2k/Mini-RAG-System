from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


# Define schema for evaluation output
class EvalScore(BaseModel):
    score: int = Field(description="Score from 0 to 1")
    reasoning: str = Field(description="Reasoning for the score")


class RAGEvaluator:
    def __init__(self, model_name="llama3.2"):
        # self.llm = ChatOllama(model=model_name, temperature=0)  # Temp=0 for consistency
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b", temperature=0
        )  # Temp=0 for consistency

    def evaluate_faithfulness(self, query, answer, context):
        """
        Evaluates if the answer is grounded in the retrieved context.
        (Hallucination Check)
        """
        parser = JsonOutputParser(pydantic_object=EvalScore)

        prompt = ChatPromptTemplate.from_template(
            """You are a strict grader evaluating a RAG system.
            
            Check if the Answer is derived ONLY from the Context provided.
            
            Context: {context}
            Answer: {answer}
            
            Return a JSON with:
            - "score": 1 if the answer is fully supported by context, 0 otherwise.
            - "reasoning": A brief explanation.
            
            {format_instructions}
            """
        )

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke(
                {
                    "context": context,
                    "answer": answer,
                    "format_instructions": parser.get_format_instructions(),
                }
            )
            return result
        except Exception as e:
            return {"score": -1, "reasoning": f"Eval failed: {str(e)}"}
