"""
BhojRAG Prompt Templates
===========================
Grounded prompting strategies for faithful RAG generation.
Designed to reduce hallucination and encourage evidence-based answers.

Templates:
    1. grounded_qa: Direct QA with strict faithfulness constraints
    2. chain_of_thought: Step-by-step reasoning over retrieved context
    3. hindi_bridge: Translate Bhojpuri context to Hindi for reasoning,
                     then produce Bhojpuri-aware responses
"""

from typing import List


class PromptTemplates:
    """
    Prompt template manager for RAG generation.

    Each template takes a query and list of context passages,
    returns a formatted prompt string ready for LLM consumption.
    """

    @staticmethod
    def grounded_qa(query: str, contexts: List[str]) -> str:
        """
        Standard grounded QA prompt.

        Instructs the LLM to answer strictly from the provided context
        and explicitly state when information is insufficient.
        """
        context_block = "\n\n".join(
            f"[Source {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        return f"""You are a knowledgeable assistant specializing in Bhojpuri language and culture.
Answer the question ONLY using the provided context passages.

STRICT RULES:
- Do NOT use any information outside the provided context.
- If the context does not contain enough information, say: "दिहल संदर्भ में ई जानकारी उपलब्ध नइखे।" (This information is not available in the given context.)
- Quote or paraphrase specific passages to support your answer.
- Keep your answer concise and directly relevant to the question.

CONTEXT:
{context_block}

QUESTION: {query}

ANSWER:"""

    @staticmethod
    def chain_of_thought(query: str, contexts: List[str]) -> str:
        """
        Chain-of-thought prompt for complex reasoning.

        Guides the LLM through step-by-step reasoning over
        multiple context passages before generating a final answer.
        """
        context_block = "\n\n".join(
            f"[Passage {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        return f"""You are an expert assistant for Bhojpuri language questions.

TASK: Answer the following question using ONLY the provided passages.

INSTRUCTIONS:
1. First, identify which passages are relevant to the question.
2. Extract key information from each relevant passage.
3. Reason step-by-step about how the information answers the question.
4. Provide a clear, concise final answer.
5. If the passages don't contain the answer, explicitly say so.

PASSAGES:
{context_block}

QUESTION: {query}

STEP-BY-STEP REASONING:
- Relevant passages:
- Key information:
- Reasoning:

FINAL ANSWER:"""

    @staticmethod
    def hindi_bridge(query: str, contexts: List[str]) -> str:
        """
        Hindi-bridge prompt for cross-lingual reasoning.

        Since LLMs typically reason better in Hindi than Bhojpuri,
        this prompt:
        1. Presents Bhojpuri context
        2. Asks the model to internally reason in Hindi
        3. Produces a response that is Bhojpuri-aware

        This is a key research contribution: using Hindi as a
        reasoning bridge for low-resource Bhojpuri.
        """
        context_block = "\n\n".join(
            f"[स्रोत {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
        )

        return f"""आप भोजपुरी भाषा और संस्कृति के विशेषज्ञ सहायक हैं।

नीचे भोजपुरी भाषा में कुछ संदर्भ दिए गए हैं। इन संदर्भों का उपयोग करके प्रश्न का उत्तर दें।

कार्य:
1. पहले संदर्भों को ध्यान से पढ़ें (ये भोजपुरी में हैं)
2. हिंदी में सोचते हुए उत्तर तैयार करें
3. उत्तर हिंदी या भोजपुरी में दें — जो भी स्वाभाविक हो
4. केवल दिए गए संदर्भों की जानकारी का उपयोग करें
5. अगर उत्तर संदर्भ में नहीं है, तो स्पष्ट रूप से बताएं

संदर्भ:
{context_block}

प्रश्न: {query}

उत्तर:"""

    @classmethod
    def get_template(cls, name: str):
        """
        Get a prompt template function by name.

        Args:
            name: One of "grounded_qa", "chain_of_thought", "hindi_bridge"

        Returns:
            Template function.
        """
        templates = {
            "grounded_qa": cls.grounded_qa,
            "chain_of_thought": cls.chain_of_thought,
            "hindi_bridge": cls.hindi_bridge,
        }
        if name not in templates:
            raise ValueError(
                f"Unknown template: {name}. " f"Available: {list(templates.keys())}"
            )
        return templates[name]
