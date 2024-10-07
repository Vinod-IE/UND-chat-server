from timings import time_it, logger
from openai import OpenAI
from settings import Settings
import tiktoken
import time

class Responder:
    def __init__(self):
        self.client = OpenAI(api_key=Settings.OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8192

    def _trim_context(self, context, query, max_tokens):
        sys_prompt = """You are a knowledgeable staff member and administrative expert at the University of North Dakota (UND), with in-depth understanding of academic programs, policies, campus resources, and services. Your role is to assist students, faculty, and staff by providing accurate, up-to-date, and detailed information on all aspects of UND, including admissions, academic guidelines, student life, faculty concerns, and administrative procedures. 
                    Your responses should be clear, professional, and free from unnecessary elaboration, tailored specifically to the needs of the requester."""
        query_prompt = f"""
        Question: {query}
        Please provide a direct and specific answer to the question, without using phrases like 'based on the given information or context' or 'according to the context.' If certain details are not available, mention it clearly only once and avoid elaborating on missing information.
        """

        
        fixed_tokens = len(self.encoder.encode(sys_prompt + query_prompt))
        available_tokens = max_tokens - fixed_tokens - 1000  # Reserve tokens for the response

        context_tokens = self.encoder.encode(context)
        if len(context_tokens) > available_tokens:
            context_tokens = context_tokens[:available_tokens]
        
        return self.encoder.decode(context_tokens)

    @time_it
    def respond(self, query, context):
        max_retries = 3
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                trimmed_context = self._trim_context(context, query, self.max_tokens)
                
                prompt = f"""
                You are a knowledgeable staff member at the University of North Dakota. Your goal is to provide direct, accurate, and concise responses.

                Here's the information available to answer the question:

                {trimmed_context}

                Current question: {query}

                Please follow these guidelines:
                1. Respond directly to the specific question asked, without using phrases like 'based on the given information' or 'according to the context.'
                2. Provide accurate and relevant information from what's available to you.
                3. If specific details (such as exact figures) are not available, mention this clearly and only once.
                4. Avoid giving general or unrelated information unless it directly answers the question.
                5. Provide detailed explanations if the question requires it; otherwise, keep the answer concise and precise.
                6. Speak as if you are directly communicating with the person asking the question.
                7. Avoid repeating the same information multiple times. If specific information is not available, mention it once.
                8. Do not use any phrases like 'Additional Information,' 'given context,' or similar expressions. Just provide the actual answer.
                9. Remember, you are responding as a knowledgeable staff member, not as an AI or a third party reading from a database.
                """


                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a knowledgeable staff member at the University of North Dakota. Provide direct, accurate, and concise responses, without using any phrases such as 'based on the given information or context' or 'according to the context.' Just give the answer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=4000 
                )

                answer = response.choices[0].message.content
                logger.info(f"Generated response for: '{query}'")
                return answer.strip()
            except Exception as e:
                if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Error generating response: {str(e)}")
                    return "I'm sorry, but I can't access that information right now. Could you please ask your question again in a moment?"

        return "I apologize, but I'm having some technical issues at the moment. Could you please try asking your question again later?"
