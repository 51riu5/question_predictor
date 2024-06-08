import nltk
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize

class QGPipeline:
    def _init_(self, model_name='valhalla/t5-base-qg-hl', use_cuda=True):
        """
        Initializes the Question Generation Pipeline.

        Args:
            model_name (str): Name of the pre-trained model to use.
            use_cuda (bool): Whether to use CUDA (GPU) for inference.
        """
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def generate(self, text, num_beams=5, num_return_sequences=5, max_length=64):
        """
        Generates questions from the input text.

        Args:
            text (str): Input text to generate questions from.
            num_beams (int): Number of beams for beam search.
            num_return_sequences (int): Number of sequences to return.
            max_length (int): Maximum length of the generated sequences.

        Returns:
            list: List of generated questions.
        """
        sentences = sent_tokenize(text)
        questions = []
        for sentence in sentences:
            inputs = self._prepare_inputs_for_qg_from_text(sentence)
            input_ids = self.tokenizer.encode(inputs, return_tensors='pt').to(self.device)
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True
            )
            generated_questions = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            questions.extend(self._unique_questions(generated_questions, questions))
        return questions

    def _prepare_inputs_for_qg_from_text(self, text):
        """
        Prepares input for question generation from the given text.

        Args:
            text (str): Input text.

        Returns:
            str: Prepared input for question generation.
        """
        inputs = "context: {} </s> ".format(text.strip())
        return inputs

    def _unique_questions(self, new_questions, existing_questions):
        """
        Filters out duplicate questions.

        Args:
            new_questions (list): List of newly generated questions.
            existing_questions (list): List of existing questions.

        Returns:
            list: Unique questions.
        """
        unique_questions = []
        for question in new_questions:
            if question not in existing_questions and question not in unique_questions:
                unique_questions.append(question)
        return unique_questions

# Initialize the question generation pipeline
qg = QGPipeline()

# Input previous year questions from the user
print("Enter previous year questions (Enter 'done' when finished):")
prev_year_questions = ""
while True:
    line = input()
    if line.lower() == 'done':
        break
    prev_year_questions += line + "\n"

# Input text from the user
text = input("Enter your notes/text: ")

# Combine previous year questions and notes
combined_text = prev_year_questions + text

# Generate questions
questions = qg.generate(combined_text)

# Print the generated questions
print("\nGenerated Questions:")
for i, question in enumerate(questions):
    print(f"Q{i+1}: {question}")