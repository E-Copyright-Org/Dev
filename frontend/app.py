import transformers
from transformers import pipeline

# Choose an LLM model (consider task-specific models if available)
model_name = "facebook/bart-large-mnli"

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model_name)


def read_file(file_path):
    """Reads content from a text file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2 (optional)."""
    try:
        import PyPDF2

        pdf_file = open(pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        pdf_file.close()
        return text
    except ImportError:
        print("Error: PyPDF2 library not installed. Install using 'pip install PyPDF2'.")
        return None
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None


def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file using docx2text (optional)."""
    try:
        import docx2text

        with open(docx_path, 'rb') as file:
            text = docx2text.process(file)
        return text
    except ImportError:
        print("Error: docx2text library not installed. Install using 'pip install docx2text'.")
        return None
    except Exception as e:
        print(f"Error processing DOCX: {e}")
        return None


def answer_question(question, file_content):
    """Answers a question using the LLM model and provided file content."""
    if not file_content:
        print("Error: Unable to read file content.")
        return

    context = file_content
    query = question

    try:
        answer = qa_pipeline({"context": context, "question": query})
        answer_text = answer['answer']
        answer_score = answer['score']
        print(f"DISCLAIMER: This LLM model's answers may not be accurate for legal purposes. Consult legal professionals for copyright advice.")
        print(f"Answer: {answer_text} (Confidence: {answer_score:.4f})")
    except Exception as e:
        print(f"Error processing question: {e}")


def main():
    """Prompts user for file path and questions, handles input."""
    file_path = input("Enter the path to the text file (or a PDF/DOCX file): ")
    file_type = file_path.split('.')[-1].lower()  # Get file extension (lowercase)

    if file_type == 'txt':
        file_content = read_file(file_path)
    elif file_type in {'pdf', 'docx'}:
        if file_type == 'pdf':
            file_content = extract_text_from_pdf(file_path)
        else:
            file_content = extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_type}. Only TXT, PDF, and DOCX files are supported.")
        return

    while True:
        question = input("Ask your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        answer_question(question, file_content)


if __name__ == "__main__":


    main()
