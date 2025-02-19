from transformers import pipeline

def classify_document(text, candidate_labels=["legal", "academic", "news", "blog", "fiction"]):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels)
    return result

if __name__ == "__main__":
    text = input("Enter document text: ")
    classification = classify_document(text)
    print("Classification:", classification)
