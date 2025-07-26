
import gradio as gr
import pickle

# Load the trained model and TF-IDF vectorizer
with open("clf.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf.pkl", "rb") as tfidf_file:
    vectorizer = pickle.load(tfidf_file)

# Prediction function
def predict(text):
    transformed_text = vectorizer.transform([text])
    pred = model.predict(transformed_text)[0]
    label = "AI-Generated" if pred == 1 else "Human-Written"
    return label

# Define Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, placeholder="Enter your text here...", label="Input Text"),
    outputs=gr.Textbox(label="Prediction"),
    title="🧠 AI vs Human Text Classifier",
    description="""
    This tool uses a trained ML model to determine whether a given passage of text is **AI-generated** or **Human-written**.
    """,
    theme="soft",
    examples=[
        ["Artificial Intelligence is transforming the way we live and work..."],
        ["I remember the days when cars were rare and roads were quiet..."],
        ["ChatGPT is an example of a large language model developed by OpenAI..."]
    ]
)

# Launch app
if __name__ == "__main__":
    interface.launch()
