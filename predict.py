from cog import BasePredictor, Input
from transformers import pipeline

class Predictor(BasePredictor):
    def setup(self):
        self.generator = pipeline("text-generation", model="gpt2")

    def predict(self, prompt: str = Input(description="Texte à compléter")) -> str:
        result = self.generator(prompt, max_length=50, num_return_sequences=1)
        return result[0]["generated_text"]
