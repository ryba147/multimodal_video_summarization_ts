import os
from transformers import AutoTokenizer, AutoModelWithLMHead

from config import BASE_DIR

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelWithLMHead.from_pretrained("google/flan-t5-base", return_dict=True)

transcripts_folder = BASE_DIR + "/transcripts/TVSum"

file_names = os.listdir(transcripts_folder)

for file_name in file_names:
    file_path = os.path.join(transcripts_folder, file_name)

    with open(file_path) as f:
        input_text = f.read()

    input_ids = tokenizer.encode("summarize " + input_text, return_tensors="pt")

    summary_ids = model.generate(
        input_ids,
        max_length=100,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("File:", file_name)
    print("Summary:", summary)
    print()
