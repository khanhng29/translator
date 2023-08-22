from flask import Flask, request, render_template, Response, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ForCTC, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, AutoModelForTokenClassification, pipeline
from datasets import load_dataset
import sounddevice as sd
import soundfile as sf
from gtts import gTTS
import torch
import io


app = Flask(__name__)

trans_model_name = "VietAI/envit5-translation"
trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name)
trans_model = AutoModelForSeq2SeqLM.from_pretrained(trans_model_name)

stt_vi = Wav2Vec2Processor.from_pretrained(
    "nguyenvulebinh/wav2vec2-base-vietnamese-250h")
model_stt_vi = Wav2Vec2ForCTC.from_pretrained(
    "nguyenvulebinh/wav2vec2-base-vietnamese-250h")

stt_en = Wav2Vec2Processor.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-english")
model_stt_en = Wav2Vec2ForCTC.from_pretrained(
    "jonatasgrosman/wav2vec2-large-xlsr-53-english")

tts_en = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_tts_en = SpeechT5ForTextToSpeech.from_pretrained(
    "microsoft/speecht5_tts")
embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors", split="validation")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

ner_tokenizer_vi = AutoTokenizer.from_pretrained(
    "NlpHUST/ner-vietnamese-electra-base")
ner_model_vi = AutoModelForTokenClassification.from_pretrained(
    "NlpHUST/ner-vietnamese-electra-base")
ner_pipeline_vi = pipeline("ner", model=ner_model_vi,
                           tokenizer=ner_tokenizer_vi)

ner_tokenizer_en = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_model_en = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER")
ner_pipeline_en = pipeline("ner", model=ner_model_en,
                           tokenizer=ner_tokenizer_en)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate")
def translate():
    text = request.args.get("text")
    source_lang = request.args.get("source_lang")
    target_lang = request.args.get("target_lang")
    # app.logger.info(f"Source Language: {source_lang}")

    inputs = [f"{source_lang}: {text}"]
    input_ids = trans_tokenizer(
        inputs, return_tensors="pt", padding=True).input_ids

    trans_model.to('cpu')

    outputs = trans_model.generate(input_ids, max_length=512)
    decoded_outputs = trans_tokenizer.batch_decode(
        outputs, skip_special_tokens=True)

    translated_text = ""
    if target_lang == "vi" and source_lang != "vi":
        translated_text = decoded_outputs[0].split("vi:", 1)[1]
    elif target_lang == "en" and source_lang != "en":
        translated_text = decoded_outputs[0].split("en:", 1)[1]

    return translated_text


@app.route("/record_and_predict")
def record_and_predict():
    # Record audio from the microphone
    audio = record_audio()

    source_lang = request.args.get("source_lang")

    if source_lang == "vi":
        transcription = process_audio(audio, stt_vi, model_stt_vi)
    elif source_lang == "en":
        transcription = process_audio(audio, stt_en, model_stt_en)
    else:
        transcription = ["UNKNOWN"]

    return transcription[0]


def process_audio(input_audio, processor, model):
    input_audio = input_audio.astype("float32")  # Convert to float32
    input_values = processor(
        input_audio, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription

# Define a function to record audio from the microphone


def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()  # Wait for recording to finish
    print("Recording finished.")
    return audio.flatten()


@app.route("/speak")
def speak():
    text = request.args.get("text")
    tts = gTTS(text, lang='vi')

    # Lưu dữ liệu gTTS vào tệp tin audio.mp3
    tts.save("vi_out_audio.mp3")

    # Đọc dữ liệu từ tệp tin audio.mp3
    with open("vi_out_audio.mp3", "rb") as audio_file:
        audio_data = audio_file.read()

    # Trả về dữ liệu như một response có kiểu 'audio/mpeg'
    return Response(audio_data, mimetype="audio/mpeg")


@app.route("/speak_english")
def speak_english():
    text = request.args.get("text")

    # Xử lý văn bản tiếng Anh thành giọng nói
    inputs = tts_en(text=text, return_tensors="pt")
    speaker_embeddings = torch.tensor(
        embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    spectrogram = model_tts_en.generate_speech(
        inputs["input_ids"], speaker_embeddings)

    with torch.no_grad():
        speech = vocoder(spectrogram)

    # Ghi âm thanh vào tệp tin
    output_path = "en_out_audio.wav"
    sf.write(output_path, speech.numpy(), samplerate=16000)

    with open(output_path, "rb") as audio_file:
        audio_data = audio_file.read()

    # Trả về dữ liệu âm thanh như một response có kiểu 'audio/wav'
    return Response(audio_data, mimetype="audio/wav")


@app.route("/extract_entities")
def extract_entities():
    text = request.args.get("text")
    # Thực hiện trích xuất thực thể từ văn bản
    entities = extract_entities_from_text(text)
    return jsonify(entities)


def extract_entities_from_text(text):
    target_lang = request.args.get("target_lang")

    if target_lang == "vi":
        ner_results = ner_pipeline_vi(text)
    elif target_lang == "en":
        ner_results = ner_pipeline_en(text)

    extracted_entities = []
    current_entity = None
    current_tokens = []

    for result in ner_results:
        word = result['word']
        entity = result['entity']
        if entity.startswith("B-"):
            if current_entity is not None:
                entity_string = ' '.join(current_tokens)
                if entity_string:
                    extracted_entities.append(
                        (current_entity[2:], entity_string))
            current_entity = entity
            current_tokens = [word]
        else:
            current_tokens.append(word)

    if current_entity is not None:
        entity_string = ' '.join(current_tokens)
        if entity_string:
            extracted_entities.append(
                (current_entity[2:], entity_string))

    return extracted_entities


if __name__ == "__main__":
    app.run(debug=False)
