from fastapi import FastAPI, File , UploadFile
from fastapi.responses import JSONResponse,RedirectResponse
from transformers import ViltProcessor,ViltForQuestionAnswering
from PIL import Image
import io


app = FastAPI(title="Visual Question and Answering API", version="0.0.1")

#Initializing Tokenizer and Model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def generate_answer(image,txt):
    try:
        # Load and pre-process the image to RGB format
        img = Image.open(io.BytesIO(image)).convert("RGB")

        # Prepare inputs
        encoding = processor(img,txt,return_tensors = 'pt')

        #Forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer
    except Exception as e:
        return str(e)
    

@app.get("/",include_in_schema=False)
async def index():
    return RedirectResponse(url='/docs')

@app.post("/answer")
async def process_image(image: UploadFile= File(...),text: str = None):
    try:
        answer = generate_answer(await image.read(),text)
        return JSONResponse({"Answer":answer})
    except Exception as e:
        return JSONResponse({"Error":str(e)})
