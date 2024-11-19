from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import sqlite3

app = FastAPI()

# Configuración de la base de datos
def init_db():
    conn = sqlite3.connect('conversations.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            response_text TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Carga el pipeline de Hugging Face para generación de texto
text_generator = pipeline("text-generation", model="gpt2")

# Definimos el modelo para el cuerpo de la solicitud
class TextRequest(BaseModel):
    input_text: str

@app.post("/process-text/")
async def process_text(request: TextRequest):
    try:
        # Genera una respuesta basada en el texto proporcionado
        response = text_generator(request.input_text, max_length=50, num_return_sequences=1)
        response_text = response[0]['generated_text']
        
        # Guardar la conversación en la base de datos
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO conversations (input_text, response_text) VALUES (?, ?)', 
                       (request.input_text, response_text))
        conn.commit()
        conn.close()

        return {"input": request.input_text, "response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/")
async def get_conversations():
    """
    Endpoint para obtener todas las conversaciones almacenadas.
    """
    try:
        conn = sqlite3.connect('conversations.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, input_text, response_text FROM conversations')
        conversations = cursor.fetchall()
        conn.close()

        return {
            "conversations": [
                {"id": row[0], "input": row[1], "response": row[2]} for row in conversations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
