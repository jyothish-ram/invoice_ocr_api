# Invoice OCR API

This project is an ocr data extraction from Invoices or bills. THis project utilizes Tensorflow, Pytesseract, Ollama, Gemma_2b.

## Identifiable parameters

`Company Name`, 
`Company Address`, 
`Customer Name`, 
`Customer Address`, 
`Invoice Number`, 
`Invoice Date`, 
`Due Date`, 
`Description`, 
`Quantity`, 
`Unit Price`, 
`Taxes`, 
`Amount`, 
`Total`

## Working

This project mainly consists of three parts
1. Tensorflow Model: this model finds the roi(region of interest) from the invoice Image. the roi is given to tesseract.
2. Tesseract: Pytessract is used to extract text from the image(ocr engine).
3. NLP: Gemma_2b is used with Ollama for NLP which corrects the text extracted by tesseract 

> [!NOTE]
> - tensorflow model is stored in `models/saved_model` folder

## Installation

> [!NOTE]
> - `python3 -m venv venv` to create python virtual env
> - `./venv/scripts/activate` to activate venv in windows or `source venv/bin/activate` in linux
> - `run pip install -r requirements.txt` to install necessary packages
> - Need to in install Ollama as per ollama documentation, for linux `curl -fsSL https://ollama.com/install.sh | sh`
> - run `ollama run gemma2:2b` to download NLP Gemma_2b model.
> - To run the program `python app.py`

### API Request Model

Sample API Request Model(POST) body:

```
    {
    "image": "{image in base64 Format}"

    } 
```

### API Response Model

Sample API Response:

```
{
    "Company Name": "TEMPUSTIC CONSULTORIA TECNOLOGICA SL",
    "Company Address": "C/ PIE DE ALTAR N° 7\n28229 VILLANUEVA DEL PARRDILLO\nMADRID",
    "Customer Name": "SM TECNOLOGIA, S.L.U.",
    "Customer Address": "Poligono Industrial Os Airios, Sector 2 - Parcela 4\n15320 As Pontes\nA Corufia",
    "Invoice Number": "2023.11",
    "Invoice Date": "31/05/2023",
    "Due Date": null,
    "Description": "Hora Programador Java Junior",
    "Quantity": 30,
    "Unit Price": 176.00,
    "Taxes": 1108.80,
    "Amount": 5280.00,
    "Total": 6388.80
}
```
