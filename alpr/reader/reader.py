"""
reader.py
=========
Postprocesado de los caracteres OCR para obtener la matrícula final.

Recibe una lista de caracteres predichos por la CNN y aplica correcciones
contextuales según el formato esperado: 4 números + 3 letras.
"""

import re


LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "A": "4",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
    "4": "A",
}


def clean_text(chars: list[str]) -> str:
    """
    Convierte la lista de caracteres en texto limpio.
    Elimina '?', espacios y símbolos no alfanuméricos.
    """
    text = "".join(chars).upper()
    text = text.replace("?", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def force_digit(ch: str) -> str:
    """
    Corrige un carácter cuando esperamos un número.
    """
    if ch.isdigit():
        return ch
    return LETTER_TO_DIGIT.get(ch, ch)


def force_letter(ch: str) -> str:
    """
    Corrige un carácter cuando esperamos una letra.
    """
    if ch.isalpha():
        return ch
    return DIGIT_TO_LETTER.get(ch, ch)


def correct_spanish_plate(text: str) -> str:
    """
    Aplica formato 4 números + 3 letras.
    """
    corrected = ""

    corrected += "".join(force_digit(ch) for ch in text[:4])
    corrected += "".join(force_letter(ch) for ch in text[4:7])

    return corrected


def read_plate(chars: list[str]) -> str:
    """
    Devuelve la matrícula final a partir de los caracteres OCR.

    Ejemplo:
        ["1", "2", "3", "4", "A", "B", "C"] -> "1234ABC"
        ["O", "2", "3", "A", "8", "C", "D"] -> "0234BCD"
    """

    text = clean_text(chars)

    if not text:
        return ""

    # Caso normal: matrícula española moderna
    if len(text) == 7:
        plate = correct_spanish_plate(text)

        if re.fullmatch(r"\d{4}[A-Z]{3}", plate):
            return plate

        return plate

    # Si hay más de 7 caracteres, buscamos una ventana válida
    if len(text) > 7:
        for i in range(len(text) - 6):
            window = text[i:i + 7]
            plate = correct_spanish_plate(window)

            if re.fullmatch(r"\d{4}[A-Z]{3}", plate):
                return plate

    # Si no encaja, devolvemos el texto limpio sin inventar demasiado
    return text