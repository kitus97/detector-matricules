import numpy as np
import cv2
import os
import logging
import csv

from src.detector import plate_detector
from src.segmenter import segmenter_test
from src.ocr import ocr_test
from src.types import LicensePlate, PlateDetection 

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    csv_path = "data/a_paths.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: No s'ha trobat el fitxer CSV a {csv_path}")
        return

    # Obrim el CSV i llegim les files
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            img_id = row['id']
            image_path = row['image_path']
            
            print(f"\n--- Processant imatge {img_id}: {image_path} ---")
    
            # Comprovem si el fitxer existeix realment
            if not os.path.exists(image_path):
                print(f"Error: No s'ha trobat cap fitxer a: {image_path}")
                return

            # 2. Carregar la imatge
            # OpenCV carrega en BGR per defecte
            img = cv2.imread(image_path)
            
            if img is None:
                print("Error: No es pot carregar la imatge. Comprova el format (jpg, png...).")
                return

            print(f"Imatge carregada correctament: {img.shape[1]}x{img.shape[0]} píxels.")

            # 3. Crear l'objecte mestre de la matrícula
            matricula = LicensePlate(original_image=img)

            # 4. Executar la primera fase: Detecció
            print("Iniciant detecció de la ROI...")
            matricula.detection = plate_detector(matricula.original_image, img_id)

            # 5. Comprovar el resultat
            if matricula.detection.contour.any():
                print("Èxit! S'ha detectat un possible contorn de matrícula.")
                print(f"Vèrtexs trobats:\n{matricula.detection.contour}")
            else:
                print("No s'ha trobat cap polígon de 4 vèrtexs que sembli una matrícula.")

def pipeline_test(image: str) -> str:
    """
    Funció de prova per a la pipeline completa. Aquesta funció crida les funcions de prova del detector, segmentador i OCR en ordre i retorna el resultat final.

    Args:
        image (str): El nom o camí de la imatge d'entrada.
    Returns:
        str: El resultat final de la pipeline de prova.
    """
    # Crida al detector
    detected_image = plate_detector(image)
    print(f"Detector Test Result: {detected_image}")

    # Crida al segmentador
    segmented_image = segmenter_test(detected_image)
    print(f"Segmenter Test Result: {segmented_image}")

    # Crida al OCR
    ocr_result = ocr_test(segmented_image)
    print(f"OCR Test Result: {ocr_result}")

    return ocr_result


if __name__ == "__main__":
    main()
