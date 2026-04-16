import numpy as np
import logging as logger
import cv2
import os

from src.types import PlateDetection

def plate_detector(image: np.ndarray, image_name: str) -> PlateDetection:
    """
    Detector de matrícules. Rep una imatge d'entrada (matriu de píxels) i retorna
    la Bounding Box de la matrícula detectada amb els següents passos.

    1. Convertim a escala de grisos i apliquem filtre Bilateral per reduir el soroll.
    2. Apliquem Canny per obtenir un mapa de vores binari.
    3. Apliquem findContours per trobar les formes a la imatge.
    4. Per a cada contorn, aproximem la seva forma a un polígon amb approxPolyDP i busquem quadrilàters.

    Args:
        image (numpy.ndarray): La imatge d'entrada.
    Returns:
        PlateDetection: L'objecte de detecció de matrícula.
    """
    
    logger.info("Iniciant el detector de matrícules...")
    imatge_display = image.copy()  # Còpia de la imatge per a mostrar els resultats de debug

    logger.debug("Canviem a escala de grisos i apliquem filtre Bilateral...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Debugging - Deteccio Matricula", gray)
    cv2.waitKey(0)  # Pausa l'execució fins que prems una tecla
    cv2.destroyAllWindows()  # Tanca la finestra neta

    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    cv2.imshow("Debugging - Deteccio Matricula", blur)
    cv2.waitKey(0)  # Pausa l'execució fins que prems una tecla
    cv2.destroyAllWindows()  # Tanca la finestra neta

    logger.debug("Apliquem Canny per obtenir les vores...")
    edges = cv2.Canny(blur, 20, 200)

    cv2.imshow("Debugging - Deteccio Matricula", edges)
    cv2.waitKey(0)  # Pausa l'execució fins que prems una tecla
    cv2.destroyAllWindows()  # Tanca la finestra neta

    logger.debug("Busquem contorns a la imatge...")
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    logger.debug(f"Contorns trobats: {len(contours)}. Analitzant els més grans...")

    logger.debug("Analitzem els contorns per trobar quadrilàters...")
    plate_contour = None
    candidats = []

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        logger.debug(f"Contorn amb {len(approx)} vèrtexs detectat.")

        if len(approx) == 4:
            # Calculem el Bounding Box recte (Sense rotació) per extreure l'Aspect Ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / float(h)
            area = cv2.contourArea(approx)
            
            # Filtrem per proporció (ex: entre 1.5 i 7.0) i per una àrea mínima per evitar soroll
            if 1.5 <= aspect_ratio <= 7.0 and area > 600:
                logger.debug(f"Candidat vàlid trobat -> AR: {aspect_ratio:.2f}, Àrea: {area}")
                candidats.append({
                    "contorn": approx,
                    "ar": aspect_ratio,
                    "area": area,
                    "box": (x, y, w, h)
                })

    if len(candidats) == 1:
        plate_contour = candidats[0]["contorn"]
        logger.info("1 sola matrícula candidata detectada. Adjudicada!")
    elif len(candidats) > 1:
        logger.info(f"Conflicte: {len(candidats)} candidats trobats. Aplicant estratègia de selecció...")
        
        # Estratègia: Triar el que tingui l'Aspect Ratio més proper a l'estàndard europeu (4.72)
        AR_IDEAL = 4.72
        
        # Ordenem la llista basada en qui té el menor error respecte a l'AR ideal
        candidats_ordenats = sorted(candidats, key=lambda c: abs(c["ar"] - AR_IDEAL))
        
        # Ens quedem amb el millor (el primer de la llista ordenada)
        millor_candidat = candidats_ordenats[0]
        plate_contour = millor_candidat["contorn"]
        
        logger.info(f"Seleccionat el candidat amb AR {millor_candidat['ar']:.2f} (Diferència de {abs(millor_candidat['ar'] - AR_IDEAL):.2f})")

    
    if plate_contour is None:
        logger.warning("No s'ha detectat cap matrícula a la imatge.")
        return PlateDetection(
            contour=np.array([]),
            aligned_image=image,
            binary_image=image
        )
    else:
        logger.info("Matrícula detectada i validada!")
        cv2.drawContours(imatge_display, [plate_contour], -1, (0, 255, 0), 3)
        # Mostrem la imatge amb el contorn (si l'ha trobat)
        cv2.imshow("Debugging - Deteccio Matricula", imatge_display)
        print("Prem qualsevol tecla a la finestra de la imatge per continuar...")
        cv2.waitKey(0)  # Pausa l'execució fins que prems una tecla
        cv2.destroyAllWindows()  # Tanca la finestra neta
        if not os.path.exists("test"):
            os.makedirs("test")

        # Generem el path de sortida (ex: test/det_eu1.jpg)
        output_path = os.path.join("test", f"det_{image_name}.jpg")
        logger.info(f"Guardant la imatge de debug a: {output_path}")
        cv2.imwrite(output_path, imatge_display)
        
        # Opcional: imprimir si ha trobat alguna cosa per consola
        status = "OK"
        print(f"  [Detector] Guardat: {output_path} ({status})")
    
    
    return PlateDetection(
        contour=plate_contour,
        aligned_image=image,
        binary_image=gray
    )