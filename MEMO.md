## Bilateral Filtering vs Gaussian filter

L'ús del filtre bilateral en lloc del gaussià per a la detecció de matrícules ofereix un avantatge crític: la **preservació de les vora de la imatge**. Mentre que el filtre gaussià és un filtre lineal que difumina la imatge de manera uniforme basant-se només en la distància física entre píxels, el filtre bilateral és una tècnica no lineal que considera també la diferència d'intensitat (color o brillantor). Això vol dir que el filtre gaussià tendeix a suavitzar les vores dels caràcters de la matrícula, barrejant els valors dels píxels dels números amb els del fons i fent que la identificació sigui més difícil.


Per contra, el filtre bilateral només permet que un píxel n'influenciï un altre si, a més d'estar a prop, tenen un valor similar. En el context d'una matrícula, això permet eliminar el soroll de la imatge sense difuminar el contrast nítid entre els caràcters i el fons. En conservar aquestes vores afilades, els algorismes de reconeixement de caràcters (OCR) poden segmentar i identificar els números i les lletres amb molta més precisió, ja que l'estructura geomètrica essencial de la informació es manté intacta malgrat el suavitzat.

## Implementació 16/04

El desenvolupament del sistema de visió per computador s'ha iniciat amb la creació d'un mòdul robust per a la localització de la regió d'interès. En la primera iteració, es va dissenyar una pipeline de processament d'imatge que començava amb l'aplicació d'un filtre bilateral per reduir el soroll de la carrosseria mantenint la nitidesa de les vores, seguit de l'algoritme de Canny per a l'extracció de contorns. Mitjançant la funció de cerca de contorns d'OpenCV i l'aproximacióDP, el sistema identificava qualsevol polígon tancat que presentés exactament quatre vèrtexs, classificant-lo inicialment com una matrícula.

Davant la presència de nombrosos falsos positius en entorns complexos, es va evolucionar la lògica de detecció incorporant criteris geomètrics més estrictes per filtrar els candidats. S'ha integrat un control de la relació d'aspecte que només accepta polígons amb un ràtio d'amplada per alçada situat entre 2 i 6, a més d'exigir una àrea mínima de 1000 píxels per descartar el soroll de fons. En els casos on el programa detecta múltiples zones que compleixen aquests requisits, s'ha implementat una estratègia de selecció basada en la proximitat estadística a l'ideal europeu, fixat en un ràtio de 4.72, escollint el candidat que presenta la menor desviació absoluta.

Tot i les millores geomètriques, els resultats actuals demostren que el sistema encara és vulnerable a confusions amb elements del vehicle com reixetes de ventilació o llums de frenada que mimetitzen la forma de la placa. Com a via de millora per a les properes fases, s'ha plantejat la substitució del detector de Canny per un operador de Sobel vertical. Aquesta tècnica podria ser clau per ressaltar la densitat de canvis de contrast propis dels caràcters alfanumèrics, permetent així una discriminació més efectiva entre les formes rectangulars buides i les matrícules reals.

### Visual

# Progrés del Sistema de Detecció de Matrícules

---

## 1. Primera Fase: Estructura de Detecció Base
La pipeline inicial es va centrar en la geometria pura per localitzar la placa dins la imatge original:
* **Filtratge Bilateral:** S'aplica per suavitzar les textures de la carrosseria i l'asfalt sense perdre la definició de les vores de la matrícula.
* **Detector de Canny:** Algoritme per extreure un mapa de vores binari.
* **Cerca de Contorns:** Ús de `findContours` per identificar totes les formes tancades de l'escena.
* **Aproximació Poligonal:** Filtre basat en el nombre de vèrtexs; si el contorn es podia simplificar a exactament **4 punts**, es marcava com a matrícula.



---

## 2. Segona Fase: Refinament i Filtres Intel·ligents
Per reduir els errors, vam passar d'una detecció basada només en vèrtexs a una validació geomètrica avançada:
* **Relació d'Aspecte (Aspect Ratio):** S'ha limitat la cerca a polígons que tinguin una proporció d'entre **2.0 i 6.0** (amplada/alçada).
* **Filtre d'Àrea:** S'ignoren tots els objectes amb una superfície inferior a **1000 píxels** per eliminar el soroll.
* **Estratègia de Selecció:** En cas de trobar diversos candidats (com fars o reixetes), el sistema tria el que més s'aproxima al ràtio de **4.72**, que és l'estàndard de les matrícules europees.

---

## 3. Estat Actual i Propostes de Millora
Tot i la millora en la lògica, el sistema encara presenta falsos positius en elements repetitius o llums del vehicle. Les línies de treball futures són:
* **Sobel Vertical:** Investigar la substitució de Canny per Sobel en l'eix X. Això permetria ressaltar l'acumulació de línies verticals de les lletres i ignorar línies horitzontals del cotxe.
* **Validació de Contingut:** Cal investigar mètodes per comprovar si l'interior del rectangle conté prou variació de contrast per ser text real.