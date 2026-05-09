Šis repozitorijs satur pirmkodu Latvijas Universitātes maģistra darbam 
**"Ģenētisko algoritmu stāvokļa diagnostika, izmantojot topoloģisko datu analīzi"** 
(Autors: Edijs Bergholcs). 

## Gatavie rezultāti un datu kopas
Lai ietaupītu skaitļošanas laiku, visi ģenerētie punktu mākoņi (csv faili) un
gatavie eksperimentu rezultāti (pickle faili) ir brīvi pieejami un lejupielādējami Google Drive mapē:

**[Skatīt un lejupielādēt gatavos rezultātus šeit](https://drive.google.com/drive/folders/1agHcghSgitDPPPbe0poUjRTrvLNmPgFm?usp=sharing)**

## Koda izmantošana un konfigurēšana
Šis kods ir izstrādāts kā pētniecības prototips maģistra darba eksperimentu vajadzībām. 
Lai gan tas nav optimizēts un strukturēts kā plaša patēriņa programmatūra, visi pētījumā aprakstītie eksperimenti ir pilnībā atkārtojami un pārbaudāmi.
Pēc noklusējuma ģenētisko algoritmu kodu (```tda_ml_ga.py```) var izpildīt tūlītēji, taču, lai pielāgotu eksperimentu parametrus vai lokālos ceļus, jāveic nelielas izmaiņas konkrētās koda rindās:
1. Direktoriju norādīšana: 221. un 303. koda rindā ir jānorāda atbilstošie ceļi datu lasīšanai un saglabāšanai (rakstīšanai) jūsu lokālajā vidē.
2. Testēšanas kompozītfunkcijas maiņa: Lai pārslēgtos starp testa funkcijām, 121. koda rindā jānomaina reģistrētā funkcija. Pēc noklusējuma ir iestatīta RAS kompozītfunkcija (composite_function_2). Lai izmantotu RMA kompozītfunkciju, rinda jānomaina uz:
```toolbox.register("composite", composite_function_1)```
4. Rezultātu faila nosaukums: Pēc testēšanas kompozītfunkcijas nomaiņas, neaizmirstiet 725. koda rindā attiecīgi pielāgot arī saglabājamā gala rezultātu faila nosaukumu, lai novērstu datu pārrakstīšanu.
5. Izpildes laika optimizācija: Ja kods ir vienreiz sekmīgi palaists un sintētiskā datu kopa jau ir ģenerēta, laika taupīšanas nolūkos visu 2. posmu (datu ģenerēšanas daļu) var aizkomentēt.

## Rezultātu vizualizēšana
Pēc tam, kad rezultāti ir ģenerēti ar kodu vai lejupielādēti no pievienotās Google Drive mapes, to grafiskai attēlošanai un analīzei var izmantot failu ```analytics.py```.
Lai pielāgotu vizualizāciju atbilstošajai testēšanas funkcijai, šajā skriptā ir jāveic izmaiņas failu ielādes rindās (18. un 19. rinda), tās attiecīgi atkomentējot vai aizkomentējot:
- RMA kompozītfunkcijas rezultātu apskatei: atkomentējiet 18. koda rindu (un pārliecinieties, ka 19. rinda ir aizkomentēta).
- RAS kompozītfunkcijas rezultātu apskatei: atkomentējiet 19. koda rindu (un pārliecinieties, ka 18. rinda ir aizkomentēta).
