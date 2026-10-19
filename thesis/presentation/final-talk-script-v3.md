# Final Talk v3 — Sprechskript (20 Minuten)

Sprechtext auf Deutsch zur gekürzten Fassung der Folien (`slides/talk_v3.py`, Szenen
`UseCase3`, `Data3`, `Generative3`, `Bands3`, `Models3`, `Summary3`). Die Folien selbst
sind englisch. Jeder **[Klick]** entspricht einem `next_slide()` (48 Klicks, 23 Folien).

**Warum diese Fassung kürzer ist.** Der Probelauf am 19.10. mit v2 dauerte 37:58 statt
19:30 (`2026-10-19_talk-timing-analysis.md`): 57 % mehr Wörter als im Skript, gesprochen
mit etwa 105 statt 130 Wörtern pro Minute. Dieses Skript hat rund 1.150 Wörter, also
etwa 11 Minuten bei 105 Wörtern pro Minute. Die Folienzeiten unten (zusammen 19:15)
lassen damit gut 8 Minuten für Animationen, das Video, Zeigen und freies Ergänzen; in
v2 kamen beim Sprechen 57 % Text dazu. Faustregel: **pro Folie eine Aussage**, Details
nur auf Nachfrage.

Alle Genauigkeitswerte sind mAP@[.5:.95] auf **echten** Fotos, außer es steht etwas
anderes dabei. Quellen: `slides/data.py` und Kapitel 3 und 4 des Manuskripts.

## Zeitplan

| Akt | Szene | Folien | Zeit | kumuliert |
|-----|-------|--------|------|-----------|
| 1 Anwendungsfall | `UseCase3` | 1–4 | 2:45 | 2:45 |
| 2 Daten | `Data3` | 5–9 | 3:30 | 6:15 |
| 3 Generierte Bilder | `Generative3` | 10–15 | 4:45 | 11:00 |
| 4 Klassen ergänzen | `Bands3` | 16–18 | 3:30 | 14:30 |
| 5 Modelle & Hardware | `Models3` | 19–22 | 3:45 | 18:15 |
| 6 Zusammenfassung | `Summary3` | 23 | 1:00 | 19:15 |

**Kontrollpunkte:** Folie 10 (Binturong) spätestens bei **6:30**, Folie 16 (Final
Dataset) bei **11:00**, Folie 19 (Three candidates) bei **14:30**. Liegt man hinten:
Folie 7 (Klassenliste) und die Punktwolke auf Folie 17 mit einem Satz durchklicken, die
Elefanten auf Folie 21 weglassen.

**Wo v2 die meiste Zeit verlor, und was hier anders ist:**
- *Data Quality:* nur die Filterkette nennen, nicht jede Stufe erklären.
- *Prompt:* nur die Länge. Der Versuchsaufbau (zwölf Arten, 1.200 Bilder) kommt erst auf
  Folie 14, die genau dafür da ist.
- *Knowledge Distillation:* das Ergebnis einmal sagen und einen Grund nennen.

---

## Akt 1 — Der Anwendungsfall (`UseCase3`)

### Folie 1 — Titel · 0:10

> Guten Tag zusammen. Meine Masterarbeit geht um Objekterkennung in Echtzeit auf
> eingebetteten Plattformen, Schritt für Schritt vom Anwendungsfall zum fertigen Modell.

**[Klick]**

### Folie 2 — Das Produkt · 0:45

> Der Anwendungsfall ist dieses Fernglas: das AX Visio von Swarovski Optik, ein Fernglas
> mit eingebauter KI.

**[Klick]** *(Video läuft, bleibt auf dem letzten Bild stehen)*

> So sieht das im Okular aus: Man drückt einen Knopf, die Kamera nimmt den Vogel auf, ein
> neuronales Netz auf dem Chip, einem Qualcomm QCS605, bestimmt ihn, und der Name wird ins
> Okular eingespiegelt. Alles auf dem Gerät, ohne Internet.

**[Klick]**

### Folie 3 — MammalID · 1:25

> Auf dem Wahlrad gibt es auch einen Modus für Säugetiere. *(auf das Symbol zeigen)*

**[Klick]** *(SpeciesNet-Blogpost im Hintergrund, darüber das Ensemble komplett)*

> Dahinter stehen zwei frei verfügbare Modelle: MegaDetector findet das Tier, hier mit 94
> Prozent, und SpeciesNet von Google bestimmt den Ausschnitt: Ozelot. Beide sind für
> Kamerafallen gebaut, oft nachts im Infrarot. Durch ein Fernglas sieht man anderes:
> Tageslicht, große Entfernung, zum Teil andere Tiere.

**[Klick]** *(MegaDetector wird grau, YOLOv5 erscheint)*

> Im Fernglas selbst läuft als Detektor statt MegaDetector ein kleineres YOLOv5-Modell.
> Meine Frage war: Kann ein Modell, das gezielt auf Säugetiere im Fernglas trainiert ist,
> das besser?

**[Klick]**

### Folie 4 — Three challenges · 0:30

> Drei Herausforderungen: die Daten, 225 Klassen, viele davon kaum fotografiert. Die
> Hardware, ein mobiler Chip, offline. Und die Geschwindigkeit, solange das Tier noch im
> Blick ist. Anfangen muss man bei den Daten.

**[Klick]**

---

## Akt 2 — Daten (`Data3`)

### Folie 5 — SpeciesNet · 0:30

> SpeciesNet kennt rund zweieinhalbtausend Arten, auch Vögel, Reptilien, sogar Spinnen.
> Für das Fernglas braucht man nur die Säugetiere, die man draußen wirklich sieht.

**[Klick]**

### Folie 6 — Igel · 0:25

> Und man muss festlegen, wie fein man unterscheidet.

**[Klick]**

> Siebzehn Igelarten, aus fünfzig Metern nicht zu unterscheiden, werden eine Klasse.

**[Klick]**

### Folie 7 — 225 Custom Classes · 0:15

> Am Ende stehen 225 Klassen, vom Product Owner bei Swarovski abgenommen.

**[Klick]**

### Folie 8 — Images per class · 1:30

> Dann braucht jede Klasse Bilder. Startpunkt war GBIF. Jeder Balken ist eine Klasse, die
> Achse logarithmisch: 114 Klassen haben weniger als hundert Bilder, 30 gar keine.

**[Klick]**

> Also vier weitere Quellen, nur mit kommerziell nutzbaren Lizenzen. iNaturalist liefert
> mit Abstand am meisten, rund 400.000 Bilder.

**[Klick]**

> Neu sortiert: Viele Klassen profitieren, aber 29 bleiben unter hundert Bildern.

**[Klick]**

### Folie 9 — Data Quality · 0:50

> Und nicht jedes Bild unter „Tier“ zeigt eines: eine ägyptische Wandmalerei, ein
> Maskottchen, eine Verbreitungskarte.

**[Klick]**

> Deshalb eine Filterkette bis zur Durchsicht von Hand. Zwei Stufen liefern gleich die
> Labels: MegaDetector die Bounding Boxes, SpeciesNet die Art. Das Grundproblem bleibt:
> Für viele Klassen gibt es zu wenige Fotos.

**[Klick]**

---

## Akt 3 — Generierte Bilder (`Generative3`)

### Folie 10 — Synthetic images? – Binturong · 0:45

> Die naheliegende Idee: Bilder generieren. Ein Binturong, links echte Fotos.

**[Klick]**

> Drei Generatoren, die lokal auf einer GPU laufen. Das sind eher Bären.

**[Klick]**

> Gemini, über eine API: deutlich besser. Damit habe ich den Datensatz erzeugt.

**[Klick]**

### Folie 11 — Prompt · 0:40

> Damit die Details stimmen, sind die Prompts lang: Artbeschreibung, Lebensraum, Szene,
> Erkennungsmerkmale. Hier über viertausend Wörter für einen Löwen.

**[Klick]**

> Lokale Generatoren lesen aber nur 512, 256 oder gar 75 Tokens. Wichtig ist nur der
> Unterschied in der Länge.

**[Klick]**

### Folie 12 — Synthetischer Datensatz · 0:15

> So entstanden 23.850 Bilder, hier Grévyzebras, für rund 785 Dollar.

**[Klick]**

### Folie 13 — Drei Zebras · 0:30

> Wie genau das sein muss, zeigen Zebras: Grévy-, Berg- und Steppenzebra unterscheiden
> sich nur in den Streifen.

**[Klick]**

> Alle drei von Gemini.

**[Klick]**

### Folie 14 — Training Images Comparison (zwölf Arten) · 0:40

> Aber sieht gut aus heißt nicht, dass ein Detektor daraus lernt. Deshalb ein Vergleich:
> diese zwölf Arten, von häufig bis selten, je 100 Bilder, also 1.200 pro Generator. Gleicher
> Detektor, gleiches Training, bewertet auf echten Fotos.

**[Klick]**

### Folie 15 — Training Images Comparison (Rangliste) · 1:55

> Blau die API-Modelle, grau die lokalen. Alle API-Modelle liegen vor allen lokalen.
> Platz eins ist gpt-image-2 in der günstigsten Stufe, 1,7 Cent pro Bild, 13,4 Prozent.
> Gemini, mit dem ich den Datensatz erzeugt habe, liegt auf Platz drei. Die lokalen lesen
> nur 75 bis 512 Tokens.

**[Klick]** *(Kurzprompt-Zeilen und ihre Originale grün)*

> Liegt es am Modell oder am Prompt? Dieselben zwei API-Modelle, grün markiert, mit
> höchstens 75 Tokens: Sie verlieren fast die Hälfte, gpt-image-2 fällt von 13,4 auf 7,4
> Prozent. Es zählt also beides: wie viel der Generator über das Tier erfährt, und welcher
> Generator zeichnet.

**[Klick]**

---

## Akt 4 — Die Klassen ergänzen (`Bands3`)

### Folie 16 — Final Dataset · 1:10

> Der fertige Datensatz. Jeder Balken eine Klasse, nach Anzahl echter Fotos sortiert.

**[Klick]**

> Danach vier Bänder: A die 50 Klassen mit den wenigsten Fotos, D die gut dokumentierten.

**[Klick]**

> Band A trainiert nur mit synthetischen Bildern, B halb und halb, C nur mit echten,
> jeweils 200 pro Klasse. So kann man echte und künstliche Bilder direkt vergleichen.
> D hat bis zu 1.500 echte Bilder.

**[Klick]**

> Dazu 50 synthetische Testbilder pro Klasse, in gelb.

**[Klick]** *(nur die echten Testbilder bleiben hell, ihr Etikett wandert nach oben)*

> Gemessen wird ab jetzt nur auf den echten Testfotos, in pink.

**[Klick]**

### Folie 17 — Recognition Results · 1:15

> Das Ergebnis auf echten Testfotos: A, nur synthetisch, 20 Prozent. B, halb synthetisch, 51.
> C, nur echt, 54. D 66 Prozent. Die Hälfte durch synthetische Bilder zu ersetzen kostet
> wenig, alles zu ersetzen funktioniert nicht.

**[Klick]**

> Dahinter viel Streuung, jeder Punkt eine Klasse.

**[Klick]**

> In Band A schafft die Streifenhyäne 48 Prozent, die Mangaben fast null. Meine
> Hypothese: Einheitlich aussehende Arten lassen sich gut generieren, vielfältige nicht.

**[Klick]** *(der Datensatz noch einmal, die synthetischen Testbilder hell; die beiden
Test-Etiketten wandern zu den Zebras)*

### Folie 18 — Test Data · 0:45

> Warum nur echte Fotos? Oben synthetische Testbilder, unten echte: Gegenlicht, Gestrüpp,
> Farbstich.

**[Klick]**

> Auf synthetischen Testbildern sind alle Modelle viel besser, in Band A 93 statt 20
> Prozent. Sie sind schlicht leichter. Deshalb zählen nur die echten Fotos.

**[Klick]**

---

## Akt 5 — Modelle und Hardware (`Models3`)

### Folie 19 — Three candidates · 0:20

> Drei Kandidaten, Kreisfläche gleich Parameterzahl: MegaDetector plus SpeciesNet mit 193
> Millionen, YOLOv5s mit 7,6 Millionen, YOLO26n mit 2,7 Millionen.

**[Klick]**

### Folie 20 — Accuracy on real photos (mAP) · 1:30

> Artgenauigkeit auf echten Fotos. MD plus SN, off the shelf: 44,5 Prozent. Mit
> feinjustiertem SpeciesNet: 59,9. YOLO26n: 52,9, besser als das Original-Ensemble und
> besser als YOLOv5s.

**[Klick]**

> Darüber, dünn: Wird das Tier überhaupt gefunden? Das klappt viel besser. Finden ist
> leichter als Benennen.

**[Klick]**

> Auf einem Raspberry Pi als Stellvertreter für den Chip braucht das Ensemble 35 Sekunden
> pro Bild, YOLO26n 0,42 Sekunden.

**[Klick]**

> 83-mal schneller, 71-mal kleiner, nur 7 Prozentpunkte weniger. Echtzeit fehlt noch, dafür
> braucht es Quantisierung und den Beschleuniger des Chips.

**[Klick]**

### Folie 21 — Similar Species · 1:05

> Ein Teil der Lücke sind ähnliche Arten. Die drei Zebras als echte Fotos.

**[Klick]**

> 40 Prozent der Grévyzebras werden als Steppenzebra benannt, umgekehrt fast nie. Die
> Fehler laufen zur häufigeren Art. Über alle drei Zebras sind es knapp 10 Prozent.

**[Klick]**

> Extremfall Elefanten: 79 Prozent der Asiatischen werden Afrikanische, umgekehrt 0,2
> Prozent. Insgesamt wird jeder zweite Elefant falsch benannt.

**[Klick]**

### Folie 22 — Knowledge Distillation · 0:50

> Kann das große Ensemble das kleine Modell verbessern, per Knowledge Distillation?

**[Klick]**

> Der distillierte Schüler, unten: 47,9 Prozent, schlechter als direkt feinjustiert.

**[Klick]**

> Direktes Training gewinnt um 5 Prozentpunkte. Der Grund: Die Labels kamen schon von
> SpeciesNet, der Lehrer hat dem Schüler nichts Neues beizubringen.

**[Klick]**

---

## Akt 6 — Zusammenfassung (`Summary3`)

### Folie 23 — Summary · 1:00

*(Die sechs Kacheln erscheinen von selbst nacheinander, in gut drei Sekunden. Der Text
läuft über sie hinweg, ohne Klick.)*

> Zusammengefasst: Erstens MammalID, Tiere direkt im Fernglas bestimmen. Zweitens die
> Datensammlung: Für viele Klassen gibt es zu wenige echte Bilder. Drittens die
> Bildgenerierung, um diese Lücke zu füllen. Viertens der Generatorvergleich: API-Modelle
> mit langem Prompt gewinnen. Fünftens das Training: Synthetische Bilder strecken echte
> Daten, ersetzen sie aber nicht. Und sechstens der Modellvergleich: ein kleines, schnelles
> Modell nah am großen Ensemble. Nächster Schritt ist die Optimierung für den Chip.
>
> Vielen Dank, ich freue mich auf Ihre Fragen.

*(Letzte Folie, sie bleibt für die Fragen stehen.)*

---

## Vorbereitung auf Fragen

Was auf den Folien von v2 stand und hier fehlt, gehört in die Antworten.

| Frage | Antwort |
|-------|---------|
| Ist SpeciesNet wirklich nicht auf Nashörnern trainiert? | Von 1.510 Nashornfotos hat SpeciesNet nur 40 (2,6 %) richtig zugeordnet. Eine Rückfallregel in der Filterkette hat rund 14.500 Bilder aus 17 Klassen gerettet. |
| Wie groß ist das Ensemble? | Rund 193 Millionen Parameter (Folie 19). |
| Welche Stufen hat die Filterkette? | Metadaten, Bildqualität (Unschärfe, weiße Pixel), MegaDetector, ein Vision-Language-Modell als zweite Meinung, SpeciesNet. Knapp 18.000 unklare Fälle von Hand geprüft. |
| Und seltene Arten? | Die sechs seltenen Arten im Vergleich liegen bei jedem Generator unter 7 % AP. Beispiel Saiga: Die lokalen Generatoren zeichnen eine gewöhnliche Gazelle ohne die rüsselartige Nase. Ein Generator zeichnet nur, was er oft genug gesehen hat. |
| Warum Raspberry Pi statt QCS605? | Der Hexagon-DSP hängt an einer Toolchain von 2018 mit schlecht dokumentierter Operator-Unterstützung. Der Pi 400 ist etwa 10–15 % langsamer als die Kryo-CPU des QCS605. |
| Wie kommt man auf Echtzeit? | Quantisierung, idealerweise quantisierungsbewusstes Training, dann der Integer-Pfad des Chips. Laut Hersteller schafft YOLO26n auf einem Raspberry Pi 5 etwa 68 ms, selbst gemessen nicht. |
| Detection-mAP-Werte? | Ensemble rund 95 % (auch wegen der Referenzboxen), YOLO26n 78 %, YOLOv5s 75 %. |
| Warum ist das Ensemble eine Obergrenze? | MegaDetector hat die Ground-Truth-Boxen erzeugt. Beim Ensemble ist mAP gleich mAP50, was kein anderer Detektor erreicht. |
| Woher kommen die Gruppenwerte auf Folie 21 (Zebras 9,6 %, Elefanten 52,3 %)? | YOLO26n, echte Testfotos: von den richtig lokalisierten und in die richtige Gruppe sortierten Tieren der Anteil mit der falschen Art (Zebras 147 von 1.538, Elefanten 1.547 von 2.957). Über alle 30 Gruppen sind es 19,7 % (5.528 von 28.000), auf dem gemischten Testset 19,3 % (Manuskript). |
| Wie sind die Pfeil-Werte auf Folie 21 berechnet? | Pro Art: von allen richtig lokalisierten (IoU ≥ 0,5) Tieren dieser Art in der richtigen Gruppe der Anteil mit der falschen Art, auf den echten Testfotos. |
| Warum stehen auf Folie 17 andere Bandwerte als in der Arbeit (30,0 / 58,8 / 59,1 / 74,2 %)? | Die Folie zeigt den Mittelwert der AP pro Klasse aus der gemeinsamen Auswertung aller 225 Klassen. Die Tabelle der Arbeit wertet jedes Band nur auf seinen eigenen Bildern aus. |
| Warum nur ein Lauf pro Modell? | Ein Lauf dauerte auf einer A40 21 Stunden bis 8,8 Tage. Unterschiede unter etwa 3 Prozentpunkten sind nicht belastbar. |
| Wurden die synthetischen Bilder automatisch annotiert? | Ja, mit MegaDetector. Im Generatorvergleich betrifft das alle gleich, die Rangfolge bleibt gültig. |
| Warum keine NanoDet, PicoDet usw.? | Aus Zeitgründen; die Detektorfamilie war durch das Produkt vorgegeben. |
| Lizenzen der Generatoren? | Nur zwei lokale Modelle haben wirklich freie Lizenzen. Die FLUX-API verbietet Training auf ihren Ausgaben. |
| Kosten der Bildgenerierung? | 23.850 Bilder, rund 785 $ zum Listenpreis. Für 1.500 Bilder pro Klasse etwa vierzehnmal so viel. |
